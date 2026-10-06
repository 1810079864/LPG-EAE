
"""
LPG-EAE: Light-Prompt and Graph-enhanced Event Argument Extraction

Architecture Overview:
  1. Semantic Dependency Graph  — trigger-centric edges weighted by syntactic role importance
                                   (subj/obj/pred >> modifier/complement >> adjunct)
                                   + coreference super-node merging:
                                     coreferent mentions are collapsed onto their
                                     representative token; all edges that originally
                                     pointed to a non-representative token are
                                     redirected to the representative token.
  2. Co-event Trigger Graph     — triggers in the same event-group connected by
                                   inverse-distance; different groups NOT connected.
  3. Dual-Stream GAT            — dependency/coreference and trigger graphs are
                                   encoded once per document.
  4. Trigger Query Enhancement  — shared trigger topology augments each role
                                   query; no document compression is performed.
  5. Full-context Span Pointer  — boundaries are scored on the original H_x.
"""
import time
import torch
import torch.nn as nn
from collections import defaultdict
from transformers import RobertaModel, RobertaPreTrainedModel
from utils import hungarian_matcher, get_best_span, get_best_span_simple


class LightPromptCrossAttentionLayer(nn.Module):
    """A small prompt-query -> document cross-attention refinement block."""

    def __init__(self, hidden_size, num_heads, ffn_ratio=2.0, dropout=0.1):
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_size)
        self.context_norm = nn.LayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.attn_dropout = nn.Dropout(dropout)
        inner_size = int(hidden_size * ffn_ratio)
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, inner_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_size, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, prompt, document, prompt_mask, document_mask):
        query = self.query_norm(prompt)
        context = self.context_norm(document)
        attended, _ = self.cross_attn(
            query=query,
            key=context,
            value=context,
            key_padding_mask=~document_mask.bool(),
            need_weights=False,
        )
        prompt = prompt + self.attn_dropout(attended)
        prompt = prompt + self.ffn(self.ffn_norm(prompt))
        return prompt * prompt_mask.unsqueeze(-1).to(prompt.dtype)


def graph_to_edge_list(graph, num_nodes: int, device: torch.device):
    """Accept offline sparse edges; dense tensors remain backward-compatible."""
    if isinstance(graph, (tuple, list)) and len(graph) >= 2:
        edge_index, weights = graph[:2]
        edge_index = edge_index.to(device=device, dtype=torch.long, non_blocking=True)
        weights = weights.to(device=device, dtype=torch.float, non_blocking=True)
        if edge_index.numel() == 0:
            empty_idx = torch.empty(0, dtype=torch.long, device=device)
            empty_weight = torch.empty(0, dtype=torch.float, device=device)
            return empty_idx, empty_idx, empty_weight
        return edge_index[0], edge_index[1], weights

    adj_w = graph
    edge_idx = adj_w.nonzero(as_tuple=False)
    if edge_idx.numel() == 0:
        idx = torch.arange(num_nodes, device=device)
        return idx, idx, torch.ones(num_nodes, device=device)
    src     = edge_idx[:, 0]
    dst     = edge_idx[:, 1]
    weights = adj_w[src, dst]
    return src, dst, weights


# ---------------------------------------------------------------------------
# Edge-Weighted GAT Layer  ── sparse O(E·d)
# ---------------------------------------------------------------------------
class EdgeWeightedGATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int,
                 dropout: float = 0.1, negative_slope: float = 0.2):
        super().__init__()
        self.out_dim = out_dim
        self.W_q     = nn.Linear(in_dim, out_dim, bias=False)
        self.W_k     = nn.Linear(in_dim, out_dim, bias=False)
        self.attn    = nn.Linear(2 * out_dim, 1, bias=False)
        self.lrelu   = nn.LeakyReLU(negative_slope)
        self.drop    = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, graph) -> torch.Tensor:
        N = h.size(0)
        q = self.W_q(h)
        k = self.W_k(h)

        src, dst, w = graph_to_edge_list(graph, N, h.device)
        E = src.size(0)

        q_dst = q[dst]
        k_src = k[src]
        e     = self.lrelu(
            self.attn(torch.cat([q_dst, k_src], dim=-1)).squeeze(-1)
        )
        e = e * w

        # scatter softmax（数值稳定）
        e_max = torch.full((N,), float('-inf'), device=h.device)
        e_max.scatter_reduce_(0, dst, e, reduce='amax', include_self=True)
        e_exp = torch.exp(e - e_max[dst])
        e_sum = torch.zeros(N, device=h.device)
        e_sum.scatter_add_(0, dst, e_exp)
        e_sum = e_sum.clamp(min=1e-9)

        alpha = self.drop(e_exp / e_sum[dst])

        out = torch.zeros(N, self.out_dim, device=h.device)
        out.scatter_add_(
            0,
            dst.unsqueeze(1).expand(E, self.out_dim),
            alpha.unsqueeze(1) * k_src,
        )
        return out


# ---------------------------------------------------------------------------
# Multi-layer GAT stream
# ---------------------------------------------------------------------------
class GATStream(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EdgeWeightedGATLayer(hidden_dim, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj_w: torch.Tensor) -> torch.Tensor:
        for layer, norm in zip(self.layers, self.norms):
            h = norm(h + self.drop(layer(h, adj_w)))
        return h


# ---------------------------------------------------------------------------
# Dual-Stream GAT
# ---------------------------------------------------------------------------
class DualStreamGAT(nn.Module):
    """
    Stream-A → 图1（句法依存图 + 共指超级节点合并）
    Stream-B → 意群内触发词图
    """
    def __init__(self, hidden_dim: int, num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.stream_a = GATStream(hidden_dim, num_layers, dropout)
        self.stream_b = GATStream(hidden_dim, num_layers, dropout)

        self.gate_proj = nn.Linear(2 * hidden_dim, hidden_dim)
        self.out_proj  = nn.Linear(hidden_dim, hidden_dim)
        self.norm      = nn.LayerNorm(hidden_dim)

    def forward(self,
                h:       torch.Tensor,
                adj_dep: torch.Tensor,
                adj_trg: torch.Tensor,
                use_dependency_coref: bool = True,
                use_trigger: bool = True) -> torch.Tensor:
        if use_dependency_coref and use_trigger:
            out_a = self.stream_a(h, adj_dep)
            out_b = self.stream_b(h, adj_trg)
            gate = torch.sigmoid(
                self.gate_proj(torch.cat([out_a, out_b], dim=-1))
            )
            fused = gate * out_a + (1 - gate) * out_b
        elif use_dependency_coref:
            fused = self.stream_a(h, adj_dep)
        elif use_trigger:
            fused = self.stream_b(h, adj_trg)
        else:
            return h
        return self.norm(h + self.out_proj(fused))


# ---------------------------------------------------------------------------
# LPG-EAE model
# ---------------------------------------------------------------------------
class LPGEAE(RobertaPreTrainedModel):
    """
    LPG-EAE with Light Prompt refinement and Dual-Stream GAT.

    forward() 新增参数：
        dep_heads_batch     : list[list[int]]
        dep_rels_batch      : list[list[str]]
        event_groups_batch  : list[list[int]]
        coref_clusters_batch: list[list[list[int]]]  ← 新增
            每个样本的共指簇列表，格式：
            [ [sw_rep_0, sw_rep_1, ...],   # 簇0：各 mention 的代表 subword
              [sw_rep_0, sw_rep_1, ...],   # 簇1
              ... ]
            簇内第一个元素为超级节点（代表词），其余元素被重定向到它。
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.decode_layer_start = config.encoder_layers

        self.roberta = RobertaModel(config)

        light_layers = int(getattr(config, 'light_prompt_layers', 1))
        if not getattr(config, 'use_light_prompt', True):
            light_layers = 0
        light_heads = int(getattr(config, 'light_prompt_heads', 8))
        if config.hidden_size % light_heads != 0:
            raise ValueError(
                'hidden_size={} must be divisible by light_prompt_heads={}'.format(
                    config.hidden_size, light_heads
                )
            )
        self.light_prompt_layers = nn.ModuleList([
            LightPromptCrossAttentionLayer(
                hidden_size=config.hidden_size,
                num_heads=light_heads,
                ffn_ratio=float(getattr(config, 'light_prompt_ffn_ratio', 2.0)),
                dropout=float(getattr(config, 'light_prompt_dropout', 0.1)),
            )
            for _ in range(light_layers)
        ])

        self.w_prompt_start = nn.Parameter(torch.rand(config.hidden_size))
        self.w_prompt_end   = nn.Parameter(torch.rand(config.hidden_size))

        gat_layers  = getattr(config, 'gat_num_layers', 2)
        gat_dropout = getattr(config, 'gat_dropout', 0.1)
        self.ds_gat = DualStreamGAT(config.hidden_size, gat_layers, gat_dropout)

        self.loss_fct = nn.CrossEntropyLoss(reduction='sum')

        # Component-level training-forward profiler. CUDA Events avoid a
        # synchronization at every boundary; timings are resolved once at the
        # end of each forward pass.
        self._profile_totals_ms = defaultdict(float)
        self._profile_call_counts = defaultdict(int)
        self._profile_forward_count = 0
        self._profile_pending = []

    def _profile_start(self, name: str):
        if not self.training:
            return None
        if torch.cuda.is_available() and next(self.parameters()).is_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            return name, start, end, None
        return name, None, None, time.perf_counter()

    def _profile_stop(self, token):
        if token is None:
            return
        name, start, end, cpu_start = token
        self._profile_call_counts[name] += 1
        if start is not None:
            end.record()
            self._profile_pending.append((name, start, end))
        else:
            self._profile_totals_ms[name] += (
                time.perf_counter() - cpu_start
            ) * 1000.0

    def _profile_finish_forward(self):
        if not self.training:
            return
        if self._profile_pending:
            self._profile_pending[-1][2].synchronize()
            for name, start, end in self._profile_pending:
                self._profile_totals_ms[name] += start.elapsed_time(end)
            self._profile_pending.clear()
        self._profile_forward_count += 1

    def get_component_profile(self):
        return {
            'forward_count': self._profile_forward_count,
            'totals_ms': dict(self._profile_totals_ms),
            'call_counts': dict(self._profile_call_counts),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_trigger_positions(self, event_trigger: list) -> list:
        """
        支持多种触发词格式：
          - int
          - [start, end] 或 [start, end, "text"]  → range(start, end)
          - dict with 'tok_pos'
        """
        positions = []
        for t in event_trigger:
            if isinstance(t, int):
                positions.append(t)
            elif isinstance(t, (list, tuple)):
                int_vals = [v for v in t if isinstance(v, int)]
                if len(int_vals) >= 2:
                    positions.extend(range(int_vals[0], int_vals[1]))
                elif len(int_vals) == 1:
                    positions.append(int_vals[0])
            elif isinstance(t, dict):
                pos = t.get('tok_pos', t.get('position', None))
                if pos is not None:
                    if isinstance(pos, int):
                        positions.append(pos)
                    else:
                        positions.extend(pos)
        return positions

    def _enhance_query(
        self, prompt_query_sub, gat_output, trigger_positions, seq_len
    ):
        """Add only the trigger-centred shared graph representation to a role."""
        if not getattr(self.config, 'use_dual_stream_gat', True):
            return prompt_query_sub
        if (
            not getattr(self.config, 'use_dependency_coref_graph', True)
            and not getattr(self.config, 'use_trigger_graph', True)
        ):
            return prompt_query_sub
        valid = [p for p in trigger_positions if 0 <= p < seq_len]
        if not valid:
            return prompt_query_sub
        trigger_repr = gat_output[valid].mean(0, keepdim=True)
        q_norm_sq = (
            torch.sum(prompt_query_sub * prompt_query_sub, dim=-1, keepdim=True)
            + 1e-8
        )
        projection_scalar = (
            torch.sum(trigger_repr * prompt_query_sub, dim=-1, keepdim=True)
            / q_norm_sq
        )
        trigger_orthogonal = trigger_repr - projection_scalar * prompt_query_sub
        fuse_weight = float(getattr(self.config, 'gat_query_fuse_weight', 0.2))
        return prompt_query_sub + fuse_weight * trigger_orthogonal

    @staticmethod
    def forward(
        self,
        all_ids=None,
        all_mask_ids=None,
        dec_prompt_ids=None,
        dec_prompt_mask_ids=None,
        arg_joint_prompts=None,
        target_info=None,
        old_tok_to_new_tok_indexs=None,
        event_triggers=None,
        dep_heads_batch=None,
        dep_rels_batch=None,
        event_groups_batch=None,
        coref_clusters_batch=None,   # ← 新增
        coref_logits_batch=None,
        precomputed_dep_graphs=None,
        precomputed_trigger_graphs=None,
        enc_input_ids=None,
        enc_mask_ids=None,
        enc_attention_mask=None,
        arg_list=None,
        **kwargs,
    ):
        device = all_ids.device

        if precomputed_dep_graphs is None or precomputed_trigger_graphs is None:
            raise RuntimeError(
                'Offline graphs are required. Rebuild the dataloader so '
                'graph_cache/<split>_graph.json is generated.'
            )
        # ── 1. RoBERTa 编码上下文 ──────────────────────────────────────
        profile_token = self._profile_start('roberta_joint_encoding')
        context_outputs_ = self.roberta(
            input_ids=all_ids,
            attention_mask=all_mask_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        enc_outputs     = context_outputs_.hidden_states
        decoder_context = enc_outputs[self.decode_layer_start]

        encoded_outputs = (enc_outputs[-1]
                           if self.config.context_representation == 'decoder'
                           else decoder_context)
        self._profile_stop(profile_token)

        context_outputs = encoded_outputs

        # ── 2. 从联合输出提取 Prompt hidden，进行轻量文档交叉注意力 ────
        # all_ids 的布局是 [document; multi-event prompts; padding]。
        # 不再对 dec_prompt_ids 调用第二次完整 RoBERTa。
        profile_token = self._profile_start('light_prompt_cross_attention')
        batch_size, prompt_width = dec_prompt_mask_ids.shape
        hidden_size = context_outputs.size(-1)
        decoder_prompt_outputs = context_outputs.new_zeros(
            batch_size, prompt_width, hidden_size
        )
        document_mask = torch.zeros_like(all_mask_ids, dtype=torch.bool)
        for batch_idx in range(batch_size):
            document_length = int(enc_mask_ids[batch_idx].sum().item())
            prompt_length = int(dec_prompt_mask_ids[batch_idx].sum().item())
            joint_length = int(all_mask_ids[batch_idx].sum().item())
            available_prompt = max(
                0,
                min(prompt_length, joint_length - document_length,
                    context_outputs.size(1) - document_length),
            )
            if available_prompt:
                decoder_prompt_outputs[batch_idx, :available_prompt] = \
                    context_outputs[
                        batch_idx,
                        document_length:document_length + available_prompt,
                    ]
            document_mask[batch_idx, :document_length] = True

        prompt_mask = dec_prompt_mask_ids.bool()
        for light_layer in self.light_prompt_layers:
            decoder_prompt_outputs = light_layer(
                decoder_prompt_outputs,
                context_outputs,
                prompt_mask,
                document_mask,
            )
        self._profile_stop(profile_token)

        logit_lists = []
        total_loss  = 0.0

        # ── 3. 逐样本处理 ──────────────────────────────────────────────
        for i, (context_output, decoder_prompt_output, arg_joint_prompt,
                old_tok_to_new_tok_index, event_trigger) in enumerate(zip(
            context_outputs, decoder_prompt_outputs, arg_joint_prompts,
            old_tok_to_new_tok_indexs, event_triggers,
        )):
            seq_len = context_output.size(0)
            # 触发词位置
            trigger_pos_list = [
                self._get_trigger_positions([ev]) for ev in event_trigger
            ]

            # ── 图1：句法依存图 + 共指超级节点合并（只调用一次）────────
            # if not _GRAPH_PRINT_DONE:
            #     print(f"触发词：{dep_rels}")
            profile_token = self._profile_start('offline_graph_load')
            dep_edge_index, dep_edge_weight, dep_edge_types = precomputed_dep_graphs[i]
            trg_edge_index, trg_edge_weight, trg_edge_types = precomputed_trigger_graphs[i]
            adj_dep = (
                dep_edge_index.to(device, non_blocking=True),
                dep_edge_weight.to(device, non_blocking=True),
                dep_edge_types.to(device, non_blocking=True),
            )
            adj_trg = (
                trg_edge_index.to(device, non_blocking=True),
                trg_edge_weight.to(device, non_blocking=True),
                trg_edge_types.to(device, non_blocking=True),
            )
            self._profile_stop(profile_token)

            profile_token = self._profile_start('dependency_trigger_gat')
            if getattr(self.config, 'use_dual_stream_gat', True):
                gat_output = self.ds_gat(
                    context_output,
                    adj_dep,
                    adj_trg,
                    use_dependency_coref=getattr(
                        self.config, 'use_dependency_coref_graph', True
                    ),
                    use_trigger=getattr(
                        self.config, 'use_trigger_graph', True
                    ),
                )
            else:
                gat_output = context_output
            self._profile_stop(profile_token)

            # ── 4. 逐事件论元抽取 ──────────────────────────────────────
            batch_loss, cnt, list_output = [], 0, []

            for ii in range(len(event_trigger)):
                trig_pos = trigger_pos_list[ii]
                output   = {}
                # Roles are decoded independently, in their original prompt order.
                for arg_role, slots in arg_joint_prompt[ii].items():
                    s_logits_list = []
                    e_logits_list = []

                    for (p_start, p_end,
                         p_start_off, p_end_off) in zip(
                        slots['tok_s'],     slots['tok_e'],
                        slots['tok_s_off'], slots['tok_e_off'],
                    ):
                        # tok_s/tok_e are offsets in the concatenated
                        # multi-event prompt for both WikiEvents and RAMS.
                        # Both datasets must construct role queries from the
                        # same Light-Prompt-refined representation.
                        raw_sub = decoder_prompt_output[p_start:p_end]

                        if raw_sub.shape[0] == 0:
                            raise RuntimeError(
                                'Empty Light Prompt role span: dataset={}, '
                                'prompt_span=({}, {}), prompt_length={}'.format(
                                    self.config.dataset, p_start, p_end,
                                    decoder_prompt_output.size(0),
                                )
                            )

                        prompt_query_sub = raw_sub.mean(0).unsqueeze(0)
                        prompt_query_sub = self._enhance_query(
                            prompt_query_sub, gat_output, trig_pos, seq_len
                        )
                        profile_token = self._profile_start('span_pointer_prediction')
                        start_q = (prompt_query_sub * self.w_prompt_start).unsqueeze(-1)
                        end_q   = (prompt_query_sub * self.w_prompt_end).unsqueeze(-1)

                        start_logits = torch.bmm(
                            context_output.unsqueeze(0), start_q
                        ).squeeze()
                        end_logits = torch.bmm(
                            context_output.unsqueeze(0), end_q
                        ).squeeze()
                        self._profile_stop(profile_token)




                        s_logits_list.append(start_logits)
                        e_logits_list.append(end_logits)

                    output[arg_role] = [s_logits_list, e_logits_list]

                    if self.training:
                        target = target_info[i][ii][arg_role]
                        predicted_spans = []
                        for s_l, e_l in zip(s_logits_list, e_logits_list):
                            if self.config.matching_method_train == 'accurate':
                                predicted_spans.append(
                                    get_best_span(s_l, e_l,
                                                  old_tok_to_new_tok_index,
                                                  self.config.max_span_length)
                                )
                            else:
                                predicted_spans.append(
                                    get_best_span_simple(s_l, e_l)
                                )

                        target_spans = [
                            [s, e] for s, e in zip(
                                target["span_s"], target["span_e"]
                            )
                        ]
                        if len(target_spans) < len(predicted_spans):
                            pad = len(predicted_spans) - len(target_spans)
                            target_spans     += [[0, 0]] * pad
                            target["span_s"] += [0] * pad
                            target["span_e"] += [0] * pad

                        if self.config.bipartite:
                            idx_preds, idx_targets = hungarian_matcher(
                                predicted_spans, target_spans
                            )
                        else:
                            idx_preds   = torch.arange(
                                len(predicted_spans), dtype=torch.int64
                            )
                            idx_targets = torch.arange(
                                min(len(target_spans), len(predicted_spans)),
                                dtype=torch.int64,
                            )

                        cnt    += len(idx_preds)
                        s_loss  = self.loss_fct(
                            torch.stack(s_logits_list)[idx_preds],
                            torch.LongTensor(
                                target["span_s"]
                            ).to(device)[idx_targets],
                        )
                        e_loss  = self.loss_fct(
                            torch.stack(e_logits_list)[idx_preds],
                            torch.LongTensor(
                                target["span_e"]
                            ).to(device)[idx_targets],
                        )
                        batch_loss.append((s_loss + e_loss) / 2)

                list_output.append(output)

            logit_lists.append(list_output)
            if self.training and batch_loss:
                total_loss += torch.stack(batch_loss).sum() / cnt

        self._profile_finish_forward()
        if self.training:
            span_loss = total_loss / len(context_outputs)
            return span_loss, logit_lists
        else:
            return [], logit_lists
