import os
import json
import math
import hashlib
import logging
import re
import torch
from copy import deepcopy
from torch.utils.data import Dataset
from processors.processor_base import DSET_processor, SyntaxProvider
from graph_relations import EDGE_TYPE_NAMES, EDGE_TYPE_TO_ID, dependency_type
from utils import EXTERNAL_TOKENS, _PREDEFINED_QUERY_TEMPLATE

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# InputFeatures
# ─────────────────────────────────────────────────────────────────────────────
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, example_id, feature_id,
                 event_type, event_trigger,
                 enc_text, enc_input_ids, enc_mask_ids, all_ids, all_mask_ids,
                 dec_prompt_text, dec_prompt_ids, dec_prompt_mask_ids,
                 arg_quries, arg_joint_prompt, target_info, enc_attention_mask,
                 old_tok_to_new_tok_index=None, full_text=None, arg_list=None,
                 dep_heads=None,
                 dep_rels=None,
                 event_groups=None,
                 coref_clusters=None,   # ← 新增：list[list[int]] subword级共指簇
                 coref_logits=None,
                 precomputed_dep_graph=None,
                 precomputed_trigger_graph=None,
                 ):

        self.example_id    = example_id
        self.feature_id    = feature_id
        self.event_type    = event_type
        self.event_trigger = event_trigger
        self.num_events    = len(event_trigger)

        self.enc_text           = enc_text
        self.enc_input_ids      = enc_input_ids
        self.enc_mask_ids       = enc_mask_ids
        self.all_ids            = all_ids
        self.all_mask_ids       = all_mask_ids
        self.enc_attention_mask = enc_attention_mask

        self.dec_prompt_texts    = dec_prompt_text
        self.dec_prompt_ids      = dec_prompt_ids
        self.dec_prompt_mask_ids = dec_prompt_mask_ids

        if arg_quries is not None:
            self.dec_arg_query_ids       = [v[0] for k, v in arg_quries.items()]
            self.dec_arg_query_masks     = [v[1] for k, v in arg_quries.items()]
            self.dec_arg_start_positions = [v[2] for k, v in arg_quries.items()]
            self.dec_arg_end_positions   = [v[3] for k, v in arg_quries.items()]
            self.start_position_ids      = [v['span_s'] for k, v in target_info.items()]
            self.end_position_ids        = [v['span_e'] for k, v in target_info.items()]
        else:
            self.dec_arg_query_ids       = None
            self.dec_arg_query_masks     = None

        self.arg_joint_prompt         = arg_joint_prompt
        self.target_info              = target_info
        self.old_tok_to_new_tok_index  = old_tok_to_new_tok_index
        self.full_text                = full_text
        self.arg_list                 = arg_list

        # ── 图相关字段 ────────────────────────────────────────────────
        self.dep_heads      = dep_heads      if dep_heads      is not None else []
        self.dep_rels       = dep_rels       if dep_rels       is not None else []
        self.event_groups   = event_groups   if event_groups   is not None else []
        # coref_clusters：subword 粒度共指簇
        # 格式：[[sw_rep_0, sw_rep_1, ...], [sw_rep_0, ...], ...]
        # 每个子列表为同一实体的各 mention 代表 subword，第一个为超级节点
        # self.coref_clusters = coref_clusters if coref_clusters is not None else []
        self.coref_clusters  = coref_clusters
        self.coref_logits    = coref_logits if coref_logits is not None else []
        self.precomputed_dep_graph = precomputed_dep_graph
        self.precomputed_trigger_graph = precomputed_trigger_graph

    def find_idx(self, target, lst):
        for i, item in enumerate(lst):
            if item == target:
                return i

    def init_pred(self):
        self.pred_dict_tok  = [dict() for _ in range(self.num_events)]
        self.pred_dict_word = [dict() for _ in range(self.num_events)]

    def add_pred(self, role, span, event_index):
        pred_dict_tok  = self.pred_dict_tok[event_index]
        pred_dict_word = self.pred_dict_word[event_index]
        if role not in pred_dict_tok:
            pred_dict_tok[role] = list()
        if span not in pred_dict_tok[role]:
            pred_dict_tok[role].append(span)
            if span != (0, 0):
                if role not in pred_dict_word:
                    pred_dict_word[role] = list()
                word_span = self.get_word_span(span)
                if word_span not in pred_dict_word[role]:
                    pred_dict_word[role].append(word_span)

    def set_gt(self):
        self.gt_dict_tok = [dict() for _ in range(self.num_events)]
        for i, target_info in enumerate(self.target_info):
            for k, v in target_info.items():
                self.gt_dict_tok[i][k] = [
                    (s, e) for (s, e) in zip(v["span_s"], v["span_e"])
                ]

        self.gt_dict_word = [dict() for _ in range(self.num_events)]
        for i, gt_dict_tok in enumerate(self.gt_dict_tok):
            gt_dict_word = self.gt_dict_word[i]
            for role, spans in gt_dict_tok.items():
                for span in spans:
                    if span != (0, 0):
                        if role not in gt_dict_word:
                            gt_dict_word[role] = list()
                        word_span = self.get_word_span(span)
                        gt_dict_word[role].append(word_span)

    @property
    def old_tok_index(self):
        new_tok_index_to_old_tok_index = dict()
        for old_tok_id, (new_tok_id_s, new_tok_id_e) in enumerate(
                self.old_tok_to_new_tok_index):
            for j in range(new_tok_id_s, new_tok_id_e):
                new_tok_index_to_old_tok_index[j] = old_tok_id
        return new_tok_index_to_old_tok_index

    def get_word_span(self, span):
        if span == (0, 0):
            raise AssertionError()
        offset  = 0
        span    = list(span)
        span[0] = min(span[0], max(self.old_tok_index.keys()))
        span[1] = max(span[1] - 1, min(self.old_tok_index.keys()))
        while span[0] not in self.old_tok_index:
            span[0] += 1
        span_s = self.old_tok_index[span[0]] + offset
        while span[1] not in self.old_tok_index:
            span[1] -= 1
        span_e = self.old_tok_index[span[1]] + offset
        while span_e < span_s:
            span_e += 1
        return (span_s, span_e)

    def __repr__(self):
        s  = ""
        s += "example_id: {}\n".format(self.example_id)
        s += "event_type: {}\n".format(self.event_type)
        s += "trigger_word: {}\n".format(self.event_trigger)
        s += "old_tok_to_new_tok_index: {}\n".format(self.old_tok_to_new_tok_index)
        s += "enc_input_ids: {}\n".format(self.enc_input_ids)
        s += "enc_mask_ids: {}\n".format(self.enc_mask_ids)
        s += "dec_prompt_ids: {}\n".format(self.dec_prompt_ids)
        s += "dec_prompt_mask_ids: {}\n".format(self.dec_prompt_mask_ids)
        s += "event_groups: {}\n".format(self.event_groups)
        s += "dep_heads[:10]: {}\n".format(self.dep_heads[:10])
        s += "dep_rels[:10]: {}\n".format(self.dep_rels[:10])
        s += "coref_clusters[:3]: {}\n".format(self.coref_clusters[:3])
        return s


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class ArgumentExtractionDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]

    @staticmethod
    def _pad_prompt_batch(batch):
        max_len = max(len(f.dec_prompt_ids) for f in batch)

        pad_token_id = 1
        for f in batch:
            for tok_id, mask_id in zip(f.dec_prompt_ids, f.dec_prompt_mask_ids):
                if mask_id == 0:
                    pad_token_id = tok_id
                    break
            else:
                continue
            break

        padded_ids, padded_masks = [], []
        for f in batch:
            pad_len = max_len - len(f.dec_prompt_ids)
            padded_ids.append(f.dec_prompt_ids + [pad_token_id] * pad_len)
            padded_masks.append(f.dec_prompt_mask_ids + [0] * pad_len)
        return torch.tensor(padded_ids), torch.tensor(padded_masks)

    @staticmethod
    def collate_fn(batch):
        enc_input_ids  = torch.tensor([f.enc_input_ids  for f in batch])
        enc_mask_ids   = torch.tensor([f.enc_mask_ids   for f in batch])
        all_ids        = torch.tensor([f.all_ids        for f in batch])
        all_mask_ids   = torch.tensor([f.all_mask_ids   for f in batch])

        if batch[0].dec_prompt_ids is not None:
            dec_prompt_ids, dec_prompt_mask_ids = ArgumentExtractionDataset._pad_prompt_batch(batch)
        else:
            dec_prompt_ids      = None
            dec_prompt_mask_ids = None

        example_idx = [f.example_id for f in batch]
        feature_idx = torch.tensor([f.feature_id for f in batch])

        if batch[0].dec_arg_query_ids is not None:
            dec_arg_query_ids       = [torch.LongTensor(f.dec_arg_query_ids)       for f in batch]
            dec_arg_query_mask_ids  = [torch.LongTensor(f.dec_arg_query_masks)     for f in batch]
            dec_arg_start_positions = [torch.LongTensor(f.dec_arg_start_positions) for f in batch]
            dec_arg_end_positions   = [torch.LongTensor(f.dec_arg_end_positions)   for f in batch]
            start_position_ids      = [torch.FloatTensor(f.start_position_ids)     for f in batch]
            end_position_ids        = [torch.FloatTensor(f.end_position_ids)       for f in batch]
        else:
            dec_arg_query_ids       = None
            dec_arg_query_mask_ids  = None
            dec_arg_start_positions = None
            dec_arg_end_positions   = None
            start_position_ids      = None
            end_position_ids        = None

        target_info              = [f.target_info              for f in batch]
        old_tok_to_new_tok_index = [f.old_tok_to_new_tok_index for f in batch]
        arg_joint_prompt         = [f.arg_joint_prompt         for f in batch]
        arg_lists                = [f.arg_list                 for f in batch]
        event_trigger            = [f.event_trigger            for f in batch]
        enc_attention_mask       = [f.enc_attention_mask       for f in batch]

        # ── 图1 字段 ──────────────────────────────────────────────────
        dep_heads_batch    = [f.dep_heads    for f in batch]   # list[list[int]]
        dep_rels_batch     = [f.dep_rels     for f in batch]   # list[list[str]]
        event_groups_batch = [f.event_groups for f in batch]   # list[list[int]]

        # ── 共指簇字段（新增）────────────────────────────────────────
        # coref_clusters 是嵌套列表（每个样本的簇数量和每簇大小均不同）
        # 直接以 list 传出，不 padding
        coref_clusters_batch = [f.coref_clusters for f in batch]  # list[list[list[int]]]
        coref_logits_batch   = [f.coref_logits   for f in batch]

        def graph_to_tensors(graph):
            if graph is None:
                raise RuntimeError(
                    'Missing offline graph. Delete the stale feature cache or '
                    'let prepare_offline_graphs rebuild graph_cache/*.json.'
                )
            edge_index = torch.tensor(
                [graph['src'], graph['dst']], dtype=torch.long
            )
            edge_weight = torch.tensor(graph['weight'], dtype=torch.float)
            if 'type_ids' not in graph or len(graph['type_ids']) != edge_weight.numel():
                raise RuntimeError('Offline graph lacks edge types; rebuild graph_cache.')
            edge_types = torch.tensor(graph['type_ids'], dtype=torch.long)
            if edge_types.numel() == 0:
                edge_types = edge_types.reshape(0, 1)
            return edge_index, edge_weight, edge_types

        precomputed_dep_graphs = [
            graph_to_tensors(f.precomputed_dep_graph) for f in batch
        ]
        precomputed_trigger_graphs = [
            graph_to_tensors(f.precomputed_trigger_graph) for f in batch
        ]

        return (
            enc_input_ids, enc_mask_ids,
            all_ids, all_mask_ids,
            dec_arg_query_ids, dec_arg_query_mask_ids,
            dec_prompt_ids, dec_prompt_mask_ids,
            target_info, old_tok_to_new_tok_index,
            arg_joint_prompt, arg_lists,
            example_idx, feature_idx,
            dec_arg_start_positions, dec_arg_end_positions,
            start_position_ids, end_position_ids,
            event_trigger, enc_attention_mask,
            dep_heads_batch, dep_rels_batch, event_groups_batch,  # [20][21][22]
            coref_clusters_batch,                                  # [23] ← 新增
            coref_logits_batch,
            precomputed_dep_graphs,                                # [25]
            precomputed_trigger_graphs,                            # [26]
        )


# ─────────────────────────────────────────────────────────────────────────────
# Processor
# ─────────────────────────────────────────────────────────────────────────────
class MultiargProcessor(DSET_processor):
    GRAPH_CACHE_VERSION = 3

    def __init__(self, args, tokenizer):
        super().__init__(args, tokenizer)
        self.set_dec_input()
        self._coref_sample_count=0
        self.collate_fn = ArgumentExtractionDataset.collate_fn

        # Parser/coreference providers are initialized lazily on cache misses.

    @staticmethod
    def _trigger_positions(event_trigger):
        positions = []
        for trigger in event_trigger:
            if isinstance(trigger, int):
                positions.append(trigger)
            elif isinstance(trigger, (list, tuple)):
                ints = [value for value in trigger if isinstance(value, int)]
                if len(ints) >= 2:
                    positions.extend(range(ints[0], ints[1]))
                elif ints:
                    positions.append(ints[0])
            elif isinstance(trigger, dict):
                pos = trigger.get('tok_pos', trigger.get('position'))
                if isinstance(pos, int):
                    positions.append(pos)
                elif pos is not None:
                    positions.extend(pos)
        return positions

    @staticmethod
    def _serialize_edges(edges, edge_types):
        ordered = sorted(edges.items())
        type_labels = [sorted(edge_types[edge]) for edge, _ in ordered]
        type_ids = [
            sorted({EDGE_TYPE_TO_ID[label] for label in labels})
            for labels in type_labels
        ]
        width = max((len(ids) for ids in type_ids), default=1)
        return {
            'src': [int(edge[0][0]) for edge in ordered],
            'dst': [int(edge[0][1]) for edge in ordered],
            'weight': [float(edge[1]) for edge in ordered],
            'type_ids': [ids + [0] * (width - len(ids)) for ids in type_ids],
            'type_labels': type_labels,
        }

    @staticmethod
    def _json_safe(value):
        """Recursively convert NumPy/Torch scalars and containers for JSON."""
        if isinstance(value, dict):
            return {
                str(key): MultiargProcessor._json_safe(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [MultiargProcessor._json_safe(item) for item in value]
        if torch.is_tensor(value):
            return MultiargProcessor._json_safe(value.detach().cpu().tolist())
        # NumPy scalar types expose item(); native Python scalar types do not
        # need conversion. Avoid importing NumPy solely for this check.
        if not isinstance(value, (str, int, float, bool, type(None))):
            item = getattr(value, 'item', None)
            if callable(item):
                return MultiargProcessor._json_safe(item())
        return value

    def _graph_ablation_set(self):
        raw = str(getattr(self.args, 'graph_ablation', 'none')).lower()
        ablations = {
            item.strip() for item in raw.split(',') if item.strip()
        } or {'none'}
        if len(ablations) > 1:
            ablations.discard('none')
        valid_special = {
            'none', 'dependency_all', 'coreference', 'trigger',
            'self_loop', 'trigger_boost',
        }
        invalid = [
            item for item in ablations
            if item not in valid_special and not item.startswith('dep:')
        ]
        if invalid:
            raise ValueError(
                'Unsupported --graph_ablation item(s): ' + ', '.join(invalid)
            )
        return ablations

    def _canonical_graph_ablation(self):
        return ','.join(sorted(self._graph_ablation_set()))

    def _build_offline_graphs(self, feature):
        """Build sparse dependency/coreference and trigger graphs once."""
        ablations = self._graph_ablation_set()

        seq_len = len(feature.all_ids)
        trigger_pos_list = [
            self._trigger_positions([event])
            for event in feature.event_trigger
        ]
        all_trigger_pos = [p for positions in trigger_pos_list for p in positions]

        dep_weights = {
            'nsubj': 1.0, 'nsubjpass': 1.0, 'dobj': 1.0, 'iobj': 0.9,
            'csubj': 0.9, 'ccomp': 0.8, 'xcomp': 0.8, 'root': 1.0,
            'aux': 0.7, 'auxpass': 0.7, 'nmod': 0.5, 'amod': 0.4,
            'advmod': 0.4, 'nummod': 0.4, 'appos': 0.5, 'advcl': 0.3,
            'acl': 0.3, 'relcl': 0.3, 'prep': 0.3, 'pobj': 0.4,
        }
        dep_edges = {}
        dep_edge_types = {}

        def update_edge(container, type_container, src, dst, weight, relation_type):
            if 0 <= src < seq_len and 0 <= dst < seq_len:
                key = (int(src), int(dst))
                container[key] = max(container.get(key, 0.0), float(weight))
                type_container.setdefault(key, set()).add(relation_type)

        for tok_idx, (head, rel) in enumerate(zip(feature.dep_heads, feature.dep_rels)):
            if tok_idx >= seq_len or head >= seq_len or tok_idx == head:
                continue
            relation = str(rel).lower()
            relation_bucket = relation if relation in dep_weights else 'default'
            if (
                'dependency_all' in ablations
                or f'dep:{relation_bucket}' in ablations
            ):
                continue
            weight = dep_weights.get(relation, 0.2)
            update_edge(
                dep_edges, dep_edge_types, tok_idx, head, weight,
                dependency_type(relation),
            )
            update_edge(
                dep_edges, dep_edge_types, head, tok_idx, weight,
                dependency_type(relation, reverse=True),
            )

        clusters = feature.coref_clusters
        logits = feature.coref_logits
        if clusters is None:
            clusters = []
        if logits is None:
            logits = []
        for cluster_idx, cluster in enumerate(clusters):
            if 'coreference' in ablations:
                break
            if cluster_idx < len(logits):
                sigmoid = 1.0 / (1.0 + math.exp(-float(logits[cluster_idx]) / 5.0))
                coref_weight = 0.1 + 0.8 * sigmoid
            else:
                coref_weight = 0.5
            for left in range(len(cluster)):
                for right in range(left + 1, len(cluster)):
                    update_edge(
                        dep_edges, dep_edge_types, cluster[left], cluster[right],
                        coref_weight, 'coreference',
                    )
                    update_edge(
                        dep_edges, dep_edge_types, cluster[right], cluster[left],
                        coref_weight, 'coreference',
                    )

        if 'trigger_boost' not in ablations:
            valid_triggers = [p for p in all_trigger_pos if 0 <= p < seq_len]
            for trigger in valid_triggers:
                for edge in list(dep_edges):
                    weight = dep_edges[edge]
                    if edge[0] == trigger:
                        weight = min(1.0, weight * 1.2)
                    if edge[1] == trigger:
                        weight = min(1.0, weight * 1.2)
                    dep_edges[edge] = weight
        if 'self_loop' not in ablations:
            for idx in range(seq_len):
                dep_edges[(idx, idx)] = 1.0
                dep_edge_types[(idx, idx)] = {'self_loop'}

        trigger_edges = {}
        trigger_edge_types = {}
        grouped_positions = {}
        for positions, group_id in zip(trigger_pos_list, feature.event_groups):
            grouped_positions.setdefault(group_id, []).extend(
                p for p in positions if 0 <= p < seq_len
            )
        if 'trigger' not in ablations:
            for positions in grouped_positions.values():
                for src in positions:
                    for dst in positions:
                        if src != dst:
                            update_edge(
                                trigger_edges, trigger_edge_types, src, dst,
                                1.0 / (1.0 + abs(src - dst)),
                                'trigger_group',
                            )
                    if 'self_loop' not in ablations:
                        trigger_edges[(src, src)] = 1.0
                        trigger_edge_types[(src, src)] = {'self_loop'}

        return (
            self._serialize_edges(dep_edges, dep_edge_types),
            self._serialize_edges(trigger_edges, trigger_edge_types),
        )

    def prepare_offline_graphs(self, features, set_type, data_file):
        """Load split graphs from JSON, or build and atomically persist them."""
        graph_dir = os.path.join(os.path.dirname(data_file), 'graph_cache')
        os.makedirs(graph_dir, exist_ok=True)
        ablation = self._canonical_graph_ablation()
        safe_ablation = re.sub(r'[^a-z0-9_.-]+', '_', ablation)
        graph_path = os.path.join(
            graph_dir, f'{set_type}_graph_{safe_ablation}.json'
        )

        signature = hashlib.sha256()
        signature.update(ablation.encode('utf-8'))
        for feature in features:
            signature.update(str(feature.example_id).encode('utf-8'))
            signature.update(str(feature.feature_id).encode('ascii'))
            signature.update(json.dumps(
                self._json_safe(feature.event_trigger), sort_keys=True
            ).encode('utf-8'))
            signature.update(json.dumps(
                self._json_safe(feature.event_groups)
            ).encode('utf-8'))
            signature.update(json.dumps(
                self._json_safe(feature.dep_heads)
            ).encode('utf-8'))
            signature.update(json.dumps(
                self._json_safe(feature.dep_rels)
            ).encode('utf-8'))
            coref_clusters = feature.coref_clusters
            coref_logits = feature.coref_logits
            signature.update(json.dumps(
                self._json_safe([] if coref_clusters is None else coref_clusters)
            ).encode('utf-8'))
            signature.update(json.dumps(
                self._json_safe([] if coref_logits is None else coref_logits)
            ).encode('utf-8'))
        feature_signature = signature.hexdigest()

        payload = None
        if os.path.exists(graph_path):
            try:
                with open(graph_path, encoding='utf-8') as graph_file:
                    candidate = json.load(graph_file)
                if (
                    candidate.get('version') == self.GRAPH_CACHE_VERSION
                    and candidate.get('feature_count') == len(features)
                    and candidate.get('max_enc_seq_length') == self.args.max_enc_seq_length
                    and candidate.get('graph_ablation') == ablation
                    and candidate.get('feature_signature') == feature_signature
                    and candidate.get('edge_type_names') == list(EDGE_TYPE_NAMES)
                ):
                    payload = candidate
            except (OSError, ValueError, TypeError) as exc:
                logger.warning('[GraphCache] Ignoring invalid %s: %s', graph_path, exc)

        if payload is None:
            logger.info('[GraphCache] Building offline %s graphs: %s', set_type, graph_path)
            records = []
            for feature in features:
                dep_graph, trigger_graph = self._build_offline_graphs(feature)
                records.append({
                    'example_id': str(feature.example_id),
                    'feature_id': int(feature.feature_id),
                    'dependency_coref_graph': dep_graph,
                    'trigger_graph': trigger_graph,
                })
            payload = {
                'version': self.GRAPH_CACHE_VERSION,
                'set_type': set_type,
                'feature_count': len(features),
                'max_enc_seq_length': self.args.max_enc_seq_length,
                'graph_ablation': ablation,
                'feature_signature': feature_signature,
                'edge_type_names': list(EDGE_TYPE_NAMES),
                'records': records,
            }
            tmp_path = graph_path + f'.tmp.{os.getpid()}'
            with open(tmp_path, 'w', encoding='utf-8') as graph_file:
                json.dump(payload, graph_file, separators=(',', ':'))
            os.replace(tmp_path, graph_path)
            print(f'[GraphCache] 已保存离线图 → {graph_path}')
        else:
            print(f'[GraphCache] ✓ 加载离线图: {graph_path}')

        records = payload['records']
        if len(records) != len(features):
            raise RuntimeError(f'Graph cache size mismatch: {graph_path}')
        for feature, record in zip(features, records):
            if str(feature.example_id) != record['example_id']:
                raise RuntimeError(f'Graph cache order mismatch: {graph_path}')
            feature.precomputed_dep_graph = record['dependency_coref_graph']
            feature.precomputed_trigger_graph = record['trigger_graph']

    def set_dec_input(self):
        self.arg_query    = False
        self.prompt_query = False
        if self.args.model_type == "base":
            self.arg_query = True
        elif self.args.model_type.lower() in {"lpg-eae", "lpgeae"}:
            self.prompt_query = True
        else:
            raise NotImplementedError(f"Unexpected setting {self.args.model_type}")

    @staticmethod
    def _read_prompt_group(prompt_path):
        with open(prompt_path) as f:
            lines = f.readlines()
        prompts = dict()
        for line in lines:
            if not line:
                continue
            event_type, prompt = line.split(":")
            prompts[event_type] = prompt
        return prompts

    def create_dec_qury(self, arg, event_trigger):
        dec_text = _PREDEFINED_QUERY_TEMPLATE.format(
            arg=arg, trigger=event_trigger
        )
        dec = self.tokenizer(dec_text)
        dec_input_ids, dec_mask_ids = dec["input_ids"], dec["attention_mask"]
        while len(dec_input_ids) < self.args.max_dec_seq_length:
            dec_input_ids.append(self.tokenizer.pad_token_id)
            dec_mask_ids.append(self.args.pad_mask_token)
        matching_result = re.search(arg, dec_text)
        char_idx_s, char_idx_e = matching_result.span()
        char_idx_e -= 1
        tok_prompt_s = dec.char_to_token(char_idx_s)
        tok_prompt_e = dec.char_to_token(char_idx_e) + 1
        return dec_input_ids, dec_mask_ids, tok_prompt_s, tok_prompt_e

    def find_idx(self, target, lst):
        for i, item in enumerate(lst):
            if item == target:
                return i

    # ── 依存解析 + subword 对齐 ──────────────────────────────────────
    def _build_dep_fields(
        self,
        context:                  list,
        marked_context:           list,
        old_tok_to_new_tok_index: list,
        enc:                      object,
        seq_len:                  int,
    ):
        if self.syntax_provider is None:
            self.syntax_provider = SyntaxProvider()

        dep_heads = list(range(seq_len))
        dep_rels  = ['none'] * seq_len

        plain_text = " ".join(context)
        try:
            doc = self.syntax_provider.nlp(plain_text)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"[dep_parse] spaCy 解析失败: {e}")
            return dep_heads, dep_rels

        char_start_to_word_idx = {}
        char_pos = 0
        for w_i, tok in enumerate(context):
            char_start_to_word_idx[char_pos] = w_i
            char_pos += len(tok) + 1

        word_dep = {}
        for spacy_tok in doc:
            w_i = char_start_to_word_idx.get(spacy_tok.idx, None)
            if w_i is None:
                continue
            head_w_i = char_start_to_word_idx.get(spacy_tok.head.idx, w_i)
            word_dep[w_i] = (head_w_i, spacy_tok.dep_)

        for w_i, (head_w_i, rel) in word_dep.items():
            if w_i >= len(old_tok_to_new_tok_index):
                continue
            sw_range   = old_tok_to_new_tok_index[w_i]
            sw_s, sw_e = sw_range[0], sw_range[1]
            if head_w_i < len(old_tok_to_new_tok_index):
                head_sw = old_tok_to_new_tok_index[head_w_i][0]
            else:
                head_sw = sw_s
            for sw in range(sw_s, min(sw_e, seq_len)):
                dep_heads[sw] = min(head_sw, seq_len - 1)
                dep_rels[sw]  = rel

        return dep_heads, dep_rels

    # ── 共指解析 + subword 对齐（新增）──────────────────────────────
    def _build_coref_clusters(self, context, old_tok_to_new_tok_index, seq_len):
        if self.coref_provider is None:
            from processors.processor_base import CorefProvider
            self.coref_provider = CorefProvider()
        if not self.coref_provider.enabled:
            return [], []
        plain_text = ' '.join(context)
        try:
            clusters, logits = self.coref_provider.get_coref_clusters(
                text=plain_text,
                context=context,
                old_tok_to_new_tok_index=old_tok_to_new_tok_index,
                seq_len=seq_len,
            )
            return clusters, logits
        except Exception as e:
            logger.warning(f"[_build_coref_clusters] 失败: {e}")
            return [], []

    # ── 主转换函数 ────────────────────────────────────────────────────
    def convert_examples_to_features(self, examples, role_name_mapping=None):
        features  = []
        over_nums = 0

        if self.prompt_query:
            prompts = self._read_prompt_group(self.args.prompt_path)

        if os.environ.get("DEBUG", False):
            counter = [0, 0, 0]

        for example in examples:
            example_id           = example.doc_id
            context              = example.context
            event_type_2_events  = example.event_type_2_events

            list_event_type = []
            triggers        = []
            for event_type, events in event_type_2_events.items():
                list_event_type += [e['event_type'] for e in events]
                triggers        += [tuple(e['trigger']) for e in events]

            set_triggers = sorted(list(set(triggers)))

            trigger_overlap = False
            for t1 in set_triggers:
                for t2 in set_triggers:
                    if t1[0] == t2[0] and t1[1] == t2[1]:
                        continue
                    if ((t1[0] < t2[1] and t2[0] < t1[1]) or
                            (t2[0] < t1[1] and t1[0] < t2[1])):
                        trigger_overlap = True
                        break
            if trigger_overlap:
                print('[trigger_overlap]', event_type_2_events)
                exit(0)

            # ── marked_context ────────────────────────────────────────
            offset         = 0
            marked_context = deepcopy(context)
            marker_indice  = list(range(len(triggers)))
            for i, t in enumerate(set_triggers):
                t_start = t[0]
                t_end   = t[1]
                marked_context = (
                    marked_context[:(t_start + offset)]
                    + ['<t-%d>' % marker_indice[i]]
                    + context[t_start: t_end]
                    + ['</t-%d>' % marker_indice[i]]
                    + context[t_end:]
                )
                offset += 2
            enc_text = " ".join(marked_context)

            # ── old_tok_to_new_tok_index ──────────────────────────────
            old_tok_to_char_index    = []
            old_tok_to_new_tok_index = []

            curr         = 0
            enc          = self.tokenizer(enc_text, add_special_tokens=True)
            trigger_list = [[] for _ in range(len(triggers))]
            for tok in marked_context:
                if tok not in EXTERNAL_TOKENS:
                    old_tok_to_char_index.append([curr, curr + len(tok) - 1])
                curr += len(tok) + 1

            enc_input_ids, enc_mask_ids = enc["input_ids"], enc["attention_mask"]
            if len(enc_input_ids) > self.args.max_enc_seq_length:
                raise ValueError(
                    f"Please increase max_enc_seq_length above {len(enc_input_ids)}"
                )

            all_ids      = enc_input_ids.copy()
            all_mask_ids = enc_mask_ids.copy()
            type_ids     = enc_mask_ids.copy()

            offset_prompt = len(enc_input_ids)

            while len(enc_input_ids) < self.args.max_enc_seq_length:
                enc_input_ids.append(self.tokenizer.pad_token_id)
                enc_mask_ids.append(self.args.pad_mask_token)

            for old_tok_idx, (char_idx_s, char_idx_e) in enumerate(
                    old_tok_to_char_index):
                new_tok_s = enc.char_to_token(char_idx_s)
                new_tok_e = enc.char_to_token(char_idx_e) + 1
                old_tok_to_new_tok_index.append([new_tok_s, new_tok_e])

            trigger_enc_token_index = []
            for t in triggers:
                new_t_start = old_tok_to_new_tok_index[t[0]][0]
                new_t_end   = old_tok_to_new_tok_index[t[1] - 1][1]
                trigger_enc_token_index.append([new_t_start, new_t_end])
            for ii, it in enumerate(trigger_enc_token_index):
                type_ids[it[0] - 1] = ii + 2

            dec_table_ids  = []
            dec_table_mask = []

            list_arg_2_prompt_slots      = []
            list_num_prompt_slots        = []
            list_dec_prompt_ids          = []
            list_arg_2_prompt_slot_spans = []
            offset_prompt_               = 0
            kk                           = 0
            enc_attention_mask           = torch.zeros(
                (2, self.args.max_enc_seq_length, self.args.max_enc_seq_length),
                dtype=torch.float32,
            )

            for i, event_type in enumerate(event_type_2_events):
                events     = event_type_2_events[event_type]
                event_name = event_type.split('.')
                event_name = ['<e-%d>' % i] + event_name + ['</e-%d>' % i]
                for event in events:
                    enc_trigger_start = trigger_enc_token_index[kk][0] - 1
                    enc_trigger_end   = trigger_enc_token_index[kk][1] + 1
                    kk += 1
                    dec_prompt_text = prompts[event_type].strip()
                    assert dec_prompt_text
                    dec_prompt_text = ' '.join(event_name) + ' ' + dec_prompt_text
                    dec_prompt      = self.tokenizer(
                        dec_prompt_text, add_special_tokens=True
                    )
                    dec_prompt_ids_, dec_prompt_mask_ids_ = (
                        dec_prompt["input_ids"],
                        dec_prompt["attention_mask"],
                    )

                    arg_list = self.argument_dict[
                        event_type.replace(':', '.')
                    ]
                    arg_2_prompt_slots      = dict()
                    arg_2_prompt_slot_spans = dict()
                    num_prompt_slots        = 0
                    for arg in arg_list:
                        prompt_slots      = {
                            "tok_s": [], "tok_e": [],
                            "tok_s_off": [], "tok_e_off": [],
                        }
                        prompt_slot_spans = []
                        if role_name_mapping is not None:
                            arg_ = role_name_mapping[event_type][arg]
                        else:
                            arg_ = arg
                        for matching_result in re.finditer(
                            r'\b' + re.escape(arg_) + r'\b',
                            dec_prompt_text.split('.')[0],
                        ):
                            char_idx_s, char_idx_e = matching_result.span()
                            char_idx_e -= 1
                            tok_prompt_s = dec_prompt.char_to_token(char_idx_s)
                            tok_prompt_e = dec_prompt.char_to_token(char_idx_e) + 1
                            prompt_slot_spans.append((tok_prompt_s, tok_prompt_e))
                            prompt_slots["tok_s"].append(
                                tok_prompt_s + offset_prompt_
                            )
                            prompt_slots["tok_e"].append(
                                tok_prompt_e + offset_prompt_
                            )
                            prompt_slots["tok_s_off"].append(
                                tok_prompt_s + offset_prompt + offset_prompt_
                            )
                            prompt_slots["tok_e_off"].append(
                                tok_prompt_e + offset_prompt + offset_prompt_
                            )
                            num_prompt_slots += 1
                        arg_2_prompt_slots[arg]      = prompt_slots
                        arg_2_prompt_slot_spans[arg] = prompt_slot_spans

                    list_arg_2_prompt_slots.append(arg_2_prompt_slots)
                    list_num_prompt_slots.append(num_prompt_slots)
                    list_dec_prompt_ids.append(dec_prompt_ids_)
                    list_arg_2_prompt_slot_spans.append(arg_2_prompt_slot_spans)

                    enc_attention_mask[
                        0,
                        enc_trigger_start:enc_trigger_end,
                        offset_prompt + offset_prompt_:
                        offset_prompt + offset_prompt_ + len(dec_prompt_ids_),
                    ] = 1
                    enc_attention_mask[
                        0,
                        offset_prompt + offset_prompt_:
                        offset_prompt + offset_prompt_ + len(dec_prompt_ids_),
                        enc_trigger_start:enc_trigger_end,
                    ] = 1
                    enc_attention_mask[
                        1,
                        enc_trigger_start:enc_trigger_end,
                        offset_prompt:offset_prompt + offset_prompt_,
                    ] = 1
                    enc_attention_mask[
                        1,
                        enc_trigger_start:enc_trigger_end,
                        offset_prompt + offset_prompt_ + len(dec_prompt_ids_):,
                    ] = 1
                    enc_attention_mask[
                        1,
                        offset_prompt:offset_prompt + offset_prompt_,
                        enc_trigger_start:enc_trigger_end,
                    ] = 1
                    enc_attention_mask[
                        1,
                        offset_prompt + offset_prompt_ + len(dec_prompt_ids_):,
                        enc_trigger_start:enc_trigger_end,
                    ] = 1

                enc_attention_mask[
                    0,
                    offset_prompt + offset_prompt_:
                    offset_prompt + offset_prompt_ + len(dec_prompt_ids_),
                    offset_prompt + offset_prompt_:
                    offset_prompt + offset_prompt_ + len(dec_prompt_ids_),
                ] = 1
                enc_attention_mask[
                    1,
                    offset_prompt + offset_prompt_:
                    offset_prompt + offset_prompt_ + len(dec_prompt_ids_),
                    offset_prompt:offset_prompt + offset_prompt_,
                ] = 1
                enc_attention_mask[
                    1,
                    offset_prompt + offset_prompt_:
                    offset_prompt + offset_prompt_ + len(dec_prompt_ids_),
                    offset_prompt + offset_prompt_ + len(dec_prompt_ids_):,
                ] = 1
                enc_attention_mask[
                    1,
                    offset_prompt:offset_prompt + offset_prompt_,
                    offset_prompt + offset_prompt_:
                    offset_prompt + offset_prompt_ + len(dec_prompt_ids_),
                ] = 1
                enc_attention_mask[
                    1,
                    offset_prompt + offset_prompt_ + len(dec_prompt_ids_):,
                    offset_prompt + offset_prompt_:
                    offset_prompt + offset_prompt_ + len(dec_prompt_ids_),
                ] = 1

                offset_prompt_ += len(dec_prompt_ids_)
                dec_table_ids   += dec_prompt_ids_
                dec_table_mask  += dec_prompt_mask_ids_

            all_ids.extend(dec_table_ids)
            all_mask_ids.extend(dec_table_mask)
            if len(all_ids) > self.args.max_enc_seq_length:
                over_nums += 1

            while len(all_ids) < self.args.max_enc_seq_length:
                all_ids.append(self.tokenizer.pad_token_id)
                all_mask_ids.append(self.args.pad_mask_token)

            # ── 依存解析 ──────────────────────────────────────────────
            actual_seq_len = min(
                offset_prompt + offset_prompt_,
                self.args.max_enc_seq_length,
            )
            dep_heads, dep_rels = self._build_dep_fields(
                context=context,
                marked_context=marked_context,
                old_tok_to_new_tok_index=old_tok_to_new_tok_index,
                enc=enc,
                seq_len=actual_seq_len,
            )

            # ── 意群 group id ─────────────────────────────────────────
            token_chunk_ids = example.token_chunk_ids
            event_groups    = []
            for trig in triggers:
                trig_start = trig[0]
                if token_chunk_ids and trig_start < len(token_chunk_ids):
                    event_groups.append(token_chunk_ids[trig_start])
                else:
                    event_groups.append(0)

            # ── 共指解析（新增）───────────────────────────────────────
            # 使用原始 context（无 marker）进行 AllenNLP 共指消解
            # 得到 subword 粒度的共指簇，直接对应 LPG-EAE graph construction的输入格式
            if self._coref_sample_count==0:
                if not hasattr(self, '_coref_sample_count'):
                    self._coref_sample_count = 0
                self._coref_sample_count += 1
                print(f"[Coref] 第 {self._coref_sample_count} 条样本: {example_id}")
            coref_clusters, coref_logits = self._build_coref_clusters(
                    context=context,
                    old_tok_to_new_tok_index=old_tok_to_new_tok_index,
                    seq_len=actual_seq_len,
                )
            # print(f"[Coref] 第 {self._coref_sample_count} 条样本共指簇数: {len(coref_clusters)}")
            # for ci, cluster in enumerate(coref_clusters):
            #     print(f"[Coref]   簇{ci}: {cluster}")

            # ── 处理 target arguments（与原逻辑完全一致）────────────────
            row_index        = 0
            list_trigger_pos = []
            list_arg_slots   = []
            list_target_info = []
            list_roles       = []
            k                = 0

            for i, (event_type, events) in enumerate(
                    event_type_2_events.items()):
                for event in events:
                    arg_2_prompt_slots = list_arg_2_prompt_slots[k]
                    num_prompt_slots   = list_num_prompt_slots[k]
                    dec_prompt_ids_    = list_dec_prompt_ids[k]
                    k         += 1
                    row_index += 1

                    list_trigger_pos.append(len(dec_table_ids))
                    arg_slots = []
                    cursor    = len(dec_table_ids) + 1
                    event_args      = event['args']
                    event_args_name = [arg[-1] for arg in event_args]
                    target_info     = dict()

                    for arg, prompt_slots in arg_2_prompt_slots.items():
                        num_slots = len(prompt_slots['tok_s'])
                        arg_slots.append(
                            [cursor + x for x in range(num_slots)]
                        )
                        cursor += num_slots

                        arg_target = {"text": [], "span_s": [], "span_e": []}
                        if arg in event_args_name:
                            if os.environ.get("DEBUG", False):
                                counter[0] += 1
                            arg_idxs = [
                                j for j, x in enumerate(event_args_name)
                                if x == arg
                            ]
                            if os.environ.get("DEBUG", False):
                                counter[1] += len(arg_idxs)
                            for arg_idx in arg_idxs:
                                event_arg_info  = event_args[arg_idx]
                                answer_text     = event_arg_info[2]
                                start_old, end_old = (
                                    event_arg_info[0], event_arg_info[1]
                                )
                                start_position = old_tok_to_new_tok_index[
                                    start_old
                                ][0]
                                end_position   = old_tok_to_new_tok_index[
                                    end_old - 1
                                ][1]
                                arg_target["text"].append(answer_text)
                                arg_target["span_s"].append(start_position)
                                arg_target["span_e"].append(end_position)

                        target_info[arg] = arg_target

                    assert sum(
                        [len(slots) for slots in arg_slots]
                    ) == num_prompt_slots
                    list_arg_slots.append(arg_slots)
                    list_target_info.append(target_info)
                    roles = self.argument_dict[
                        event_type.replace(':', '.')
                    ]
                    assert len(roles) == len(arg_slots)
                    list_roles.append(roles)

            max_dec_seq_len = self.args.max_prompt_seq_length
            while len(dec_table_ids) < max_dec_seq_len:
                dec_table_ids.append(self.tokenizer.pad_token_id)
                dec_table_mask.append(self.args.pad_mask_token)

            if len(all_ids) > self.args.max_enc_seq_length:
                enc_attention_mask = torch.zeros(
                    (2, self.args.max_enc_seq_length, self.args.max_enc_seq_length),
                    dtype=torch.float32,
                )
            if len(all_ids) > self.args.max_enc_seq_length:
                all_ids      = all_ids[:self.args.max_enc_seq_length]
                all_mask_ids = all_mask_ids[:self.args.max_enc_seq_length]
            if len(list_arg_2_prompt_slots) == 1:
                enc_attention_mask = torch.zeros(
                    (2, self.args.max_enc_seq_length, self.args.max_enc_seq_length),
                    dtype=torch.float32,
                )

            feature_idx = len(features)
            features.append(
                InputFeatures(
                    example_id, feature_idx,
                    list_event_type,
                    trigger_enc_token_index,
                    enc_text, enc_input_ids, enc_mask_ids,
                    all_ids, all_mask_ids,
                    dec_prompt_text, dec_table_ids, dec_table_mask,
                    None,
                    list_arg_2_prompt_slots, list_target_info,
                    enc_attention_mask,
                    old_tok_to_new_tok_index=old_tok_to_new_tok_index,
                    full_text=example.context,
                    arg_list=list_roles,
                    dep_heads=dep_heads,
                    dep_rels=dep_rels,
                    event_groups=event_groups,
                    coref_clusters=coref_clusters,   # ← 新增
                    coref_logits=coref_logits,
                )
            )

        print(over_nums)
        if os.environ.get("DEBUG", False):
            print(
                '\033[91m'
                + f"distinct/tot arg_role: {counter[0]}/{counter[1]} ({counter[2]})"
                + '\033[0m'
            )
        return features

    def convert_features_to_dataset(self, features):
        return ArgumentExtractionDataset(features)

