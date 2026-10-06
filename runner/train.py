import torch.nn as nn
import logging
from torch.utils.data import DataLoader, RandomSampler

logger = logging.getLogger(__name__)

# collate_fn 返回的 batch tuple 结构（共 27 个元素）：
# [0]  enc_input_ids
# [1]  enc_mask_ids
# [2]  all_ids
# [3]  all_mask_ids
# [4]  dec_arg_query_ids
# [5]  dec_arg_query_mask_ids
# [6]  dec_prompt_ids
# [7]  dec_prompt_mask_ids
# [8]  target_info
# [9]  old_tok_to_new_tok_index
# [10] arg_joint_prompt
# [11] arg_lists
# [12] example_idx
# [13] feature_idx
# [14] dec_arg_start_positions
# [15] dec_arg_end_positions
# [16] start_position_ids
# [17] end_position_ids
# [18] event_trigger
# [19] enc_attention_mask
# [20] dep_heads_batch         ← 图1：依存 head 索引
# [21] dep_rels_batch          ← 图1：依存关系标签
# [22] event_groups_batch      ← 图1：意群 group id
# [23] coref_clusters_batch    ← 图1：共指簇（新增）
#                                list[list[list[int]]]
#                                每个样本是一组共指簇，每个簇是一组 subword 索引
#                                第一个元素为超级节点（代表词），其余被重定向到它
# [24] coref_logits_batch      ← 共指置信度
# [25] precomputed_dep_graphs  ← 离线依存+共指稀疏图
# [26] precomputed_trigger_graphs ← 离线触发词稀疏图


class BaseTrainer:
    def __init__(
        self,
        cfg=None,
        data_loader=None,
        model=None,
        optimizer=None,
        scheduler=None,
        processor=None,
    ):
        self.cfg           = cfg
        self.data_loader   = data_loader
        self.data_iterator = iter(self.data_loader)
        self.model         = model
        self.optimizer     = optimizer
        self.scheduler     = scheduler
        self.processor     = processor
        self._init_metric()

    def _init_metric(self):
        self.metric = {
            "global_steps": 0,
            "smooth_loss":  0.0,
        }

    def write_log(self):
        logger.info(
            "-----------------------global_step: {} -------------------------------- "
            .format(self.metric['global_steps'])
        )
        logger.info('lr: {}'.format(self.scheduler.get_last_lr()[0]))
        logger.info('smooth_loss: {}'.format(self.metric['smooth_loss']))
        self.metric['smooth_loss'] = 0.0

    def train_one_step(self):
        self.model.train()
        try:
            batch = next(self.data_iterator)
        except StopIteration:
            if self.processor is not None:
                print('re-generate training dataset')
                features = self.processor.convert_examples_to_features(
                    self.examples, 'train', self.cfg.marker_range
                )
                dataset         = self.processor.convert_features_to_dataset(features)
                dataset_sampler = RandomSampler(dataset)
                self.dataloader = DataLoader(
                    dataset,
                    sampler=dataset_sampler,
                    batch_size=self.cfg.batch_size,
                    collate_fn=self.processor.collate_fn,
                )

            self.data_iterator = iter(self.data_loader)
            batch = next(self.data_iterator)

        inputs  = self.convert_batch_to_inputs(batch)
        loss, _ = self.model(**inputs)

        if self.cfg.gradient_accumulation_steps > 1:
            loss = loss / self.cfg.gradient_accumulation_steps
        loss.backward()

        if self.cfg.max_grad_norm != 0:
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.max_grad_norm
            )

        self.metric['smooth_loss'] += loss.item() / self.cfg.logging_steps
        if (self.metric['global_steps'] + 1) % self.cfg.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.scheduler.step()
            self.model.zero_grad()
            self.metric['global_steps'] += 1
        else:
            self.metric['global_steps'] += 1

    def convert_batch_to_inputs(self, batch):
        raise NotImplementedError()


class Trainer(BaseTrainer):
    def __init__(
        self,
        cfg=None,
        data_loader=None,
        model=None,
        optimizer=None,
        scheduler=None,
        processor=None,
    ):
        super().__init__(cfg, data_loader, model, optimizer, scheduler)

    def convert_batch_to_inputs(self, batch):
        inputs = {
            # ── 原有字段（索引不变）────────────────────────────────────
            'enc_input_ids':             batch[0].to(self.cfg.device),
            'enc_mask_ids':              batch[1].to(self.cfg.device),
            'all_ids':                   batch[2].to(self.cfg.device),
            'all_mask_ids':              batch[3].to(self.cfg.device),
            'dec_prompt_ids':            batch[6].to(self.cfg.device),
            'dec_prompt_mask_ids':       batch[7].to(self.cfg.device),
            'target_info':               batch[8],
            'old_tok_to_new_tok_indexs': batch[9],
            'arg_joint_prompts':         batch[10],
            'arg_list':                  batch[11],
            'event_triggers':            batch[18],
            'enc_attention_mask':        batch[19],
            # ── 图1 字段 ───────────────────────────────────────────────
            'dep_heads_batch':           batch[20],   # list[list[int]]
            'dep_rels_batch':            batch[21],   # list[list[str]]
            'event_groups_batch':        batch[22],   # list[list[int]]
            # ── 共指簇字段（新增）─────────────────────────────────────
            # 格式：list[list[list[int]]]
            # 外层 list：batch 维度
            # 中层 list：该样本的共指簇列表
            # 内层 list：同一簇内各 mention 的代表 subword 索引
            #            第一个元素为超级节点，其余被重定向到它
            'coref_clusters_batch':      batch[23],
            'coref_logits_batch':        batch[24],
            'precomputed_dep_graphs':    batch[25],
            'precomputed_trigger_graphs': batch[26],
        }
        return inputs

