import copy
import logging
logger = logging.getLogger(__name__)

from transformers import AdamW, get_linear_schedule_with_warmup

from .lpg_eae import LPGEAE
from utils import EXTERNAL_TOKENS
from processors.processor_multiarg import MultiargProcessor


from transformers import RobertaConfig, RobertaTokenizerFast

def build_model(args, model_type):
    config_class, model_class, tokenizer_class = (RobertaConfig, LPGEAE, RobertaTokenizerFast)
    if args.inference_only:
        config = config_class.from_pretrained(args.inference_model_path)
    else:
        config = config_class.from_pretrained(args.model_name_or_path)
    config.model_name_or_path = args.model_name_or_path
    config.dataset = args.dataset_type
    config.encoder_layers = args.encoder_layers
    config.context_representation = args.context_representation

    # length
    config.max_dec_seq_length= args.max_dec_seq_length
    config.max_prompt_seq_length=args.max_prompt_seq_length
    config.max_span_length = args.max_span_length
    config.light_prompt_layers = getattr(args, 'light_prompt_layers', 1)
    config.light_prompt_heads = getattr(args, 'light_prompt_heads', 8)
    config.light_prompt_ffn_ratio = getattr(args, 'light_prompt_ffn_ratio', 2.0)
    config.light_prompt_dropout = getattr(args, 'light_prompt_dropout', 0.1)
    config.use_light_prompt = getattr(args, 'use_light_prompt', True)

    config.bipartite = args.bipartite
    config.matching_method_train = args.matching_method_train

    config.graph_ablation = getattr(args, 'graph_ablation', 'none')
    config.gat_query_fuse_weight = args.gat_query_fuse_weight
    config.use_dual_stream_gat = args.use_dual_stream_gat
    config.use_dependency_coref_graph = args.use_dependency_coref_graph
    config.use_trigger_graph = args.use_trigger_graph

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    if args.inference_only:
        model = model_class.from_pretrained(args.inference_model_path, from_tf=bool('.ckpt' in args.inference_model_path), config=config)
    else:
        model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)

    # Add trigger special tokens and continuous token (maybe in prompt)
    new_token_list = copy.deepcopy(EXTERNAL_TOKENS)
    prompts = MultiargProcessor._read_prompt_group(args.prompt_path)
    for event_type, prompt in prompts.items():
        token_list = prompt.split()
        for token in token_list:
            if token.startswith('<') and token.endswith('>') and token not in new_token_list:
                new_token_list.append(token)
    tokenizer.add_tokens(new_token_list)   
    logger.info("Add tokens: {}".format(new_token_list))      
    model.resize_token_embeddings(len(tokenizer))

    if args.inference_only:
        optimizer, scheduler = None, None
    else:

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm', 'layernorm', 'layer_norm']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() if
                           not any(nd in n for nd in no_decay) and 'crossattention' in n],
                'weight_decay': args.weight_decay,
                'lr': args.learning_rate * 1.5
            },
            {
                'params': [p for n, p in model.named_parameters() if
                           any(nd in n for nd in no_decay) and 'crossattention' in n],
                'weight_decay': 0.0,
                'lr': args.learning_rate * 1.5
            },
            {
                'params': [p for n, p in model.named_parameters() if
                           not any(nd in n for nd in no_decay) and 'crossattention' not in n],
                'weight_decay': args.weight_decay,
                'lr': args.learning_rate
            },
            {
                'params': [p for n, p in model.named_parameters() if
                           any(nd in n for nd in no_decay) and 'crossattention' not in n],
                'weight_decay': 0.0,
                'lr': args.learning_rate
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        # 判断是比例还是绝对步数
        warmup = args.warmup_steps
        if warmup < 1:   # 小于1认为是比例
            warmup = int(args.max_steps * warmup)
        else:            # 大于等于1认为是绝对步数
            warmup = int(warmup)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup,
            num_training_steps=args.max_steps
        )

    return model, tokenizer, optimizer, scheduler
