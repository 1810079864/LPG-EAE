import argparse


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default='LPG-EAE', type=str,
                        help="which types of model you would use. LPG-EAE multi-event argument extraction model")
    parser.add_argument("--model_name_or_path", default="roberta-large", type=str,
                        help="pre-trained language model")
    parser.add_argument("--dataset_type", default="rams", type=str,
                        choices=("rams", "wikievent"),
                        help="dataset type: rams or wikievent")
    parser.add_argument("--role_path", default='./data/dset_meta/description_rams.csv', type=str,
                        help="a file containing all role names. Read it to access all argument roles of this dataset")
    parser.add_argument("--prompt_path", default='./data/prompts/prompts_rams_full.csv', type=str,
                        help="a file containing all prompts we use for this dataset")
    parser.add_argument("--output_dir", default='./outputs_res', type=str,
                        help="output folder storing checkpoint and all sorts of log files")
    parser.add_argument(
        '--feature_cache_dir', default=None, type=str,
        help='Optional shared feature-cache directory, independent of output_dir.',
    )
    parser.add_argument("--keep_ratio", default=1.0, type=float,
                        help="The ratio of remaining traning samples. We drop the others. Used in Few-shot setting.")
    parser.add_argument('--inference_only', default=False, action="store_true",
                        help="The model will inference directly without training if it were set as True")
    parser.add_argument('--single', default=False, action="store_true",
                        help="The model will extract one event at a time if set as True")

    parser.add_argument("--pad_mask_token", default=0, type=int,
                        help="padding token id")
    parser.add_argument('--logging_steps', default=100, type=int,
                        help="step intervals for outputting log files")
    parser.add_argument('--eval_steps', default=500, type=int,
                        help="step intervals for validation")
    parser.add_argument("--max_span_length", default=10, type=int,
                        help="a heuristic constraint: the maximum length of extracted arguments")
    parser.add_argument("--batch_size", default=4, type=int, 
                        help="batch size during training. with BP")
    parser.add_argument("--infer_batch_size", default=32, type=int, 
                        help="batch size during inference. without BP")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, 
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--max_enc_seq_length", default=500, type=int,
                        help="maximum length for context")
    parser.add_argument("--window_size", default=260, type=int,
                        help="for document exceeding the length constraint, add a window centering at the trigger word and drop words outside this window")
    parser.add_argument("--encoder_layers", default=17, type=int,
                        help="encoder_layers")
    parser.add_argument('--context_representation', default="decoder", choices=['encoder', 'decoder'], type=str,
                        help="Select the final or intermediate RoBERTa hidden representation.")

    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=5.0, type=float)
    parser.add_argument("--max_steps", default=10000, type=int)
    parser.add_argument("--warmup_steps", default=0.1, type=float)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument("--device", default='cuda', type=str)


    # setting only for the situation when inference_only
    parser.add_argument('--inference_model_path', default=None, type=str,
                        help="The path of checkpoint used for inference.")
    # setting only for base model.
    parser.add_argument("--max_dec_seq_length", default=20, type=int,
                        help="maximum length for single prompt")
    parser.add_argument("--max_span_num", default=1, type=int,
                        help="maximum arguments extracted for one role.")
    parser.add_argument('--th_delta', default=.0, type=float,
                        help="threshold controlling whether accept a candiate span as argument or not")
    # LPG-EAE modules
    parser.add_argument("--max_prompt_seq_length", default=64, type=int,
                        help="maximum length for multi-prompt")
    parser.add_argument('--light_prompt_layers', default=1, type=int, choices=[1, 2],
                        help='Number of lightweight prompt-document cross-attention layers.')
    parser.add_argument('--light_prompt_heads', default=8, type=int,
                        help='Attention heads in each lightweight prompt layer.')
    parser.add_argument('--light_prompt_ffn_ratio', default=2.0, type=float,
                        help='FFN expansion ratio in the lightweight prompt layer.')
    parser.add_argument('--light_prompt_dropout', default=0.1, type=float)
    parser.add_argument('--use_light_prompt', default=True,
                        action='store_true',
                        help='Enable lightweight prompt-document cross-attention.')
    parser.add_argument('--no_light_prompt', dest='use_light_prompt',
                        action='store_false',
                        help='Ablate lightweight prompt-document cross-attention.')
    parser.add_argument('--matching_method_train', default="max", choices=["max", 'accurate'], type=str,
                        help="start/end token matching method during training.")
    parser.add_argument('--bipartite', default=False, action="store_true",
                        help="whether use bipartite matching loss during training or not.")
    parser.add_argument(
        '--graph_ablation', default='none', type=str,
        help=(
            'Comma-separated offline graph ablations: none, dep:<label>, '
            'dep:default, dependency_all, coreference, trigger, '
            'self_loop, or trigger_boost.'
        ),
    )
    parser.add_argument("--gat_query_fuse_weight", default=0.2, type=float,
                        help="Residual weight for graph-enhanced role queries.")
    parser.add_argument("--use_dual_stream_gat", default=True,
                        action="store_true",
                        help="Enable dual-stream graph enhancement.")
    parser.add_argument("--no_dual_stream_gat",
                        dest="use_dual_stream_gat",
                        action="store_false",
                        help="Disable dual-stream graph enhancement and query injection.")
    parser.add_argument('--use_dependency_coref_graph', default=True,
                        action='store_true',
                        help='Enable the dependency/coreference GAT stream.')
    parser.add_argument('--no_dependency_coref_graph',
                        dest='use_dependency_coref_graph',
                        action='store_false',
                        help='Ablate the complete dependency/coreference GAT stream.')
    parser.add_argument('--use_trigger_graph', default=True,
                        action='store_true',
                        help='Enable the trigger GAT stream.')
    parser.add_argument('--no_trigger_graph', dest='use_trigger_graph',
                        action='store_false',
                        help='Ablate the complete trigger GAT stream.')
      
    args = parser.parse_args()

    if args.inference_only:
        args.output_dir = "/".join(args.inference_model_path.split("/")[:-1])
    return args
