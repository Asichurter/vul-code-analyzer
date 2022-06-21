local data_vol_base_path = '/data1/zhijietang/vul_data/datasets/joern_vulberta/packed_data/';
local pretrained_model = 'microsoft/codebert-base';
local code_embed_dim = 768;
local code_encode_dim = 768;
local node_dim = 64;

local code_max_tokens = 256;
local max_lines = 50;
local additional_special_tokens = [];
local code_namespace = "code_tokens";

{
    extra: {
        version: 2,
        decs: {
            main: "separated edge prediction",
        },
    },

    dataset_reader: {
        type: "packed_line_pdg",
        code_tokenizer: {
            type: "pretrained_transformer",
            model_name: pretrained_model,
            max_length: code_max_tokens,
            tokenizer_kwargs: {
              additional_special_tokens: additional_special_tokens
            }
        },
        code_indexer: {
            type: "pretrained_transformer",
            model_name: pretrained_model,
            namespace: code_namespace,
            tokenizer_kwargs: {
              additional_special_tokens: additional_special_tokens
            }
        },
        volume_range: [39,39],
        pdg_max_vertice: max_lines,
        max_lines: max_lines,
        code_max_tokens: code_max_tokens,
        tokenized_newline_char: 'Ċ',
        special_tokenizer_token_handler_type: "codebert",
        only_keep_complete_lines: true,
        code_cleaner: {
            type: "space_sub",
        },
        unified_label: false,
    },

    train_data_path: data_vol_base_path,

    model: {
        type: "code_line_pdg_analyzer",
        code_embedder: {
          token_embedders: {
            code_tokens: {
              type: "pretrained_transformer",
              model_name: pretrained_model,
              train_parameters: true,
              tokenizer_kwargs: {
                additional_special_tokens: additional_special_tokens
             }
            }
          }
        },
        code_encoder: {
            type: "pass_through",
            input_dim: code_embed_dim,
        },
        node_encoder: {
            type: "mlp",
            input_dim: code_encode_dim,
            output_dim: node_dim,
            hidden_dims: [256],
            activation: 'gelu',
            dropout: 0.1
        },
        line_extractor: {
            type: "avg",
            max_lines: max_lines
        },
        struct_decoder: {
            type: "bilinear_separated",
            input_dim: node_dim,
        },
        loss_sampler: {
            type: "separated_balanced",
            loss_func: {
                type: "bce"
            }
        },
        drop_tokenizer_special_token_type: "codebert"
    },

  data_loader: {
    batch_size: 64,
    shuffle: true,
  },
  trainer: {
    num_epochs: 10,
    patience: null,
    cuda_device: 1,
    validation_metric: "-loss",
    optimizer: {
      type: "adam",
      lr: 1e-5
    },
    num_gradient_accumulation_steps: 2,
    callbacks: [
      { type: "epoch_print" },
      { type: "model_param_stat" },
      {
        type: "save_jsonnet_config",
        file_src: 'config/jsonnet/test_line_pdg_pretrain.jsonnet',
      },
      {
        type: "save_epoch_model",
        save_epoch_points: []
      },
    ],
    checkpointer: null,     // checkpointer is set to null to avoid saving model state at each episode
  },
}