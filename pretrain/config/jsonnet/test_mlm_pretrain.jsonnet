local data_vol_base_path = '/data1/zhijietang/vul_data/datasets/joern_vulberta/packed_vol_data/';
local pretrained_model = 'microsoft/codebert-base';
local code_embed_dim = 768;
local code_encode_dim = 768;
local node_dim = 64;
local vocab_size = 50625;

local code_max_tokens = 256;
local max_lines = 50;
local code_namespace = "code_tokens";

local tokenizer_type = "codebert";
local mlm_mask_token = "<MLM>";
local additional_special_tokens = [mlm_mask_token];

{
    extra: {
        version: 15,
        decs: {
            main: "Only mlm (no neg_sampling_k, mlm_coeff=1, fix logits bug)",
            vol: "train: 30~66, val: 67~69",
            training: "20 epoch, lr=1e-4, poly_decay, min_lr=1e-6, no warmup",
        },
    },

    vocabulary: {
        type: "from_pretrained_transformer",
        model_name: pretrained_model,
        namespace: code_namespace
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
        pdg_max_vertice: max_lines,
        max_lines: max_lines,
        code_max_tokens: code_max_tokens,
        tokenized_newline_char: 'Ċ',
        special_tokenizer_token_handler_type: tokenizer_type,
        only_keep_complete_lines: true,
        code_cleaner: {
            type: "space_sub",
        },
        unified_label: false,
        from_raw_data: false,
    },

    train_data_path: {
        data_base_path: data_vol_base_path,
        volume_range: [30,66]
    },
    validation_data_path: {
        data_base_path: data_vol_base_path,
        volume_range: [67,69]
    },

    model: {
        type: "code_objective_trainer",
        code_objectives: [
            {
                type: "mlm",
                name: "mlm",
                code_namespace: code_namespace,
                token_dimension: code_embed_dim,
                as_code_embedder: true,
                tokenizer_type: tokenizer_type,
                token_id_key: "token_ids",
                mask_token: mlm_mask_token,
                vocab_size: vocab_size,
                loss_coeff: 1,
                dropout: 0.1,
                activation: "relu",
                sample_ratio: 0.15,
                mask_ratio: 0.8,
                replace_ratio: 0.1,
                negative_sampling_k: null,
            }
        ],

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
        drop_tokenizer_special_token_type: tokenizer_type,
    },

  data_loader: {
    batch_size: 32,
    shuffle: true,
  },
  validation_data_loader: {
    batch_size: 32,
    shuffle: true,
  },

  trainer: {
    num_epochs: 20,
    patience: null,
    cuda_device: 2,
    validation_metric: "-loss",
    optimizer: {
      type: "adam",
      lr: 1e-4
    },
    learning_rate_scheduler: {
        type: "polynomial_decay",
        power: 2,
        warmup_steps: 0,
        end_learning_rate: 1e-6
    },
    num_gradient_accumulation_steps: 2,
    callbacks: [
      { type: "epoch_print" },
      { type: "model_param_stat" },
      {
        type: "save_jsonnet_config",
        file_src: 'config/jsonnet/test_mlm_pretrain.jsonnet',
      },
      {
        type: "save_epoch_state",
        save_epoch_points: [4,9,14]
      },
    ],
    checkpointer: null,     // checkpointer is set to null to avoid saving model state at each episode
  },
}