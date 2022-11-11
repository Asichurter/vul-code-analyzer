local cv_data_base_path = '/data1/zhijietang/vul_data/datasets/devign/splits/';
local load_model_base_path = '/data1/zhijietang/vul_data/run_logs/pretrain/12/';
local state_dict_file_name = 'state_epoch_9.th';
local pretrained_model = 'microsoft/codebert-base';
local code_embed_dim = 768;
local code_encode_dim = 768;
local code_out_dim = 768;

local code_max_tokens = 256;
local code_namespace = "code_tokens";
local tokenizer_type = "codebert";

local mlm_mask_token = "<MLM>";
local additional_special_tokens = [mlm_mask_token];     # Add this special token to avoid embedding size mismatch
local split_index = 0;
local cuda_device = 0;

local cv_base_path = cv_data_base_path + "split_" + split_index + "/";

{
    extra: {
        version: 1,
        decs: {
            data: "Fan split_0",
            pretrain: "Ver.12 E9, small data, MLM + line PDG",
        },
    },

    vocabulary: {
        type: "from_pretrained_transformer",
        model_name: pretrained_model,
        namespace: code_namespace
    },

    dataset_reader: {
        type: "fan_vul_detect_base",
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
        code_max_tokens: code_max_tokens,
        code_namespace: code_namespace,
        code_cleaner: { type: "pre_line_truncate", max_line: 200},
        tokenizer_type: tokenizer_type
    },

    train_data_path: cv_base_path + "train.pkl",
    validation_data_path: cv_base_path + "validate.pkl",

    model: {
        type: "vul_func_predictor",
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
        code_feature_squeezer: {
            type: "cls_pooler",
            embedding_dim: code_embed_dim,
        },
        loss_func: {
            type: "bce"
        },
        classifier: {
            type: "linear_sigmoid",
            in_feature_dim: code_out_dim,
            hidden_dims: [256],
            activations: ["relu"],
            dropouts: [0.3],
            ahead_feature_dropout: 0.3,
        },
        metric: {
            type: "f1",
            positive_label: 1,
        },
    },

  data_loader: {
    batch_size: 64,
    shuffle: true,
  },
  validation_data_loader: {
    batch_size: 64,
    shuffle: true,
  },

  trainer: {
    num_epochs: 15,
    patience: null,
    cuda_device: cuda_device,
    validation_metric: "+f1",
    optimizer: {
      type: "adam",
      lr: 1e-5
    },
    num_gradient_accumulation_steps: 1,
    callbacks: [
      { type: "epoch_print" },
      { type: "model_param_stat" },
      {
        type: "save_jsonnet_config",
        file_src: '/data1/zhijietang/temp/vul_temp/config/devign/ds_devign_ver_1.jsonnet',
      },
      {
        type: "save_epoch_model",
        save_epoch_points: []
      },
      {
        type: "partial_load_state_dict",
        load_state_dict_path: load_model_base_path + state_dict_file_name,
        load_prefix_remap: {
            "code_embedder": "code_embedder"
        },
      },
    ],
    checkpointer: null,     // checkpointer is set to null to avoid saving model state at each episode
  },
}