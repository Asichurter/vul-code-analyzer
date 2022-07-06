local cv_data_base_path = '/data1/zhijietang/vul_data/datasets/reveal/random_split/';
local load_model_base_path = '/data1/zhijietang/vul_data/run_logs/pretrain/12/';
local pretrained_model = 'microsoft/codebert-base';
local code_embed_dim = 768;
local code_encode_dim = 768;
local code_out_dim = 768;

local code_max_tokens = 256;
local code_namespace = "code_tokens";

local mlm_mask_token = "<MLM>";
local additional_special_tokens = [mlm_mask_token];     # Add this special token to avoid embedding size mismatch
local split_index = 1;

local cv_base_path = cv_data_base_path + "split_" + split_index + "/";

{
    extra: {
        version: -2,
        decs: {
            main: "reveal random_split 1 + pretrain Ver.12 (no neg_samp, pdg+mlm)",
            sampler: "No balancer",
            extra: "no vocab config",
        },
    },

//    vocabulary: {
//        type: "from_pretrained_transformer",
//        model_name: pretrained_model,
//        namespace: code_namespace
//    },

//    vocabulary: {
//        type: "from_files",
//        directory: load_model_base_path + "vocabulary"
//    },

    dataset_reader: {
        type: "reveal_base",
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
        code_cleaner: {
            type: "space_sub",
        },
    },

    train_data_path: cv_base_path + "train.json",
    validation_data_path: cv_base_path + "validate.json",

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
            embedding_dim: code_encode_dim,
            cls_is_last_token: false,
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
//            top_k: 1
        },
    },

  data_loader: {
    batch_size: 16,
    shuffle: true,
  },
  validation_data_loader: {
    batch_size: 16,
    shuffle: true,
  },

  trainer: {
    num_epochs: 10,
    patience: null,
    cuda_device: 2,
    validation_metric: "+f1",
    optimizer: {
      type: "adam",
      lr: 1e-5
    },
//    learning_rate_scheduler: {
//        type: "polynomial_decay",
//        power: 2,
//        warmup_steps: 0,
//        end_learning_rate: 1e-6
//    },
    num_gradient_accumulation_steps: 1,
    callbacks: [
      { type: "epoch_print" },
      { type: "model_param_stat" },
      {
        type: "save_jsonnet_config",
        file_src: 'config/jsonnet/ds_reveal_vul_predict.jsonnet',
      },
      {
        type: "save_epoch_model",
        save_epoch_points: []
      },
      {
        type: "partial_load_state_dict",
        load_state_dict_path: load_model_base_path + "best.th",
        load_prefix_remap: {
            code_embedder: "code_embedder"
        },
      },
    ],
    checkpointer: null,     // checkpointer is set to null to avoid saving model state at each episode
  },
}