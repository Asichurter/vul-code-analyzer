local cv_data_base_path = '/data1/zhijietang/vul_data/datasets/reveal/small/random_split/';
local load_model_base_path = '/data1/zhijietang/vul_data/run_logs/pretrain/15/';
local state_dict_file_name = 'state_epoch_9.th';
local pretrained_model = 'microsoft/codebert-base';
local code_embed_dim = 768;
local code_encode_dim = 768;
local code_out_dim = 768;

local code_max_tokens = 256;
local code_namespace = "code_tokens";

local mlm_mask_token = "<MLM>";
local additional_special_tokens = [mlm_mask_token];     # Add this special token to avoid embedding size mismatch
local split_index = 0;

local cv_base_path = cv_data_base_path + "split_" + split_index + "/";

{
    extra: {
        version: 1003,
        decs: {
            main: "reveal small-256, random_split 0 + pretrain Ver.20, epoch=best (mlm+10epoch pdg)",
            embedder: "Fixed embedder after loading",
            batch: "train batch_size = 64",
        },
    },

    vocabulary: {
        type: "from_pretrained_transformer",
        model_name: pretrained_model,
        namespace: code_namespace
    },

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
              train_parameters: false,
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
        },
    },

  data_loader: {
    batch_size: 64,
    shuffle: true,
//    batch_sampler: {
//        type: "binary_duplicate_balanced_rand",
//        batch_size: 32,
//        majority_label_index: 0,
//        label_key: 'label',
//        major_instance_ratio_in_batch: 0.5,
//    }
  },
  validation_data_loader: {
    batch_size: 64,
    shuffle: true,
  },

  trainer: {
    num_epochs: 10,
    patience: null,
    cuda_device: 1,
    validation_metric: "+f1",
    optimizer: {
      type: "adam",
      lr: 5e-5
    },
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
        load_state_dict_path: load_model_base_path + state_dict_file_name,
        load_prefix_remap: {
            "code_embedder": "code_embedder"
        },
      },
    ],
    checkpointer: null,     // checkpointer is set to null to avoid saving model state at each episode
  },
}