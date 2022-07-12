local data_vol_base_path = '/data1/zhijietang/vul_data/datasets/joern_vulberta/packed_vol_data/';
local pretrained_model = 'microsoft/codebert-base';
local code_embed_dim = 768;
local code_encode_dim = 768;
local node_dim = 64;
local vocab_size = 50265;

local code_max_tokens = 256;
local max_lines = 50;
local code_namespace = "code_tokens";

local tokenizer_type = "codebert";
local mlm_mask_token = "<MLM>";
local additional_special_tokens = [mlm_mask_token];

{
    extra: {
        version: -1,
        decs: {
            main: "mlm (no neg_sampling, mlm_coeff=1, fix logits bug) + separated edge prediction",
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
        volume_range: [66,66]
    },
    validation_data_path: {
        data_base_path: data_vol_base_path,
        volume_range: [69,69]
    },

    model: {
        type: "from_archive",
        archive_file: "/data1/zhijietang/vul_data/run_logs/pretrain/12/model.tar.gz"
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
    cuda_device: 1,
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
        file_src: 'config/jsonnet/test_debug.jsonnet',
      },
      {
        type: "save_epoch_state",
        save_epoch_points: [4,9,14]
      },
    ],
    checkpointer: null,     // checkpointer is set to null to avoid saving model state at each episode
  },
}