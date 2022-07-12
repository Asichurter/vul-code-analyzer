python ../train_from_config.py -config /data1/zhijietang/temp/vul_temp/config/ds_reveal_ver_36.jsonnet
python ../eval/evaluate_reveal.py -subset split_0 -version 36 -data_file_name test.json -model_name model.tar.gz -cuda 0

python ../train_from_config.py -config /data1/zhijietang/temp/vul_temp/config/ds_reveal_ver_37.jsonnet
python ../eval/evaluate_reveal.py -subset split_1 -version 37 -data_file_name test.json -model_name model.tar.gz -cuda 0

python ../train_from_config.py -config /data1/zhijietang/temp/vul_temp/config/ds_reveal_ver_38.jsonnet
python ../eval/evaluate_reveal.py -subset split_2 -version 38 -data_file_name test.json -model_name model.tar.gz -cuda 0

python ../train_from_config.py -config /data1/zhijietang/temp/vul_temp/config/ds_reveal_ver_39.jsonnet
python ../eval/evaluate_reveal.py -subset split_3 -version 39 -data_file_name test.json -model_name model.tar.gz -cuda 0

python ../train_from_config.py -config /data1/zhijietang/temp/vul_temp/config/ds_reveal_ver_40.jsonnet
python ../eval/evaluate_reveal.py -subset split_4 -version 40 -data_file_name test.json -model_name model.tar.gz -cuda 0