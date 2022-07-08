#echo "train ds_reveal_ver_1"
python ../train_from_config.py -config /data1/zhijietang/temp/vul_temp/config/ds_reveal_ver_31.jsonnet
#echo "test ds_reveal_ver_1"
python ../eval/evaluate_reveal.py -subset split_0 -version 31 -data_file_name test.json -model_name model.tar.gz -cuda 0

#echo "train ds_reveal_ver_5"
python ../train_from_config.py -config /data1/zhijietang/temp/vul_temp/config/ds_reveal_ver_32.jsonnet
#echo "test ds_reveal_ver_5"
python ../eval/evaluate_reveal.py -subset split_1 -version 32 -data_file_name test.json -model_name model.tar.gz -cuda 0

#echo "train ds_reveal_ver_9"
python ../train_from_config.py -config /data1/zhijietang/temp/vul_temp/config/ds_reveal_ver_33.jsonnet
#echo "test ds_reveal_ver_9"
python ../eval/evaluate_reveal.py -subset split_2 -version 33 -data_file_name test.json -model_name model.tar.gz -cuda 0

#echo "train ds_reveal_ver_13"
python ../train_from_config.py -config /data1/zhijietang/temp/vul_temp/config/ds_reveal_ver_34.jsonnet
#echo "test ds_reveal_ver_13"
python ../eval/evaluate_reveal.py -subset split_3 -version 34 -data_file_name test.json -model_name model.tar.gz -cuda 0

#echo "train ds_reveal_ver_17"
python ../train_from_config.py -config /data1/zhijietang/temp/vul_temp/config/ds_reveal_ver_35.jsonnet
#echo "test ds_reveal_ver_17"
python ../eval/evaluate_reveal.py -subset split_4 -version 35 -data_file_name test.json -model_name model.tar.gz -cuda 0