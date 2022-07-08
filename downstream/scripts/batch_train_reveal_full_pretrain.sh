#echo "train ds_reveal_ver_26"
python ../train_from_config.py -config /data1/zhijietang/temp/vul_temp/config/ds_reveal_ver_26.jsonnet
#echo "test ds_reveal_ver_26"
python ../eval/evaluate_reveal.py -subset split_0 -version 26 -data_file_name test.json -model_name model.tar.gz -cuda 2

#echo "train ds_reveal_ver_27"
python ../train_from_config.py -config /data1/zhijietang/temp/vul_temp/config/ds_reveal_ver_27.jsonnet
#echo "test ds_reveal_ver_27"
python ../eval/evaluate_reveal.py -subset split_1 -version 27 -data_file_name test.json -model_name model.tar.gz -cuda 2

#echo "train ds_reveal_ver_28"
python ../train_from_config.py -config /data1/zhijietang/temp/vul_temp/config/ds_reveal_ver_28.jsonnet
#echo "test ds_reveal_ver_28"
python ../eval/evaluate_reveal.py -subset split_2 -version 28 -data_file_name test.json -model_name model.tar.gz -cuda 2

#echo "train ds_reveal_ver_29"
python ../train_from_config.py -config /data1/zhijietang/temp/vul_temp/config/ds_reveal_ver_29.jsonnet
#echo "test ds_reveal_ver_29"
python ../eval/evaluate_reveal.py -subset split_3 -version 29 -data_file_name test.json -model_name model.tar.gz -cuda 2

#echo "train ds_reveal_ver_30"
python ../train_from_config.py -config /data1/zhijietang/temp/vul_temp/config/ds_reveal_ver_30.jsonnet
#echo "test ds_reveal_ver_30"
python ../eval/evaluate_reveal.py -subset split_4 -version 30 -data_file_name test.json -model_name model.tar.gz -cuda 2
