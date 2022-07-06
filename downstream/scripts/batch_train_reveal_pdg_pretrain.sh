echo "train ds_reveal_ver_3"
python ../train_from_config.py -config /data1/zhijietang/temp/vul_temp/config/ds_reveal_ver_3.jsonnet
echo "test ds_reveal_ver_3"
python ../eval/evaluate_reveal.py -subset split_0 -version 3 -data_file_name test.json -model_name model.tar.gz -cuda 2

echo "train ds_reveal_ver_7"
python ../train_from_config.py -config /data1/zhijietang/temp/vul_temp/config/ds_reveal_ver_7.jsonnet
echo "test ds_reveal_ver_7"
python ../eval/evaluate_reveal.py -subset split_1 -version 7 -data_file_name test.json -model_name model.tar.gz -cuda 2

echo "train ds_reveal_ver_11"
python ../train_from_config.py -config /data1/zhijietang/temp/vul_temp/config/ds_reveal_ver_11.jsonnet
echo "test ds_reveal_ver_11"
python ../eval/evaluate_reveal.py -subset split_2 -version 11 -data_file_name test.json -model_name model.tar.gz -cuda 2

echo "train ds_reveal_ver_15"
python ../train_from_config.py -config /data1/zhijietang/temp/vul_temp/config/ds_reveal_ver_15.jsonnet
echo "test ds_reveal_ver_15"
python ../eval/evaluate_reveal.py -subset split_3 -version 15 -data_file_name test.json -model_name model.tar.gz -cuda 2

echo "train ds_reveal_ver_19"
python ../train_from_config.py -config /data1/zhijietang/temp/vul_temp/config/ds_reveal_ver_19.jsonnet
echo "test ds_reveal_ver_19"
python ../eval/evaluate_reveal.py -subset split_4 -version 19 -data_file_name test.json -model_name model.tar.gz -cuda 2