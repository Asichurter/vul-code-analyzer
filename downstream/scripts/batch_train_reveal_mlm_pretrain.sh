echo "train ds_reveal_ver_4"
python ../train_from_config.py -config /data1/zhijietang/temp/vul_temp/config/ds_reveal_ver_4.jsonnet
echo "test ds_reveal_ver_4"
python ../eval/evaluate_reveal.py -subset split_0 -version 4 -data_file_name test.json -model_name model.tar.gz -cuda 3

echo "train ds_reveal_ver_8"
python ../train_from_config.py -config /data1/zhijietang/temp/vul_temp/config/ds_reveal_ver_8.jsonnet
echo "test ds_reveal_ver_8"
python ../eval/evaluate_reveal.py -subset split_1 -version 8 -data_file_name test.json -model_name model.tar.gz -cuda 3

echo "train ds_reveal_ver_12"
python ../train_from_config.py -config /data1/zhijietang/temp/vul_temp/config/ds_reveal_ver_12.jsonnet
echo "test ds_reveal_ver_12"
python ../eval/evaluate_reveal.py -subset split_2 -version 12 -data_file_name test.json -model_name model.tar.gz -cuda 3

echo "train ds_reveal_ver_16"
python ../train_from_config.py -config /data1/zhijietang/temp/vul_temp/config/ds_reveal_ver_16.jsonnet
echo "test ds_reveal_ver_16"
python ../eval/evaluate_reveal.py -subset split_3 -version 16 -data_file_name test.json -model_name model.tar.gz -cuda 3

echo "train ds_reveal_ver_20"
python ../train_from_config.py -config /data1/zhijietang/temp/vul_temp/config/ds_reveal_ver_20.jsonnet
echo "test ds_reveal_ver_20"
python ../eval/evaluate_reveal.py -subset split_4 -version 20 -data_file_name test.json -model_name model.tar.gz -cuda 3