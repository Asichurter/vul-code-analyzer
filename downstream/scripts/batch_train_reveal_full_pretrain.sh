echo "train ds_reveal_ver_2"
python ../train_from_config.py -config /data1/zhijietang/temp/vul_temp/config/ds_reveal_ver_2.jsonnet
echo "test ds_reveal_ver_2"
python ../eval/evaluate_reveal.py -subset split_0 -version 2 -data_file_name test.json -model_name model.tar.gz -cuda 1

echo "train ds_reveal_ver_6"
python ../train_from_config.py -config /data1/zhijietang/temp/vul_temp/config/ds_reveal_ver_6.jsonnet
echo "test ds_reveal_ver_6"
python ../eval/evaluate_reveal.py -subset split_1 -version 6 -data_file_name test.json -model_name model.tar.gz -cuda 1

echo "train ds_reveal_ver_10"
python ../train_from_config.py -config /data1/zhijietang/temp/vul_temp/config/ds_reveal_ver_10.jsonne
echo "test ds_reveal_ver_10"
python ../eval/evaluate_reveal.py -subset split_2 -version 10 -data_file_name test.json -model_name model.tar.gz -cuda 1

echo "train ds_reveal_ver_14"
python ../train_from_config.py -config /data1/zhijietang/temp/vul_temp/config/ds_reveal_ver_14.jsonne
echo "test ds_reveal_ver_14"
python ../eval/evaluate_reveal.py -subset split_3 -version 14 -data_file_name test.json -model_name model.tar.gz -cuda 1

echo "train ds_reveal_ver_18"
python ../train_from_config.py -config /data1/zhijietang/temp/vul_temp/config/ds_reveal_ver_18.jsonne
echo "test ds_reveal_ver_18"
python ../eval/evaluate_reveal.py -subset split_4 -version 18 -data_file_name test.json -model_name model.tar.gz -cuda 1
