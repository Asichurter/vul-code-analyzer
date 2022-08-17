#echo "task 1/7"
#python ../model/devign/main.py --do_val --dataset reveal_mlm_0 --input_dir /data1/zhijietang/vul_data/datasets/reveal/small/devign/mlm_ver15_e9/rs_0/ --feature_size 768 --graph_embed_size 768 --num_steps 6 --batch_size 64 --model_type cls --lr 0.00002 --patience 50 --res_forward True --dynamic_node False --dump_key cls_rs_0 --cuda 2
#echo "task 2/7"
#python ../model/devign/main.py --do_val --dataset reveal_mlm_0 --input_dir /data1/zhijietang/vul_data/datasets/reveal/small/devign/mlm_ver15_e9/rs_0/ --feature_size 768 --graph_embed_size 768 --num_steps 6 --batch_size 64 --model_type cls --lr 0.00002 --patience 50 --res_forward True --dynamic_node False --dump_key cls_rs_0 --cuda 2
#
#echo "task 3/7"
#python ../model/devign/main.py --do_val --dataset reveal_mlm_1 --input_dir /data1/zhijietang/vul_data/datasets/reveal/small/devign/mlm_ver15_e9/rs_1/ --feature_size 768 --graph_embed_size 768 --num_steps 6 --batch_size 64 --model_type cls --lr 0.00002 --patience 50 --res_forward True --dynamic_node False --dump_key cls_rs_1 --cuda 2
#
#echo "task 4/7"
#python ../model/devign/main.py --do_val --dataset reveal_mlm_2 --input_dir /data1/zhijietang/vul_data/datasets/reveal/small/devign/mlm_ver15_e9/rs_2/ --feature_size 768 --graph_embed_size 768 --num_steps 6 --batch_size 64 --model_type cls --lr 0.00002 --patience 50 --res_forward True --dynamic_node False --dump_key cls_rs_2 --cuda 2
#echo "task 5/7"
#python ../model/devign/main.py --do_val --dataset reveal_mlm_2 --input_dir /data1/zhijietang/vul_data/datasets/reveal/small/devign/mlm_ver15_e9/rs_2/ --feature_size 768 --graph_embed_size 768 --num_steps 6 --batch_size 64 --model_type cls --lr 0.00002 --patience 50 --res_forward True --dynamic_node False --dump_key cls_rs_2 --cuda 2
#
#echo "task 6/7"
#python ../model/devign/main.py --do_val --dataset reveal_mlm_3 --input_dir /data1/zhijietang/vul_data/datasets/reveal/small/devign/mlm_ver15_e9/rs_3/ --feature_size 768 --graph_embed_size 768 --num_steps 6 --batch_size 64 --model_type cls --lr 0.00002 --patience 50 --res_forward True --dynamic_node False --dump_key cls_rs_3 --cuda 2
#echo "task 7/7"
#python ../model/devign/main.py --do_val --dataset reveal_mlm_3 --input_dir /data1/zhijietang/vul_data/datasets/reveal/small/devign/mlm_ver15_e9/rs_3/ --feature_size 768 --graph_embed_size 768 --num_steps 6 --batch_size 64 --model_type cls --lr 0.00002 --patience 50 --res_forward True --dynamic_node False --dump_key cls_rs_3 --cuda 2

echo "task 1/4 (Loop 1)"
python ../model/devign/main.py --do_val --dataset reveal_mlm_0 --input_dir /data1/zhijietang/vul_data/datasets/reveal/small/devign/mlm_ver15_e9/rs_0/ --feature_size 768 --graph_embed_size 768 --num_steps 6 --batch_size 64 --model_type node_mean --lr 0.00002 --patience 50 --res_forward False --dynamic_node True --dump_key node_mean_rs_0 --cuda 5
echo "task 2/4 (Loop 1)"
python ../model/devign/main.py --do_val --dataset reveal_mlm_1 --input_dir /data1/zhijietang/vul_data/datasets/reveal/small/devign/mlm_ver15_e9/rs_1/ --feature_size 768 --graph_embed_size 768 --num_steps 6 --batch_size 64 --model_type node_mean --lr 0.00002 --patience 50 --res_forward False --dynamic_node True --dump_key node_mean_rs_1 --cuda 5
echo "task 3/4 (Loop 1)"
python ../model/devign/main.py --do_val --dataset reveal_mlm_2 --input_dir /data1/zhijietang/vul_data/datasets/reveal/small/devign/mlm_ver15_e9/rs_2/ --feature_size 768 --graph_embed_size 768 --num_steps 6 --batch_size 64 --model_type node_mean --lr 0.00002 --patience 50 --res_forward False --dynamic_node True --dump_key node_mean_rs_2 --cuda 5
echo "task 4/4 (Loop 1)"
python ../model/devign/main.py --do_val --dataset reveal_mlm_3 --input_dir /data1/zhijietang/vul_data/datasets/reveal/small/devign/mlm_ver15_e9/rs_3/ --feature_size 768 --graph_embed_size 768 --num_steps 6 --batch_size 64 --model_type node_mean --lr 0.00002 --patience 50 --res_forward False --dynamic_node True --dump_key node_mean_rs_3 --cuda 5

echo "task 1/4 (Loop 2)"
python ../model/devign/main.py --do_val --dataset reveal_mlm_0 --input_dir /data1/zhijietang/vul_data/datasets/reveal/small/devign/mlm_ver15_e9/rs_0/ --feature_size 768 --graph_embed_size 768 --num_steps 6 --batch_size 64 --model_type node_mean --lr 0.00002 --patience 50 --res_forward False --dynamic_node True --dump_key node_mean_rs_0 --cuda 5
echo "task 2/4 (Loop 2)"
python ../model/devign/main.py --do_val --dataset reveal_mlm_1 --input_dir /data1/zhijietang/vul_data/datasets/reveal/small/devign/mlm_ver15_e9/rs_1/ --feature_size 768 --graph_embed_size 768 --num_steps 6 --batch_size 64 --model_type node_mean --lr 0.00002 --patience 50 --res_forward False --dynamic_node True --dump_key node_mean_rs_1 --cuda 5
echo "task 3/4 (Loop 2)"
python ../model/devign/main.py --do_val --dataset reveal_mlm_2 --input_dir /data1/zhijietang/vul_data/datasets/reveal/small/devign/mlm_ver15_e9/rs_2/ --feature_size 768 --graph_embed_size 768 --num_steps 6 --batch_size 64 --model_type node_mean --lr 0.00002 --patience 50 --res_forward False --dynamic_node True --dump_key node_mean_rs_2 --cuda 5
echo "task 4/4 (Loop 2)"
python ../model/devign/main.py --do_val --dataset reveal_mlm_3 --input_dir /data1/zhijietang/vul_data/datasets/reveal/small/devign/mlm_ver15_e9/rs_3/ --feature_size 768 --graph_embed_size 768 --num_steps 6 --batch_size 64 --model_type node_mean --lr 0.00002 --patience 50 --res_forward False --dynamic_node True --dump_key node_mean_rs_3 --cuda 5