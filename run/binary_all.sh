small_dir="./small_graphs"
all_dir="./all_graphs"
data_type="all"
output_dim=2
mv=0
#GCN
m1="GCN"
gcn_lr=0.0001
gcn_hl=512
python3 ../train_twitter_MPNN.py --model $m1 --data_type $data_type --multivariate $mv --hidden_dim $gcn_hl --lr $gcn_lr --output_dim $output_dim --small_graphs_path $small_dir --all_graphs_path $all_dir

#GAT
m2="GAT"
gat_lr=0.0001
gat_hl=512
python3 ../train_twitter_MPNN.py --model $m2 --data_type $data_type --multivariate $mv --hidden_dim $gat_hl --lr $gat_lr --output_dim $output_dim --small_graphs_path $small_dir --all_graphs_path $all_dir

#GIN
m3="GIN"
gin_lr=0.0001
gin_hl=512
python3 ../train_twitter_MPNN.py --model $m3 --data_type $data_type --multivariate $mv --hidden_dim $gin_hl --lr $gin_lr --output_dim $output_dim --small_graphs_path $small_dir --all_graphs_path $all_dir

#GINE
m4="GINE"
gine_lr=0.0001
gine_hl=128
python3 ../train_twitter_MPNN.py --model $m4 --data_type $data_type --multivariate $mv --hidden_dim $gine_hl --lr $gine_lr --output_dim $output_dim --small_graphs_path $small_dir --all_graphs_path $all_dir