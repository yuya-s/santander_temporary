ROOT_PATH = "/SantanderParking-RNN-CNN"

L = 48
init_new_edge_num = 3
edge_type_num = 1
known_node_num = 27 #len(pd.read_csv(ROOT_PATH + "/data/graph/node_idx.csv"))
all_node_num = 27 #len(pd.read_csv(ROOT_PATH + "/data/learning_data/node_idx.csv"))

ratio_test = 0.2
n_samples = 1986 #len(glob.glob(ROOT_PATH + "/data/learning_data/time_series_node_attribute/*"))
n_test = int(n_samples * ratio_test)
n_train = n_samples - n_test

#max_nnz_am = 178 * (L+1) # 隣接疎行列の全サンプルにおける非ゼロ要素数の最大値(式はSantanderの場合)
max_nnz_am = 178 # 隣接疎行列の全サンプルにおける非ゼロ要素数の最大値(式はSantanderの場合)
max_nnz_label_edge = 178 # label_edgeの全サンプルにおける非ゼロ要素数の最大値
attribute_dim = 3
#adj_shape = (all_node_num, all_node_num * (L+1) * edge_type_num)
adj_shape = (all_node_num, all_node_num)

worker = 2
batchSize = 1
state_dim = 5
output_dim = 1
n_steps = 5
init_L = L
niter = 100
lr = 0.01

##################################################################
# ディレクトリ名の定義

# MakeSample IO dir
MakeSample_MakeGraph_InputDir = ROOT_PATH + "/data"
MakeSample_MakeGraph_OutputDir = ROOT_PATH + "/data/forGNN/graph"
MakeSample_Preprocessing_InputDir = ROOT_PATH + "/data/forGNN/graph"
MakeSample_Preprocessing_OutputDir = ROOT_PATH + "/data/forGNN/graph"
MakeSample_main_InputDir = ROOT_PATH + "/data/forGNN/graph"
MakeSample_main_OutputDir = ROOT_PATH + "/data/forGNN/learning_data/forGNN"

# Model IO dir
Model_InputDir = ROOT_PATH + "/data/forGNN/learning_data"

# Encoding Table Path
NodeTablePath = ROOT_PATH + "/data/forGNN/graph/node_idx.csv"
TSTablePath = ROOT_PATH + "/data/forGNN/graph/ts_idx.csv"
AttributeTablePath = ROOT_PATH + "/data/forGNN/graph/attribute_idx.csv"

##################################################################
# 各種パラメータ

# MakeSample param
MakeSample_main_L = L
MakeSample_main_new_edge_num = init_new_edge_num
MakeSample_main_edge_type_num = edge_type_num

# Evaluation param
Evaluation_L = L
Evaluation_node_num = all_node_num
Evaluation_start = list(range(n_samples))[-n_test:][0] + L
Evaluation_end = list(range(n_samples))[-n_test:][-1] + L + 1

# Dataset() param
Dataset_max_nnz_am = max_nnz_am
Dataset_max_nnz_label_edge = max_nnz_label_edge
Dataset_attribute_dim = attribute_dim
Dataset_adj_shape = adj_shape
Dataset_ratio_test = ratio_test

# Model
Model_main_worker = worker
Model_main_batchSize = batchSize
Model_main_state_dim = state_dim
Model_main_output_dim = output_dim
Model_main_n_steps = n_steps
Model_main_init_L = init_L
Model_main_niter = niter
Model_main_lr = lr