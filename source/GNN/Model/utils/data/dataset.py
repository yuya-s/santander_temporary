import numpy as np
import glob
import re
from scipy.io import mmread
import os
import sys
current_dir = os.path.dirname(os.path.abspath("__file__"))
sys.path.append( str(current_dir) + '/../../' )
from utils.setting_param import Dataset_max_nnz_am, Dataset_max_nnz_label_edge, Dataset_attribute_dim, Dataset_adj_shape, Dataset_ratio_test

max_nnz_am = Dataset_max_nnz_am  # 隣接疎行列の全サンプルにおける非ゼロ要素数の最大値
max_nnz_label_edge = Dataset_max_nnz_label_edge  # label_edgeの全サンプルにおける非ゼロ要素数の最大値
attribute_dim = Dataset_attribute_dim
ratio_test = Dataset_ratio_test
adj_shape = Dataset_adj_shape

def load_paths_from_dir(dir_path):
    # dir 以下のファイル名のリストを取得
    path_list = glob.glob(dir_path + "/*")
    # ソート (ゼロ埋めされていない数字の文字列のソート)
    path_list = np.array(sorted(path_list, key=lambda s: int(re.findall(r'\d+', s)[-1])))
    return path_list

def train_test_split(n_samples, ratio_test):
    idx = list(range(n_samples))
    n_test = int(n_samples * ratio_test)
    return idx[:n_samples - n_test], idx[-n_test:]

def coo_scipy2coo_numpy(coo_scipy, max_nnz):
    # scipy_cooをvalue, indicesに分解してnumpy配列に変換
    # バッチ内のサイズを揃えるためにmax_nnzでゼロパディング
    coo_numpy = np.zeros((3, max_nnz))
    coo_numpy[:, :len(coo_scipy.data)] = np.vstack((coo_scipy.data, coo_scipy.row, coo_scipy.col))
    return coo_numpy

def in_out_generate(coo_numpy, n_node):
    coo_numpy_in = coo_numpy.copy()
    coo_numpy_out = np.zeros_like(coo_numpy)
    coo_numpy_out[0] = coo_numpy[0]
    coo_numpy_out[1] = coo_numpy[2] % n_node
    coo_numpy_out[2] = (coo_numpy[2] // n_node) * n_node + coo_numpy[1]
    return np.stack((coo_numpy_in, coo_numpy_out))

class SantanderDataset():
    def __init__(self, training_idx, path, L, is_train, is_valid):
        # 入力ファイルのPATHのリストを取得
        attribute_paths = load_paths_from_dir(path + '/time_series_node_attribute')
        adjacency_paths = load_paths_from_dir(path + '/time_series_adjacency')
        #label_edge_paths = load_paths_from_dir(path + '/label_edge')
        label_attribute_paths = load_paths_from_dir(path + '/label_node_attribute')
        #label_node_paths = load_paths_from_dir(path + '/label_node')
        #label_lost_paths = np.array([path for path in label_node_paths if 'label_lost' in path])
        #label_return_paths = np.array([path for path in label_node_paths if 'label_return' in path])

        # split data
        n_samples = len(label_attribute_paths)
        #train_idx = list(range(n_samples))[24626-8500:-3000]
        train_idx = list(range(n_samples))[training_idx-8500:-3000]
        valid_idx = list(range(n_samples))[-3000:-1000]
        test_idx = list(range(n_samples))[-1000:]
        if is_train:
            target_idx = train_idx
        elif is_valid:
            target_idx = valid_idx
        else:
            target_idx = test_idx

        # ファイル読み込み(scipy cooはDataLoaderのサポート外なので変換する)
        self.idx_list = target_idx
        self.adjacency_paths = adjacency_paths[target_idx]
        self.attribute_paths = attribute_paths[target_idx]
        self.label_attribute_paths = label_attribute_paths[target_idx]

        # 入力グラフの統計量
        self.L = L
        self.n_node = adj_shape[0]
        self.n_edge_types = adj_shape[1] // adj_shape[0]

    def __getitem__(self, index):
        sample_idx = self.idx_list[index]
        annotation = np.array(mmread(self.attribute_paths[index]).todense()).reshape((adj_shape[0], self.L, attribute_dim))[:,:,2:] # PrimaryAttribute(dummy0かdummy1か = dim0, dim1)を捨てる。label_attributeもdim2のみ。
        am = in_out_generate(coo_scipy2coo_numpy(mmread(self.adjacency_paths[index]), max_nnz_am), adj_shape[0])
        label_attribute = np.load(self.label_attribute_paths[index])[:, :, 2:].reshape(8, adj_shape[0], 5).transpose((1, 0, 2))
        return sample_idx, annotation, am, label_attribute

    def __len__(self):
        return len(self.idx_list)