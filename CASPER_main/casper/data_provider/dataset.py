import os
import pandas as pd
import numpy as np
import scanpy as sc
import torch
from torch.utils.data.dataset import Dataset
import scipy.sparse
import random
from pathlib import Path

class H5ADReader:
    def __init__(self, args, path_list):
        self.args = args
        self.pert_col = args.pert_col
        self.control_name = args.control_name
        self.path_list = path_list
        self.data = {}
        
        dist_path = os.path.join(args.dataset_dir, args.dist_graph)
        adj_path = os.path.join(args.dataset_dir, args.adj_graph)

        mtx = pd.read_csv(dist_path, header=0, index_col=0)
        mtx_adj = pd.read_csv(adj_path, header=0, index_col=0)
        
        gene_list = list(mtx.columns)
        self.mtx = mtx.values
        self.mtx_adj = mtx_adj.values
        
        for path in path_list:
            self.data[path] = {}
            adata_temp = sc.read_h5ad(path)
            adata_temp = adata_temp[:, adata_temp.var_names.isin(gene_list)][:, gene_list]
            adata = adata_temp.copy()
            adata.obs["index"] = range(len(adata.obs))
            
            self.data[path]["adata"] = adata
            self.data[path]["perturb_rows"] = adata.obs.loc[
                (adata.obs[self.pert_col] != self.control_name)
            ]["index"].tolist()
            self.data[path]["control_rows"] = adata.obs.loc[
                (adata.obs[self.pert_col] == self.control_name)
            ]["index"].tolist()

    def get_file_names(self):
        return self.path_list

    def get_adata_shape(self, name):
        adata = self.data[name]["adata"]
        return adata.shape

    def get_item(self, name, row):
        adata = self.data[name]["adata"]
        if "counts" in adata.layers:
            expr = adata.layers["counts"][row]
        else:
            expr = adata.X[row]
        
        if scipy.sparse.issparse(expr):
            expr = expr.toarray().squeeze()
        else:
            expr = expr.squeeze()

        if "log1p" in adata.uns and adata.uns["log1p"]:
            expr = np.expm1(expr)
        return expr

    def get_control_item(self, name):
        control_row = self.get_control(name)
        expr = self.get_item(name, control_row)
        return expr

    def get_gene(self, name, cols):
        adata = self.data[name]["adata"]
        v = []
        for col in cols:
            v.append(adata.var.index[col])
        return v

    def get_perturb_rows(self, name):
        return self.data[name]["perturb_rows"]

    def get_control_rows(self, name):
        return self.data[name]["control_rows"]

    def get_control(self, name):
        control_rows = self.data[name]["control_rows"]
        return random.choice(control_rows)

    def get_perturbation(self, name, row):
        adata = self.data[name]["adata"]
        return adata.obs[self.pert_col].cat.codes.iloc[row]
    
    def get_mask(self, u_cols):
        return self.mtx[u_cols][:, u_cols]

    def get_adj_mask(self, u_cols):
        return self.mtx_adj[u_cols][:, u_cols]

class HCADataset(Dataset):
    def __init__(self, args, data_name, train=True):
        self.args = args
        self.train = train
        path_list = [os.path.join(args.dataset_dir, data_name)]
        self.reader = H5ADReader(args, path_list)
        self.file_names = self.reader.get_file_names()
        
        self.weights = []
        for name in self.file_names:
            self.weights.append(self.reader.get_adata_shape(name)[0])
        self.weights = np.array(self.weights)
        self.weights = self.weights / self.weights.sum()
        
        self.perturb_all = self.reader.data[self.file_names[0]]['adata'].obs[args.pert_col]
        
        self.gene2token = {}
        gene_name_path = args.gene_name_path
        for i, x in enumerate(open(gene_name_path)):
            self.gene2token[x.strip()] = i

        for k in range(len(self.file_names)):
            for i, x in enumerate(list(self.reader.data[self.file_names[k]]['adata'].var.index)):
                if x.strip() not in self.gene2token:
                    self.gene2token[x.strip()] = i

        self.perturb_rows = {}
        for k in self.reader.get_file_names():
            perturb_rows = self.reader.get_perturb_rows(k)
            self.perturb_rows[k] = list(perturb_rows)
            self.cell_name = self.reader.data[self.file_names[0]]['adata'].obs.index[self.perturb_rows[k]]

    def __len__(self):
        return sum([len(_) for k, _ in self.perturb_rows.items()])

    def downsample_v2(self, a, p):
        rnd = scipy.stats.binom(a, p)
        return rnd.rvs()

    def get_sample(self, file_name, row, aug_prob, ratio):
        perturb = self.reader.get_perturbation(file_name, row)
        control = self.reader.get_control(file_name)

        rawcount_control = self.reader.get_control_item(file_name)
        rawcount_perturb = self.reader.get_item(file_name, row)

        def check_sparse(rawcount):
            if scipy.sparse.issparse(rawcount):
                rawcount = rawcount.toarray().squeeze().astype(int)
            else:
                rawcount = rawcount.squeeze().astype(int)
            return rawcount

        rawcount_control = check_sparse(rawcount_control)
        rawcount_perturb = check_sparse(rawcount_perturb)

        if self.train:
            rawcount_perturb = self.downsample_v2(rawcount_perturb, aug_prob)
            rawcount_control = self.downsample_v2(rawcount_control, aug_prob)

        ds_perturb = rawcount_perturb
        ds_control = rawcount_control

        u_cols = np.array(np.arange(ds_perturb.shape[0])[ds_perturb > 0])

        if self.train and u_cols.shape[0] > 2048:
            u_cols = np.random.choice(u_cols, 2048, replace=False)

        def normalize(x):
            eps = 1e-12
            x = np.log1p(x / (x.sum(axis=0) + eps) * 10000)
            return x

        ds_pert_norm = normalize(ds_perturb)
        ds_ctrl_norm = normalize(ds_control)

        u_token = []
        for x in self.reader.get_gene(file_name, u_cols):
            u_token.append(self.gene2token[x])

        sub_dis_matrix = self.reader.get_mask(u_cols)
        sub_adj_matrix = self.reader.get_adj_mask(u_cols)
        
        return {
            "u_token": u_token,
            "ds": ds_perturb[u_cols],
            "ds_norm": ds_pert_norm[u_cols],
            "ds_control": ds_control[u_cols],
            "ds_control_norm": ds_ctrl_norm[u_cols],
            "perturb": np.array([perturb]),
            "control": np.array([control]),
            "dis_matrix": sub_dis_matrix,
            "adj_matrix": sub_adj_matrix,
            "index": u_cols,
        }

    def __getitem__(self, index):
        file_name = np.random.choice(self.file_names, p=self.weights) if self.train else self.file_names[0]
        aug_prob = np.random.random() * 0.9 + 0.1
        ratio = 5

        data = []
        mx_len = 0
        mx_token = 16 * 256

        num_samples = 1024 if self.train else 1
        for i in range(num_samples):
            row = None
            if self.train:
                row = np.random.choice(self.perturb_rows[file_name])
            else:
                row = self.perturb_rows[file_name][index]
            
            sample = self.get_sample(file_name, row, aug_prob, ratio)
            cur_len = len(sample["u_token"])
            if len(data) > 0 and max(mx_len, cur_len) * (len(data) + 1) > mx_token:
                break
            data.append(sample)
            mx_len = max(mx_len, cur_len)

        def _pad_fn(a, value):
            a = [np.array(_) for _ in a]
            max_shape = np.max([_.shape for _ in a], axis=0)
            na = []
            for x in a:
                pad_shape = [(0, l2 - l1) for l1, l2 in zip(x.shape, max_shape)]
                na.append(np.pad(x, pad_shape, mode="constant", constant_values=value))
            return np.stack(na)

        pad_values = {
            "u_token": 40000,
            "ds": 0,
            "ds_norm": 0.0,
            "ds_control": 0,
            "ds_control_norm": 0.0,
            "perturb": -1,
            "control": -1,
            "dis_matrix": 0,
            "adj_matrix": 0,
            "index": 0,
        }

        ret = {}
        ret["ratio"] = np.array([ratio])
        for key in data[0].keys():
            if key in pad_values:
                ret[key] = _pad_fn([_[key] for _ in data], pad_values[key])
        return ret


class TrainDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.dataset = HCADataset(args, args.train_data, train=True)
        self._cell_name = self.dataset.cell_name
        self._perturb_all = self.dataset.perturb_all
        
        adj_path = os.path.join(args.dataset_dir, args.adj_graph)
        mtx = pd.read_csv(adj_path, header=0, index_col=0)
        self._in_degree = mtx.sum(axis=0)
        self._out_degree = mtx.sum(axis=1)

    def __getitem__(self, index):
        data = self.dataset[index]
        return {
            "ratio": torch.tensor(data["ratio"]).long(),
            "u_token": torch.tensor(data["u_token"]).long(),
            "ds": torch.tensor(data["ds"]).long(),
            "ds_norm": torch.tensor(data["ds_norm"]).float(),
            "ds_control": torch.tensor(data["ds_control"]).long(),
            "ds_control_norm": torch.tensor(data["ds_control_norm"]).float(),
            "perturb": torch.tensor(data["perturb"]).long()[:, 0],
            "control": torch.tensor(data["control"]).long()[:, 0],
            "dis_matrix": torch.tensor(data["dis_matrix"]).float(),
            "adj_matrix": torch.tensor(data["adj_matrix"]).float(), 
            "in_degree": torch.tensor(np.squeeze(self._in_degree.values[data["index"]])).long(),
            "out_degree": torch.tensor(np.squeeze(self._out_degree.values[data["index"]])).long(),
        }
    
    def __len__(self):
        return len(self.dataset)


class ValidationDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.dataset = HCADataset(args, args.test_data, train=False)
        self._cell_name = self.dataset.cell_name
        self._perturb_all = self.dataset.perturb_all
        
        adj_path = os.path.join(args.dataset_dir, args.adj_graph)
        mtx = pd.read_csv(adj_path, header=0, index_col=0)
        self._in_degree = mtx.sum(axis=0)
        self._out_degree = mtx.sum(axis=1)

    def __getitem__(self, index):
        data = self.dataset[index]
        return {
            "ratio": torch.tensor(data["ratio"]).long(),
            "u_token": torch.tensor(data["u_token"]).long(),
            "ds": torch.tensor(data["ds"]).long(),
            "ds_norm": torch.tensor(data["ds_norm"]).float(),
            "ds_control": torch.tensor(data["ds_control"]).long(),
            "ds_control_norm": torch.tensor(data["ds_control_norm"]).float(),
            "perturb": torch.tensor(data["perturb"]).long()[:, 0],
            "control": torch.tensor(data["control"]).long()[:, 0],
            "dis_matrix": torch.tensor(data["dis_matrix"]).float(),
            "adj_matrix": torch.tensor(data["adj_matrix"]).float(),
            "in_degree": torch.tensor(np.squeeze(self._in_degree.values[data["index"]])).long(),
            "out_degree": torch.tensor(np.squeeze(self._out_degree.values[data["index"]])).long(),
        }

    def __len__(self):
        return len(self.dataset)