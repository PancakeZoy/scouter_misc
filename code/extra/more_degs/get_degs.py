import anndata as ad
from scouter import ScouterData
import pickle
import pandas as pd
import torch
import random
import numpy as np
from types import MethodType


# fmt: off
def get_dropout_non_zero_genes(self, top_n=20):        
    if 'rank_genes_groups' not in self.adata.uns.keys():
        raise ValueError("Gene expression (ad.AnnData) does not have 'rank_genes_groups' in .uns, please first run function gene_ranks()")
    
    ctrl = np.mean(self.adata[self.adata.obs[self.key_label] == 'ctrl'].X, axis = 0)
    ctrl = np.array(ctrl).squeeze()
    gene_id2idx = dict(zip(self.adata.var.index.values, range(len(self.adata.var))))

    non_zeros_gene_idx = {}
    non_dropout_gene_idx = {}
    top_non_dropout_de_n = {}
    top_non_zero_de_n = {}
    top_de_n = {}

    for pert in self.adata.uns['rank_genes_groups'].keys():
        X = np.mean(self.adata[self.adata.obs[self.key_label] == pert].X, axis = 0)
        X = np.array(X).squeeze()

        non_zero = np.where(X != 0)[0]
        zero = np.where(X == 0)[0]
        true_zeros = np.intersect1d(zero, np.where(ctrl == 0)[0])
        non_dropouts = np.concatenate((non_zero, true_zeros))

        rank_genes = self.adata.uns['rank_genes_groups'][pert]
        gene_idx_top = [gene_id2idx[g] for g in rank_genes]
        
        de_n = gene_idx_top[:top_n]
        non_dropout_n = [i for i in gene_idx_top if i in non_dropouts][:top_n]
        non_zero_n = [i for i in gene_idx_top if i in non_zero][:top_n]
        de_n = gene_idx_top[:top_n]
        
        non_zeros_gene_idx[pert] = non_zero
        non_dropout_gene_idx[pert] = non_dropouts
        top_non_dropout_de_n[pert] = np.array(non_dropout_n)
        top_non_zero_de_n[pert] = np.array(non_zero_n)
        top_de_n[pert] = np.array(de_n)

    self.adata.uns[f'top{top_n}_degs'] = top_de_n
    self.adata.uns[f'top{top_n}_degs_non_zero'] = top_non_zero_de_n
    self.adata.uns[f'top{top_n}_degs_non_dropout'] = top_non_dropout_de_n
    self.adata.uns['gene_idx_non_dropout'] = non_dropout_gene_idx
    self.adata.uns['gene_idx_non_zeros'] = non_zeros_gene_idx

def set_seeds(seed=24):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
# Normalize the condition name. Make "A+B" and "B+A" the same
def condition_sort(x):
    return '+'.join(sorted(x.split('+')))

embd_path_list = {'GenePT_v1': '../../data/Data_GeneEmbd/GenePT_V1.pickle',
                  'GenePT_v2': '../../data/Data_GeneEmbd/GenePT_V2.pickle',
                  'scELMo': '../../data/Data_GeneEmbd/scELMo.pickle'}

# Set seeds for reproducibility
set_seeds(24)
data_path = '../../data/Data_GEARS/adamson/perturb_processed.h5ad'
embd_path = embd_path_list['GenePT_v2']

# Load the processed scRNA-seq dataset as Anndata
adata = ad.read_h5ad(data_path)
adata.obs['condition'] = adata.obs['condition'].astype(str).apply(lambda x: condition_sort(x)).astype('category')
adata.uns = {}; adata.obs.drop('condition_name', axis=1, inplace=True)

# Load the gene embedding as the dataframe, and rename its gene alias to match the Anndata
with open(embd_path, 'rb') as f:
    embd = pd.DataFrame(pickle.load(f)).T
ctrl_row = pd.DataFrame([np.zeros(embd.shape[1])], columns=embd.columns, index=['ctrl'])
embd = pd.concat([ctrl_row, embd])
embd.rename(index={
    'SARS1': 'SARS',
        'DARS1': 'DARS',
        'QARS1': 'QARS',
        'TARS1': 'TARS',
        'HARS1': 'HARS',
        'CARS1': 'CARS',
        'SRPRA': 'SRPR',
        'MARS1': 'MARS',
        'AARS1': 'AARS',
        'PRELID3B': 'SLMO2'}, inplace=True)

pertdata = ScouterData(adata, embd, 'condition', 'gene_name')
pertdata.setup_ad('embd_index')
pertdata.gene_ranks()
pertdata.get_dropout_non_zero_genes = MethodType(get_dropout_non_zero_genes, pertdata)
degs_ls = {}
for top_n in [20, 30, 40, 50, 60, 70, 80, 90, 100]:
    pertdata.get_dropout_non_zero_genes(top_n=top_n)
    degs_ls[top_n] = pertdata.adata.uns[f'top{top_n}_degs_non_dropout']

with open('adamson_degs.pkl', 'wb') as f:
    pickle.dump(degs_ls, f)
