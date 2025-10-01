import pickle
import torch
import anndata as ad
import pandas as pd
import numpy as np
import random
from scouter import Scouter, ScouterData


# fmt: off
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

def main():
    for split in [1,2,3,4,5]:
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
        pertdata.get_dropout_non_zero_genes()
        pertdata.split_Train_Val_Test(seed=split)
        
        scouter_model = Scouter(pertdata)
        scouter_model.model_init()
        scouter_model.train(loss_lambda=0.01, lr=0.001)
        
        test_conds = list(scouter_model.test_adata.obs[scouter_model.key_label].unique())
        test_conds.remove('ctrl')
        test_result = {}
        for condition in test_conds:
            pred = scouter_model.pred([condition])[condition]
            test_result[condition] = pred

        with open(f'Scouter_adamson_{split}.pkl', 'wb') as f:
            pickle.dump(test_result, f)

if __name__ == '__main__':
    main()
