import pickle
import torch
import anndata as ad
import pandas as pd
import numpy as np
import random
from scouter import Scouter, ScouterData

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
    metric_df_ls = []
    for split in [1,2,3,4,5,6,7,8,9,10]:
        # Set seeds for reproducibility
        set_seeds(24)
        data_path = '../../data/Data_GEARS/dixit/perturb_processed.h5ad'
        embd_path = embd_path_list['GenePT_v1']
        
        # Load the processed scRNA-seq dataset as Anndata
        adata = ad.read_h5ad(data_path)
        adata.obs['condition'] = adata.obs['condition'].astype(str).apply(lambda x: condition_sort(x)).astype('category')
        adata.uns = {}; adata.obs.drop('condition_name', axis=1, inplace=True)
        
        # Load the gene embedding as the dataframe, and rename its gene alias to match the Anndata
        with open(embd_path, 'rb') as f:
            embd = pd.DataFrame(pickle.load(f)).T
        ctrl_row = pd.DataFrame([np.zeros(embd.shape[1])], columns=embd.columns, index=['ctrl'])
        embd = pd.concat([ctrl_row, embd])
        
        pertdata = ScouterData(adata, embd, 'condition', 'gene_name')
        pertdata.setup_ad('embd_index')
        pertdata.gene_ranks()
        pertdata.get_dropout_non_zero_genes()
        pertdata.split_Train_Val_Test(test_ratio=0.1, seed=split)
        
        scouter_model = Scouter(pertdata)
        scouter_model.model_init()
        scouter_model.train(loss_lambda=0.05, lr=0.01)
        metric_df = scouter_model.evaluate().assign(split=split)
        metric_df_ls.append(metric_df)
        
        test_conds = list(scouter_model.test_adata.obs[scouter_model.key_label].unique())
        test_conds.remove('ctrl')
        test_result = {}
        for condition in test_conds:
            degs = scouter_model.all_adata.uns['top20_degs_non_dropout'][condition]
            degs = np.setdiff1d(degs, np.where(np.isin(scouter_model.all_adata.var[scouter_model.key_var_genename].values, condition.split('+'))))

            pred = scouter_model.pred([condition])[condition][:, degs]
            ctrl = scouter_model.ctrl_adata.X.toarray()[:, degs]
            true = scouter_model.all_adata[scouter_model.all_adata.obs[scouter_model.key_label]==condition].X.toarray()[:, degs]
            degs_name = np.array(scouter_model.all_adata.var.iloc[degs].gene_name.values)

            test_result[condition] = {'Pred': pred,
                                      'Ctrl': ctrl,
                                      'Truth': true, 
                                      'DE_idx': degs,
                                      'DE_name': degs_name}

        with open(f'../../results/Scouter/Scouter_dixit_{split}.pkl', 'wb') as f:
            pickle.dump(test_result, f)
        
    metric_df = pd.concat(metric_df_ls)
    metric_df.to_csv('../../results/Scouter/Scouter_dixit_result.csv')
    
if __name__ == '__main__':
    main()