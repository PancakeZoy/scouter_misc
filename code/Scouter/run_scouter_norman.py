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

def split_list_by_ratio(lst, ratios, seed):
    random.seed(seed)
    random.shuffle(lst)
    
    total_ratio = sum(ratios)
    total_length = len(lst)
    num_splits = len(ratios)
    
    if total_length < num_splits:
        raise ValueError("The length of the list must be greater than or equal to the number of ratios.")
    
    sizes = [max(1, int(total_length * ratio / total_ratio)) for ratio in ratios]
    
    while sum(sizes) > total_length:
        sizes[sizes.index(max(sizes))] -= 1
    
    sizes[-1] += total_length - sum(sizes)
    
    splits = []
    current_index = 0
    for size in sizes:
        splits.append(lst[current_index:current_index + size])
        current_index += size
    
    return splits


def subgroup(pert_list, seed):
    uniq_perts = pert_list.copy()
    uniq_perts.remove('ctrl')
    uniq_combos = [p for p in uniq_perts if 'ctrl' not in p.split('+')]
    uniq_singles = [p for p in uniq_perts if 'ctrl' in p.split('+')]
    
    test_single, val_single, train_single = split_list_by_ratio(uniq_singles, [0.2, 0.05, 0.75], seed = seed)
    
    combo_seen0 = [p for p in uniq_combos if sum([i+'+ctrl' in train_single for i in p.split('+')])==0]
    combo_seen1 = [p for p in uniq_combos if sum([i+'+ctrl' in train_single for i in p.split('+')])==1]
    combo_seen2 = [p for p in uniq_combos if sum([i+'+ctrl' in train_single for i in p.split('+')])==2]
    
    test_seen0, val_seen0, train_seen0 = split_list_by_ratio(combo_seen0, [0.2, 0.1, 0.7], seed = seed)
    test_seen1, val_seen1, train_seen1 = split_list_by_ratio(combo_seen1, [0.2, 0.1, 0.7], seed = seed)
    test_seen2, val_seen2, train_seen2 = split_list_by_ratio(combo_seen2, [0.2, 0.1, 0.7], seed = seed)
    
    test_all = test_single + test_seen0 + test_seen1 + test_seen2
    val_all = val_single + val_seen0 + val_seen1 + val_seen2
    train_all = train_single + train_seen0 + train_seen1 + train_seen2
    
    group = ['train' if p in train_all else 'val' if p in val_all else 'test' for p in uniq_perts]
    subgroup = []
    for p in uniq_perts:
        if p in uniq_singles:
            subgroup.append('single')
        elif p in combo_seen0:
            subgroup.append('seen0')
        elif p in combo_seen1:
            subgroup.append('seen1')
        elif p in combo_seen2:
            subgroup.append('seen2')
    
    df = pd.DataFrame({'group': group, 'subgroup': subgroup}, index=uniq_perts)
    return df

embd_path_list = {'GenePT_v1': '../../data/Data_GeneEmbd/GenePT_V1.pickle',
                  'GenePT_v2': '../../data/Data_GeneEmbd/GenePT_V2.pickle',
                  'scELMo': '../../data/Data_GeneEmbd/scELMo.pickle'}

def main():
    metric_df_ls = []
    for split in [1,2,3,4,5]:
        # Set seeds for reproducibility
        set_seeds(24)
        data_path = '../../data/Data_GEARS/norman/perturb_processed.h5ad'
        embd_path = embd_path_list['scELMo']
        
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
            'MAP3K21': 'KIAA1804',
            'FOXL2NB': 'C3orf72',
            'RHOXF2B': 'RHOXF2BB',
            'MIDEAS': 'ELMSAN1',
            'CBARP': 'C19orf26'}, inplace=True)
        
        
        splt_df = subgroup(list(adata.obs.condition.unique()), seed=split)
        test_conds = list(splt_df[splt_df.group=='test'].index)
        val_conds = list(splt_df[splt_df.group=='val'].index)
        
        pertdata = ScouterData(adata, embd, 'condition', 'gene_name')
        pertdata.setup_ad('embd_index')
        pertdata.gene_ranks()
        pertdata.get_dropout_non_zero_genes()
        pertdata.split_Train_Val_Test(val_conds=val_conds, test_conds=test_conds, seed=split)
        
        scouter_model = Scouter(pertdata)
        scouter_model.model_init()
        scouter_model.train(loss_lambda=0.05, lr=0.001)
        metric_df = scouter_model.evaluate().assign(split=split)
        test_df = splt_df[splt_df.group=='test'].copy()
        metric_df.loc[test_df.index, 'subgroup'] = test_df.subgroup.values
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

        with open(f'../../results/Scouter/Scouter_norman_{split}.pkl', 'wb') as f:
            pickle.dump(test_result, f)
    
    metric_df = pd.concat(metric_df_ls)
    metric_df.to_csv('../../results/Scouter/Scouter_norman_result.csv')
    
if __name__ == '__main__':
    main()
