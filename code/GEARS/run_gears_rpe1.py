from gears import PertData, GEARS
import torch
import pickle
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Run biolord module on rpe1 data.')
parser.add_argument('split_seed', type=int, help='The seed to reproduce the result.')
args = parser.parse_args()

split_seed = args.split_seed

def main():
    pert_data = PertData('../../data/Data_GEARS')
    pert_data.load(data_name = 'replogle_rpe1_essential')
    pert_data.prepare_split(split = 'simulation', seed = split_seed)
    pert_data.get_dataloader(batch_size = 32, test_batch_size = 32)
    
    gears_model = GEARS(pert_data, 
                        device = "cuda" if torch.cuda.is_available() else "cpu",
                        weight_bias_track = False, 
                        proj_name = 'rpe1',
                        exp_name = "gears_seed" + str(split_seed))
    gears_model.model_initialize(hidden_size = 64)
    gears_model.train(epochs = 15)

    adata = gears_model.adata
    topgene_dict = adata.uns['top_non_dropout_de_20']
    gene2idx = gears_model.node_map
    gene_raw2id = dict(zip(adata.var.index.values, adata.var.gene_name.values))
    
    test_subgroup = gears_model.subgroup['test_subgroup']
    for subgroup, pertbs in test_subgroup.items():
        
        result_dic = {}
        if subgroup == 'unseen_single':
            test_genes = [[sorted(i.split('+'))[0]] for i in pertbs]
        else:
            test_genes = [i.split('+') for i in pertbs]

        test_result = {key: list(gears_model.predict([gene]).values())[0] for key, gene in zip(pertbs, test_genes)}
    
        for test_key in test_result.keys():
            topgene_key = 'rpe1_'+test_key+'_1+1'
            pred = test_result[test_key]
            truth = adata[adata.obs.condition == test_key].X.toarray().mean(axis=0)
            ctrl = adata[adata.obs.condition == 'ctrl'].X.toarray().mean(axis=0)
            de_idx = [gene2idx[gene_raw2id[i]] for i in topgene_dict[topgene_key]]
            result_dic[test_key] = {'Truth': truth,
                                     'Pred': pred,
                                     'Ctrl': ctrl,
                                     'DE_idx': de_idx}
    
        with open(f'../../results/GEARS/GEARS_rpe1_{subgroup}_{split_seed}.pkl', 'wb') as f:
            pickle.dump(result_dic, f)

if __name__ == '__main__':
    main()
