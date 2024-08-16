from gears import PertData, GEARS
import torch
import pickle
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Run biolord module on adamson data.')
parser.add_argument('split_seed', type=int, help='The seed to reproduce the result.')
args = parser.parse_args()

split_seed = args.split_seed

def main():
    pert_data = PertData('../../data/Data_GEARS')
    pert_data.load(data_name = 'adamson')
    pert_data.prepare_split(split = 'simulation', seed = split_seed)
    pert_data.get_dataloader(batch_size = 32, test_batch_size = 32)
    
    gears_model = GEARS(pert_data, 
                        device = "cuda" if torch.cuda.is_available() else "cpu",
                        weight_bias_track = False, 
                        proj_name = 'adamson',
                        exp_name = "gears_seed" + str(split_seed))
    gears_model.model_initialize(hidden_size = 64)
    gears_model.train(epochs = 15)
    
    test_genes = [[i.split('+')[0]] for i in gears_model.set2conditions['test']]
    test_result = gears_model.predict(test_genes)
    
    adata = gears_model.adata
    topgene_dict = adata.uns['top_non_dropout_de_20']
    gene2idx = gears_model.node_map
    gene_raw2id = dict(zip(adata.var.index.values, adata.var.gene_name.values))
    
    result_dic = {}
    topgene_dict = adata.uns['top_non_dropout_de_20']
    for gene in test_result.keys():
        condition=gene+'+ctrl'
        topgene_key = 'K562(?)_'+gene+'+ctrl_1+1'
        pred = test_result[gene]
        truth = adata[adata.obs.condition == condition].X.toarray().mean(axis=0)
        ctrl = adata[adata.obs.condition == 'ctrl'].X.toarray().mean(axis=0)
        de_idx = [gene2idx[gene_raw2id[i]] for i in topgene_dict[topgene_key]]
        result_dic[gene] = {'Truth': truth,
                            'Pred': pred,
                            'Ctrl': ctrl,
                            'DE_idx': de_idx}
    
    with open(f'../../results/GEARS/GEARS_adamson_{split_seed}.pkl', 'wb') as f:
        pickle.dump(result_dic, f)

if __name__ == '__main__':
    main()