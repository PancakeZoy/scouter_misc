from gears import PertData, GEARS
import pytorch_lightning
import pickle
import numpy as np
import torch
import sys
import os

np.random.seed(202310)
pytorch_lightning.seed_everything(202310)

################################ Inputs ################################
ds_name = "norman"
seed_idx = int(sys.argv[1])
input_split_seed = [1, 2, 3, 4, 5][seed_idx]
input_n_epochs = 15

gpu_id = int(sys.argv[2])

log_dir = f"logs"
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, f"{ds_name}_split{input_split_seed}_gpu{gpu_id}.log")

class Unbuffered:
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
    def flush(self):
        self.stream.flush()
sys.stdout = Unbuffered(open(log_path, "w"))
sys.stderr = sys.stdout
########################################################################

# Read adata
pert_data = PertData('data')
pert_data.load(data_name = ds_name)
pert_data.prepare_split(split = 'simulation', seed = input_split_seed)
pert_data.get_dataloader(batch_size = 32, test_batch_size = 128)

# Read embeddings
with open("scELMo.pickle", "rb") as fp:
    sc_embedding = pickle.load(fp)
gene_names= list(pert_data.adata.var['gene_name'].values)
count_missing = 0
EMBED_DIM = 1536 # embedding dim from GPT-3.5
lookup_embed = np.zeros(shape=(len(gene_names),EMBED_DIM))
for i, gene in enumerate(gene_names):
    if gene in sc_embedding:
        lookup_embed[i,:] = sc_embedding[gene].flatten()
    else:
        count_missing+=1

# Train model
gears_model = GEARS(pert_data, 
                    # device = "cuda" if torch.cuda.is_available() else "cpu",
                    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu",
                    gene_emb = lookup_embed)
gears_model.model_initialize(hidden_size = 64)
gears_model.train(epochs = input_n_epochs)

# Make Prediction & Save
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
    # test_genes = np.unique(test_genes, axis=0)
    test_result = {key: list(gears_model.predict([gene]).values())[0] for key, gene in zip(pertbs, test_genes)}

    for test_key in test_result.keys():
        topgene_key = 'A549_'+test_key+'_1+1'
        pred = test_result[test_key]
        truth = adata[adata.obs.condition == test_key].X.toarray().mean(axis=0)
        ctrl = adata[adata.obs.condition == 'ctrl'].X.toarray().mean(axis=0)
        de_idx = [gene2idx[gene_raw2id[i]] for i in topgene_dict[topgene_key]]
        result_dic[test_key] = {'Truth': truth,
                                 'Pred': pred,
                                 'Ctrl': ctrl,
                                 'DE_idx': de_idx}

    with open(f'results/{ds_name}/qrsh_{subgroup}_split{input_split_seed}.pkl', 'wb') as f:
        pickle.dump(result_dic, f)