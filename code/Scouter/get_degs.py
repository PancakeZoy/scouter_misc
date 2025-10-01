import pickle
import torch
import anndata as ad
import pandas as pd
import numpy as np
import random
from scouter import ScouterData
import os

# fmt: off
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def set_seeds(seed=24):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Normalize the condition name. Make "A+B" and "B+A" the same
def condition_sort(x):
    return "+".join(sorted(x.split("+")))


def get_degs(dataset):
    set_seeds(24)
    data_path = f"../../data/Data_GEARS/{dataset}/perturb_processed.h5ad"
    embd_path = "../../data/Data_GeneEmbd/GenePT_V1.pickle"

    # Load the processed scRNA-seq dataset as Anndata
    adata = ad.read_h5ad(data_path)
    adata.obs["condition"] = (
        adata.obs["condition"]
        .astype(str)
        .apply(lambda x: condition_sort(x))
        .astype("category")
    )
    adata.uns = {}
    adata.obs.drop("condition_name", axis=1, inplace=True)

    # Load the gene embedding as the dataframe, and rename its gene alias to match the Anndata
    with open(embd_path, "rb") as f:
        embd = pd.DataFrame(pickle.load(f)).T
    ctrl_row = pd.DataFrame([np.zeros(embd.shape[1])], columns=embd.columns, index=["ctrl"])
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
        'PRELID3B': 'SLMO2',
        'MAP3K21': 'KIAA1804',
        'FOXL2NB': 'C3orf72',
        'RHOXF2B': 'RHOXF2BB',
        'MIDEAS': 'ELMSAN1',
        'CBARP': 'C19orf26',
        'CENATAC': 'CCDC84',
        'POLR1G': 'CD3EAP',
        'EPRS1': 'EPRS',
        'IARS1': 'IARS',
        'KARS1': 'KARS',
        'LARS1': 'LARS',
        'RARS1': 'RARS',
        'POLR1F': 'TWISTNB',
        'VARS1': 'VARS',
        'POLR1H': 'ZNRD1',
        'ZZZ3': 'AC118549.1'}, inplace=True)

    pertdata = ScouterData(adata, embd, "condition", "gene_name")
    pertdata.setup_ad("embd_index")
    pertdata.gene_ranks()
    pertdata.get_dropout_non_zero_genes()

    degs_dict = pertdata.adata.uns['top20_degs_non_dropout']
    for condition, degs in degs_dict.items():
        genes = condition.split('+')
        genes = sorted(list(np.setdiff1d(genes, ['ctrl'])))
        genes_id = np.where(pertdata.adata.var.gene_name.isin(genes))[0]
        degs_dict[condition] = sorted(np.setdiff1d(degs, genes_id))
    return degs_dict


degs_dixit = get_degs('dixit')
degs_adamson = get_degs('adamson')
degs_norman = get_degs('norman')
degs_k562 = get_degs('replogle_k562_essential')
degs_rpe1 = get_degs('replogle_rpe1_essential')

degs_all = {
    'dixit': degs_dixit,
    'adamson': degs_adamson,
    'norman': degs_norman,
    'k562': degs_k562,
    'rpe1': degs_rpe1
} 

with open('../../results/degs_all.pkl', 'wb') as f:
    pickle.dump(degs_all, f)
