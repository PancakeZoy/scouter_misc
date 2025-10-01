import pickle
import torch
import anndata as ad
import pandas as pd
import numpy as np
import random
from scouter import ScouterData
from custom_scouter import CustomScouter


# fmt: off
def set_seeds(seed=24):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Normalize the condition name. Make "A+B" and "B+A" the same
def condition_sort(x):
    return "+".join(sorted(x.split("+")))


# Set seeds for reproducibility
set_seeds(24)
data_path = "../../data/Data_GEARS/adamson/perturb_processed.h5ad"
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
embd.rename(
    index={
        "SARS1": "SARS",
        "DARS1": "DARS",
        "QARS1": "QARS",
        "TARS1": "TARS",
        "HARS1": "HARS",
        "CARS1": "CARS",
        "SRPRA": "SRPR",
        "MARS1": "MARS",
        "AARS1": "AARS",
        "PRELID3B": "SLMO2",
    },
    inplace=True,
)

split = 5
pertdata = ScouterData(adata, embd, "condition", "gene_name")
pertdata.setup_ad("embd_index")
pertdata.gene_ranks()
pertdata.get_dropout_non_zero_genes()
pertdata.split_Train_Val_Test(seed=split)

scouter_model = CustomScouter(pertdata)
scouter_model.model_init()
scouter_model.train(loss_lambda=0.01, lr=0.001)

pred_100_10 = scouter_model.pred(pert_list=["ATP5B+ctrl"], n_pred=100, seed=10)["ATP5B+ctrl"]
pred_1000_1 = scouter_model.pred(pert_list=["ATP5B+ctrl"], n_pred=1000, seed=1)["ATP5B+ctrl"]
ctrl = adata[adata.obs["condition"] == "ctrl"].X.toarray().mean(axis=0)
true = adata[adata.obs["condition"] == "ATP5B+ctrl"].X.toarray()
degs_idx = pertdata.adata.uns["top20_degs_non_dropout"]["ATP5B+ctrl"]
degs_name = pertdata.adata.var["gene_name"].values[degs_idx]

result = {
    "condition": "ATP5B+ctrl",
    "degs_idx": degs_idx,
    "degs_name": degs_name,
    "ctrl": ctrl,
    "pred_100_10": pred_100_10,
    "pred_1000_1": pred_1000_1,
    "true": true,
}

with open("example_result.pickle", "wb") as f:
    pickle.dump(result, f)
