import pickle
import torch
import anndata as ad
import pandas as pd
import numpy as np
import random
import argparse
from scouter import Scouter, ScouterData

parser = argparse.ArgumentParser()
parser.add_argument("seed", type=int, default=24)
args = parser.parse_args()
split = args.seed


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
data_path = "DataScouter/perturb_processed.h5ad"
embd_path = "DataScouter/GenePT_V1.pickle"
split_path = f"splits_scouter/split_dict_{split}.pkl"

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
        "MAP3K21": "KIAA1804",
        "FOXL2NB": "C3orf72",
        "RHOXF2B": "RHOXF2BB",
        "MIDEAS": "ELMSAN1",
        "CBARP": "C19orf26",
    },
    inplace=True,
)

# Load the split dictionary
split_dict = pickle.load(open(split_path, "rb"))

pertdata = ScouterData(adata, embd, "condition", "gene_name")
pertdata.setup_ad("embd_index")
pertdata.gene_ranks()
pertdata.get_dropout_non_zero_genes()
pertdata.split_Train_Val_Test(val_conds=split_dict["val"], test_conds=split_dict["test"])  # fmt: skip
print(f"If validation sets match: {pertdata.val_conds[:-1] == split_dict['val']}")
print(f"If test sets match: {pertdata.test_conds[:-1] == split_dict['test']}")
print(f"If train sets match: {pertdata.train_conds[:-1] == split_dict['train']}")


scouter_model = Scouter(pertdata)
scouter_model.model_init()
scouter_model.train(loss_lambda=0.05, lr=0.001)

# Save the model's state dictionary
save_path = f"Scouter_model/scouter_{split}.pth"
torch.save(scouter_model.network.state_dict(), save_path)