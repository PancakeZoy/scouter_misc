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
    return "+".join(sorted(x.split("+")))


metric_df_ls = []
split = 1
for split in [1, 2, 3, 4, 5]:
    # Set seeds for reproducibility
    set_seeds(24)
    data_path = "../../data/Data_GEARS/adamson/perturb_processed.h5ad"
    embd_path = "embeddings_ann.csv"

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
    embd = pd.read_csv(embd_path, index_col=0)
    ctrl_row = pd.DataFrame(
        [np.zeros(embd.shape[1])], columns=embd.columns, index=["ctrl"]
    )
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

    pertdata = ScouterData(adata, embd, "condition", "gene_name")
    pertdata.setup_ad("embd_index")
    pertdata.gene_ranks()
    pertdata.get_dropout_non_zero_genes()
    pertdata.split_Train_Val_Test(seed=split)

    scouter_model = Scouter(pertdata)
    scouter_model.model_init()
    scouter_model.train(loss_lambda=0.01, lr=0.001)
    metric_df = scouter_model.evaluate().assign(split=split)
    metric_df_ls.append(metric_df)

metric_df = pd.concat(metric_df_ls)
metric_df.to_csv("with_ann_metric.csv")