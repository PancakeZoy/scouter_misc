import pickle
import torch
import anndata as ad
import pandas as pd
import numpy as np
import random
import argparse
from scouter import Scouter, ScouterData

# fmt: off
# Set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument("proportion", type=int)
args = parser.parse_args()


def set_seeds(seed=24):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Normalize the condition name. Make "A+B" and "B+A" the same
def condition_sort(x):
    return "+".join(sorted(x.split("+")))

ad_path = {"dixit": "../../../data/Data_GEARS/dixit/perturb_processed.h5ad",
           "adamson": "../../../data/Data_GEARS/adamson/perturb_processed.h5ad",
           "k562": "../../../data/Data_GEARS/replogle_k562_essential/perturb_processed.h5ad",
           "rpe1": "../../../data/Data_GEARS/replogle_rpe1_essential/perturb_processed.h5ad"}

def main():
    proportion = args.proportion*10
    dataset = "k562"
    metric_df_ls = []
    for split in [1, 2, 3, 4, 5]:
        # Set seeds for reproducibility
        set_seeds(24)
        data_path = ad_path[dataset]
        embd_path = f"truncated_embeddings/embeddings_Proportion{proportion}.pkl"

        # Load the processed scRNA-seq dataset as Anndata
        adata = ad.read_h5ad(data_path)
        adata.obs['condition'] = adata.obs['condition'].astype(str).apply(lambda x: condition_sort(x)).astype('category')  # fmt: off
        adata.uns = {}
        adata.obs.drop("condition_name", axis=1, inplace=True)

        # Load the gene embedding as the dataframe, and rename its gene alias to match the Anndata
        with open(embd_path, "rb") as f:
            embd_pkl = pickle.load(f)
            embd = pd.DataFrame(embd_pkl).T
        ctrl_row = pd.DataFrame([np.zeros(embd.shape[1])], columns=embd.columns, index=["ctrl"])
        embd = pd.concat([ctrl_row, embd])

        pertdata = ScouterData(adata, embd, "condition", "gene_name")
        pertdata.setup_ad("embd_index")
        pertdata.gene_ranks()
        pertdata.get_dropout_non_zero_genes()
        pertdata.split_Train_Val_Test(seed=split)

        scouter_model = Scouter(pertdata)
        scouter_model.model_init()
        scouter_model.train(loss_lambda=0.5, lr=0.001)
        metric_df = scouter_model.evaluate().assign(split=split)
        metric_df_ls.append(metric_df)

        test_conds = list(scouter_model.test_adata.obs[scouter_model.key_label].unique())  # fmt: off
        test_conds.remove("ctrl")
        test_result = {}
        # fmt: off
        for condition in test_conds:
            degs = scouter_model.all_adata.uns["top20_degs_non_dropout"][condition]
            degs = np.setdiff1d(degs, np.where(np.isin(scouter_model.all_adata.var[scouter_model.key_var_genename].values, condition.split('+'))))

            pred = scouter_model.pred([condition])[condition][:, degs]
            ctrl = scouter_model.ctrl_adata.X.toarray()[:, degs]
            true = scouter_model.all_adata[scouter_model.all_adata.obs[scouter_model.key_label]==condition].X.toarray()[:, degs]
            degs_name = np.array(scouter_model.all_adata.var.iloc[degs].gene_name.values)

            test_result[condition] = {
                "Pred": pred,
                "Ctrl": ctrl,
                "Truth": true,
                "DE_idx": degs,
                "DE_name": degs_name,
            }

        with open(f"results/{dataset}/Proportion_{proportion}_Split_{split}.pkl", "wb") as f:
            pickle.dump(test_result, f)

    metric_df = pd.concat(metric_df_ls)
    metric_df.to_csv(f"results/{dataset}/Proportion_{proportion}.csv")


if __name__ == "__main__":
    main()
