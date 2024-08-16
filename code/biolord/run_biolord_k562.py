import argparse
import scanpy as sc
import numpy as np
import pandas as pd
import torch
import biolord
import pickle

# Set up argument parser
parser = argparse.ArgumentParser(description='Run biolord module on k562 data.')
parser.add_argument('split_seed', type=int, help='The seed to reproduce the result.')
args = parser.parse_args()

def main():
    
    def bool2idx(x):
        """
        Returns the indices of the True-valued entries in a boolean array `x`
        """
        return np.where(x)[0]
    
    def repeat_n(x, n):
        """combo_seen2
        Returns an n-times repeated version of the Tensor x,
        repetition dimension is axis 0
        """
        # copy tensor to device BEFORE replicating it n times
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return x.to(device).view(1, -1).repeat(n, 1)
    
    
    adata = sc.read('../../data/Data_biolord/replogle_k562_essential/k562_biolord.h5ad')
    adata_single = sc.read('../../data/Data_biolord/replogle_k562_essential/k562_single_biolord.h5ad')
    

    varying_arg = {
        "seed": 42,
        "use_batch_norm": False,
        "use_layer_norm": False, 
        "step_size_lr": 45, 
        "attribute_dropout_rate": 0.1, 
        "cosine_scheduler": True,
        "scheduler_final_lr": 1e-5,
        "n_latent": 32, 
        "n_latent_attribute_ordered": 512,
        "reconstruction_penalty": 1000.0,
        "attribute_nn_width": 64,
        "attribute_nn_depth":6, 
        "attribute_nn_lr": 0.001, 
        "attribute_nn_wd": 4e-8,
        "latent_lr": 0.0001,
        "latent_wd": 0.001,
        "unknown_attribute_penalty": 10000.0,
        "decoder_width": 64,
        "decoder_depth": 1,  
        "decoder_activation": False,
        "attribute_nn_activation": False,
        "unknown_attributes": False,
        "decoder_lr": 0.001,
        "decoder_wd": 0.01,
        "max_epochs": np.min([round((20000 / adata.n_obs) * 400), 400]),
        "early_stopping_patience": 10,
        "ordered_attributes_key": "perturbation_neighbors",
        "n_latent_attribute_categorical": 16,
    }
    
    name_map = dict(adata.obs[["condition", "condition_name"]].drop_duplicates().values)
    ctrl = np.asarray(adata[adata.obs["condition"].isin(["ctrl"])].X.mean(0)).flatten()
    
    df_perts_expression = pd.DataFrame(adata.X.toarray(), index=adata.obs_names, columns=adata.var_names)
    df_perts_expression["condition"] = adata.obs["condition"]
    df_perts_expression = df_perts_expression.groupby(["condition"]).mean()
    df_perts_expression = df_perts_expression.reset_index()
    
    single_perts_condition = []
    single_pert_val = []
    for pert in adata.obs["condition"].cat.categories:
        if len(pert.split("+")) == 1:
            continue
        elif "ctrl" in pert:
            single_perts_condition.append(pert)
            p1, p2 = pert.split("+")
            if p2 == "ctrl":
                single_pert_val.append(p1)
            else:
                single_pert_val.append(p2)
                
    single_perts_condition.append("ctrl")
    single_pert_val.append("ctrl")
    
    df_singleperts_condition = pd.Index(single_perts_condition)
    
    np.random.seed(42)
    
    module_params = {
        "attribute_nn_width":  varying_arg["attribute_nn_width"],
        "attribute_nn_depth": varying_arg["attribute_nn_depth"],
        "use_batch_norm": varying_arg["use_batch_norm"],
        "use_layer_norm": varying_arg["use_layer_norm"],
        "attribute_dropout_rate":  varying_arg["attribute_dropout_rate"],
        # "unknown_attribute_noise_param": varying_arg["unknown_attribute_noise_param"],
        "seed": varying_arg["seed"],
        "n_latent_attribute_ordered": varying_arg["n_latent_attribute_ordered"],
        "n_latent_attribute_categorical": varying_arg["n_latent_attribute_categorical"],
        "reconstruction_penalty": varying_arg["reconstruction_penalty"],
        "unknown_attribute_penalty": varying_arg["unknown_attribute_penalty"],
        "decoder_width": varying_arg["decoder_width"],
        "decoder_depth": varying_arg["decoder_depth"],
        "decoder_activation": varying_arg["decoder_activation"],
        "attribute_nn_activation": varying_arg["attribute_nn_activation"],
        "unknown_attributes": varying_arg["unknown_attributes"],
    }


    trainer_params = {
        "n_epochs_warmup": 0,
        "latent_lr": varying_arg["latent_lr"],
        "latent_wd": varying_arg["latent_wd"],
        "attribute_nn_lr": varying_arg["attribute_nn_lr"],
        "attribute_nn_wd": varying_arg["attribute_nn_wd"],
        "step_size_lr": varying_arg["step_size_lr"],
        "cosine_scheduler": varying_arg["cosine_scheduler"],
        "scheduler_final_lr": varying_arg["scheduler_final_lr"],
        "decoder_lr": varying_arg["decoder_lr"],
        "decoder_wd": varying_arg["decoder_wd"]
    }
    
    
    test_metrics_biolord_delta = {}
    test_metrics_biolord_delta_normalized = {}
    
    ordered_attributes_key = varying_arg["ordered_attributes_key"]
    
    biolord.Biolord.setup_anndata(
        adata_single,
        ordered_attributes_keys=[ordered_attributes_key],
        categorical_attributes_keys=None,
        retrieval_attribute_key=None,
    )
    
    split_seed = args.split_seed
    test_metrics_biolord_delta[split_seed] = {}
    test_metrics_biolord_delta_normalized[split_seed] = {}
    
    train_idx = df_singleperts_condition.isin(adata[adata.obs[f"split{split_seed}"] == "train"].obs["condition"].cat.categories)
    train_condition_perts = df_singleperts_condition[train_idx]
    
    model = biolord.Biolord(
        adata=adata_single,
        n_latent=varying_arg["n_latent"],
        model_name="norman",
        module_params=module_params,
        train_classifiers=False,
        split_key=f"split{split_seed}"
    )

    model.train(
        max_epochs=int(varying_arg["max_epochs"]),
        batch_size=32,
        plan_kwargs=trainer_params,
        early_stopping=True,
        early_stopping_patience=int(varying_arg["early_stopping_patience"]),
        check_val_every_n_epoch=5,
        num_workers=1,
        enable_checkpointing=False
    )
    
    adata_control = adata_single[adata_single.obs["condition"] == "ctrl"].copy()
    dataset_control = model.get_dataset(adata_control)
    
    dataset_reference = model.get_dataset(adata_single)
    
    n_obs = adata_control.shape[0]
    ood_set = "unseen_single"
    result_dic = {}
    perts = adata[adata.obs[f"subgroup{split_seed}"] == ood_set].obs["condition"].cat.categories
    
    for i, pert in enumerate(perts):
        bool_de = adata.var_names.isin(
                    np.array(adata.uns["top_non_zero_de_20"][name_map[pert]])
                )
        idx_de = bool2idx(bool_de)
        if pert in train_condition_perts:
            idx_ref =  bool2idx(adata_single.obs["condition"] == pert)[0]
            expression_pert = dataset_reference["X"][[idx_ref], :].mean(0).cpu().numpy()
            test_preds_delta = expression_pert
        
        elif "ctrl" in pert:
            idx_ref =  bool2idx(adata_single.obs["condition"] == pert)[0]
            expression_pert = dataset_reference["X"][[idx_ref], :].mean(0).cpu().numpy()
    
            dataset_pred = dataset_control.copy()
            dataset_pred[ordered_attributes_key] = repeat_n(dataset_reference[ordered_attributes_key][idx_ref, :], n_obs)
            test_preds, _ = model.module.get_expression(dataset_pred)
    
            test_preds_delta = test_preds.cpu().numpy()
    
        result_dic[name_map[pert]] = {'Truth': expression_pert.flatten(),
                                      'Pred': test_preds_delta.flatten(),
                                      'Ctrl': ctrl,
                                      'DE_idx': idx_de}
    
    with open(f'../../results/biolord/biolord_k562_{split_seed}.pkl', 'wb') as f:
        pickle.dump(result_dic, f)
        
    epoch_hist = pd.DataFrame().from_dict(model.training_plan.epoch_history)
    epoch_hist.to_csv(f'../../results/biolord/biolord_k562_epoch_hist_{split_seed}.csv')

if __name__ == '__main__':
    main()