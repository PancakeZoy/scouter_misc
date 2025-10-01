## Evaluation of prediction on gene interaction.

| File                        | Function                                                                 | Output                          |
|-----------------------------|--------------------------------------------------------------------------|---------------------------------|
| `save_gears.py`             | Save GEARS model over different splits                                   | `GEARS_model/`                  |
| `save_scouter.py`           | Save Scouter model over different splits                                 | `Scouter_model/`                |
| `GI_truth_Curation.py`      | Generate the ground truth gene perturbation type <br>(adopted from Fig. 4f in the Norman paper) | `GI_truth.pkl`                  |
| `GI_scores_truth.py`        | Generate the ground truth GI scores                                      | `GI_scores_truth.csv`           |
| `GI_scores_gears_trueab.py` | Generate the GEARS predicted GI scores                                   | `GI_scores_gears_trueab.csv`    |
| `GI_scores_scouter_trueab.py` | Generate the Scouter predicted GI scores                               | `GI_scores_scouter_trueab.csv`  |
| `genes_with_hi_mean.npy`    | List of genes with high average expression level. <br> such tradition is adopted from GEARS code       |         						   |
