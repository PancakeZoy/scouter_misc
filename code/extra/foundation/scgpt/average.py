import pickle
import pandas as pd

ds = 'norman'
for i in range(1, 6):
    # Step 1: Load the original pickle file
    with open(f"{ds}/test_res_{i}.pkl", "rb") as f:
        test_res = pickle.load(f)

    pert_cat = test_res['pert_cat']
    mean_res = {}

    # Step 2: Convert each array to group-mean DataFrame
    for key, val in test_res.items():
        if key == 'pert_cat':
            continue
        df = pd.DataFrame(val, index=pert_cat)
        df.index.name = 'perturbation'
        mean_df = df.groupby('perturbation').mean()
        mean_res[key] = mean_df

    # Step 3: Save the result dict as a new .pkl file
    with open(f"{ds}/mean_res_{i}.pkl", "wb") as f:
        pickle.dump(mean_res, f)