from openai import OpenAI
import os
import numpy as np
import json
from tqdm import tqdm
import pickle
import argparse
# fmt: off

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

args = argparse.ArgumentParser()
args.add_argument("portion_index", type=int, default=0)
args = args.parse_args()

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding
########################### Embedding ###########################
with open("GeneSummary.json", "r") as file:
    summary = json.load(file)
    
proprtion_summary = {}
for symbol, summ in summary.items():
    subsets = np.array_split(summ.split(), 9)
    subsets = [" ".join(sub) for sub in subsets]
    subsets.insert(0, "Gene: " + symbol)
    progressive_summary = [" ".join(subsets[: i + 1]) for i in range(len(subsets))]
    proprtion_summary[symbol] = progressive_summary

portion = args.portion_index
embeddings = {}
for symbol, prog_summ in tqdm(proprtion_summary.items(), desc=f"Proportion: {portion}"):
    summ = prog_summ[portion]
    embeddings[symbol] = get_embedding(summ)

with open(f"../truncated_embeddings/embeddings_Proportion{(portion + 1) * 10}.pkl", "wb") as file:
    pickle.dump(embeddings, file)
