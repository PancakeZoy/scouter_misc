from openai import OpenAI
import os
import numpy as np
import json
from tqdm import tqdm
import pickle
import argparse
import random
# fmt: off

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

args = argparse.ArgumentParser()
args.add_argument("sample_portion_index", type=int, default=0)
args = args.parse_args()

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding
########################### Embedding ###########################
with open("GeneSummary.json", "r") as file:
    summary = json.load(file)
    
sample_portion_summary = {}
for symbol, summ in summary.items():
    words = summ.split()
    random.seed(1)
    random.shuffle(words)
    subsets = np.array_split(words, 10)
    subsets = [" ".join(sub) for sub in subsets]
    progressive_summary = [" ".join(subsets[:i+1]) for i in range(len(subsets))]
    sample_portion_summary[symbol] = progressive_summary

portion = args.sample_portion_index
embeddings = {}
for symbol, prog_summ in tqdm(sample_portion_summary.items(), desc=f"Proportion: {portion}"):
    summ = prog_summ[portion]
    embeddings[symbol] = get_embedding(summ)

with open(f"randomwords_embeddings/embeddings_Proportion{(portion + 1) * 10}.pkl", "wb") as file:
    pickle.dump(embeddings, file)
