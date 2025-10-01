from openai import OpenAI
import os
import pandas as pd
import json
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import random
from adjustText import adjust_text
# fmt: off

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding
########################### Embedding ###########################
with open("GeneSummary.json", "r") as file:
    summary = json.load(file)
embeddings = {}
for symbol, text in tqdm(summary.items()):
    embeddings[symbol] = get_embedding(text)
embeddings = pd.DataFrame(embeddings).T

with open("GeneSummary_ann.json", "r") as file:
    summary_ann = json.load(file)
embeddings_ann = {}
for symbol, text in tqdm(summary_ann.items()):
    embeddings_ann[symbol] = get_embedding(text)
embeddings_ann = pd.DataFrame(embeddings_ann).T

# Save embeddings to CSV files
embeddings.to_csv("embeddings.csv")
embeddings_ann.to_csv("embeddings_ann.csv")