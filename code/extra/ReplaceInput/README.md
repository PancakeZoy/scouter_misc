# ReplaceInput: GO Terms to LLM Embeddings Replacement

Replace GO terms with LLM embeddings for GEARS and BioLORD models.

## Usage

1. **Prepare data:**
Run `biolord_h5ad.ipynb` and `GEARS_GO.py` to prepare the gene representation swap work.

2. **Install modified GEARS:**
   ```bash
   cd gears2
   pip install -e .
   ```

3. **Run training:** Use scripts in `/code/biolord/` and `/code/GEARS/` as usual.
The modified GEARS version in `gears2/` extracts embeddings instead of GO terms during training.
