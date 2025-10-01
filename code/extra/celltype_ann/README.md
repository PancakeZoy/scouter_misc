### Pipeline Overview

1. **Data Preparation** (`gene_summary.py`): Extracts gene summaries from NCBI data, filters for genes in perturbation datasets (Dixit, Adamson, Norman), and creates both regular and K-562 cell line annotated summaries.

2. **Embedding Generation** (`embd_curation.py`): Uses OpenAI's text-embedding-3-small model to generate embeddings for both regular and annotated gene summaries, saving them as CSV files.

3. **Model Training** (`train_embd.py`): Trains Scouter models using gene embeddings on Adamson perturbation data across 5 splits, and evaluating performance.

### Output Files

- `GeneSummary.json` / `GeneSummary_ann.json`: Gene summaries with/without K-562 annotations
- `embeddings.csv` / `embeddings_ann.csv`: OpenAI embeddings for gene summaries
- `with_ann_metric.csv` / `adamson.csv`: Model performance results across splits
