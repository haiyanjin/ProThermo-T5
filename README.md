# ProThermo-T5
A process-aware machine learning framework (ProtT5-XL + XGBoost) predicting protein thermostability (Tm). It uniquely simulates matrix-induced stability shifts ("Lysis Shock" in cell vs. lysate) to guide the rational design of robust affinity ligands and chromatography resins for downstream processing.
Markdown
# ProThermo-T5: Process-Aware Protein Thermostability Predictor

ProThermo-T5 is a highly generalized, process-aware machine learning framework tailored for predicting protein thermostability (Tm) under specific chromatographic matrix conditions. By integrating ProtT5-XL-UniRef50 language model embeddings with an optimized XGBoost engine, it accurately simulates **"Lysis Shock"**—the rapid destabilization proteins experience when transitioning from an intact cell to a harsh cell lysate feedstock.

This repository provides tools for both **high-throughput batch prediction** (for separation scientists designing robust affinity ligands) and **model training/benchmarking** (for reproducibility and architecture optimization).

---

## ⚙️ Installation & Environment Setup

This project requires Python 3.9+ and PyTorch with CUDA support. 
### Option 1: Using Pip (Local Environment)
Install the required dependencies directly using `pip`:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pandas numpy scikit-learn xgboost transformers sentencepiece tqdm biopython fair-esm
```

### Option 2: Using Docker
We recommend using a Docker container for a clean and reproducible environment. You can build your environment using the following base image and dependencies:

Base Image: ```pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime```

Key Packages: ```jupyter```, ```pandas```, ```scikit-learn```, ```xgboost```, ```transformers```, ```sentencepiece```

## 🚀 Usage 1: Prediction (```prediction.py```)
The ```prediction.py``` script allows you to predict the melting temperature (Tm) of proteins using the pre-trained ProThermo-T5 model. It supports both single-sequence prediction and high-throughput batch prediction from a CSV file.

Ensure you have downloaded the trained model file (```ProThermo-T5.json```) into your working directory.

Single Sequence Prediction
To quickly test a single protein sequence (e.g., Protein A Z-domain):

```bash
python prediction.py \
    --sequence "VDNKFNKEQQNAFYEILHLPNLNEEQRNAFIQSLKDDPSQSANLLAEAKKLNDAQAPK" \
    --condition 0.0 \
    --ogt 37.0
```
```--condition```: ```0.0``` simulates Cell Lysate (Capture step), ```1.0``` simulates Intact Cell.

```--ogt```: Optimum Growth Temperature (Default is 34.0).

Batch Prediction (CSV)
For high-throughput screening of mutant libraries:

```bash
python prediction.py \
    --csv_path test_sequences.csv \
    --seq_col sequence \
    --out_csv final_results.csv \
    --condition 0.0
```
```--csv_path```: Path to your input CSV file.

```--seq_col```: The name of the column containing protein sequences (e.g., ```sequence```).

```--out_csv```: The name of the output file where predictions will be saved.

## 📊 Usage 2: Training & Benchmarking (```training.py```)
The ```training.py``` script is used to reproduce our architecture optimization benchmarking. It extracts deep biophysical embeddings using state-of-the-art Protein Language Models (ProtT5, ESM-2, Ankh) and evaluates them using 5-Fold Cross-Validation.

Extract Embeddings & Train (Default: ProtT5)
```bash
python training.py --model_type prott5 --data_path final_unique_dataset_clean.csv
```
Benchmark with Other PLMs
You can easily switch the language model backend to compare performance:

```bash
# Train using ESM-2 (650M)
python training.py --model_type esm2

# Train using Ankh-Large
python training.py --model_type ankh
```
Skip Embedding Extraction (Train Only)
If you have already extracted and saved the ```.npy``` embedding files in the ```/results``` folder, you can skip the time-consuming extraction step and jump straight to training:

```bash
python training.py --model_type prott5 --skip_embedding
```
Additional Arguments
```--gpu_id```: Specify which GPU to use (e.g., ```--gpu_id 0```).

```--batch_size```: Adjust PLM inference batch size to prevent Out-Of-Memory (OOM) errors (e.g., ```--batch_size 2```).

```--n_jobs```: Number of CPU cores allocated for ML models like XGBoost/Random Forest (e.g., ```--n_jobs 10```).

## 📝 Contact & Citation
If you use ProThermo-T5 in your research, please cite our upcoming paper in the Journal of Chromatography A.
For any questions or issues, please open an issue in this repository.

