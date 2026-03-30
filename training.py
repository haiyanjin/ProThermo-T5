import os
import re
import argparse
import numpy as np
import pandas as pd
import torch
import joblib
from tqdm import tqdm
from scipy.stats import pearsonr

# Transformers
from transformers import AutoTokenizer, AutoModel, T5Tokenizer, T5EncoderModel

# Machine Learning Models
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def get_device():
    if torch.cuda.is_available():
        device = "cuda"
        print(f"[INFO] Using device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = "cpu"
        print("[WARN] CUDA not available. Using CPU.")
    return device

def extract_embeddings(data_path, exp_name, model_type, batch_size):
    print(f"\n{'='*20}")
    print(f"🚀 Starting Embedding Extraction: [{exp_name}]")
    print(f"🧬 Model Type: {model_type.upper()}")
    print(f"{'='*20}")
    
    # 1. Load Data
    if not os.path.exists(data_path):
        print(f"[ERROR] Data file not found: {data_path}")
        return False
    
    df = pd.read_csv(data_path)
    sequences = df['sequence'].tolist()
    labels = np.array(df['tm'].tolist())
    
    # Fill missing OGT with mean, and extract conditions
    ogts = df['ogt'].fillna(df['ogt'].mean()).tolist()
    conditions = df['condition'].tolist()
    
    device = get_device()
    
    # 2. Load Model based on selection
    print(f"[INFO] Loading {model_type.upper()} model...")
    try:
        if model_type.lower() == 'prott5':
            model_name = "Rostlab/prot_t5_xl_half_uniref50-enc"
            tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
            model = T5EncoderModel.from_pretrained(model_name).to(device)
            # model.half() # Uncomment if you want to force fp16 for ProtT5
            
        elif model_type.lower() == 'esm2':
            # Using the 650M parameter model as a strong baseline. 
            # Can be changed to "facebook/esm2_t36_3B_UR50D" for the 3B version
            model_name = "facebook/esm2_t33_650M_UR50D" 
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name).to(device)
            
        elif model_type.lower() == 'ankh':
            model_name = "ElnaggarLab/ankh-large"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name).to(device)
            
        else:
            print(f"[ERROR] Unsupported model type: {model_type}")
            return False
            
        model.eval()
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return False

    features = []
    
    print(f"[INFO] Extracting embeddings (Batch Size = {batch_size})...")
    for i in tqdm(range(0, len(sequences), batch_size), desc="Processing Batches"):
        batch_seqs = sequences[i:i+batch_size]
        
        # Sequence Preprocessing
        if model_type.lower() == 'prott5':
            # ProtT5 requires spaces between amino acids
            batch_seqs_prep = [" ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in batch_seqs]
        else:
            # ESM-2 and Ankh take standard contiguous strings
            batch_seqs_prep = [re.sub(r"[UZOB]", "X", seq) for seq in batch_seqs]
        
        # Tokenization
        ids = tokenizer.batch_encode_plus(batch_seqs_prep, add_special_tokens=True, padding="longest", return_tensors="pt")
        input_ids = ids['input_ids'].to(device)
        attention_mask = ids['attention_mask'].to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state
            
            # Masked Mean Pooling (ignores padding tokens)
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * attention_mask_expanded, 1)
            sum_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
            batch_emb = (sum_embeddings / sum_mask).cpu().numpy()
            
        # Append Biological & Process Context (OGT & Condition)
        batch_ogts = ogts[i:i+batch_size]
        batch_conds = conditions[i:i+batch_size]
        
        for j in range(len(batch_emb)):
            ogt_val = batch_ogts[j]
            # 1.0 for Intact Cell, 0.0 for Cell Lysate
            cond_val = 1.0 if 'cell' in str(batch_conds[j]).lower() else 0.0
            
            combined_feature = np.concatenate([batch_emb[j], [ogt_val], [cond_val]])
            features.append(combined_feature)
            
    # 3. Save Results
    X = np.array(features)
    y = labels
    
    os.makedirs("results", exist_ok=True)
    np.save(f"results/X_{exp_name}.npy", X)
    np.save(f"results/y_{exp_name}.npy", y)
    print(f"[SUCCESS] Embeddings saved to: results/X_{exp_name}.npy")
    return True

def run_training_evaluation(exp_name, n_jobs):
    print(f"\n{'='*20}")
    print(f"📊 Starting Model Training & Evaluation: [{exp_name}]")
    print(f"{'='*20}")
    
    # 1. Load Embeddings
    X_path = f"results/X_{exp_name}.npy"
    y_path = f"results/y_{exp_name}.npy"
    
    if os.path.exists(X_path) and os.path.exists(y_path):
        print(f"[INFO] Loading dataset...")
        X = np.load(X_path)
        y = np.load(y_path)
        print(f"[INFO] Dataset shape: X={X.shape}, y={y.shape}")
    else:
        print(f"[ERROR] Data files not found for experiment '{exp_name}'.")
        print(" -> Please run the embedding extraction step first.")
        return

    # 2. Evaluation Wrapper Function
    def evaluate_model(model, X, y, model_name):
        print(f"\n🔥 Training [{model_name}]...")
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        scores = {"R2": [], "PCC": [], "MAE": [], "MSE": [], "RMSE": []}
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            # Calculate metrics
            scores["R2"].append(r2_score(y_val, y_pred))
            scores["PCC"].append(pearsonr(y_val, y_pred)[0])
            scores["MAE"].append(mean_absolute_error(y_val, y_pred))
            mse = mean_squared_error(y_val, y_pred)
            scores["MSE"].append(mse)
            scores["RMSE"].append(np.sqrt(mse))
            
        print(f"\n🏆 Final 5-Fold CV Results: [{model_name}]")
        print("-" * 35)
        print(f"{'Metric':<10} | {'Mean Score':<15}")
        print("-" * 35)
        print(f"{'R2':<10} | {np.mean(scores['R2']):.4f}")
        print(f"{'PCC':<10} | {np.mean(scores['PCC']):.4f}")
        print(f"{'MAE':<10} | {np.mean(scores['MAE']):.4f}")
        print(f"{'MSE':<10} | {np.mean(scores['MSE']):.4f}")
        print(f"{'RMSE':<10} | {np.mean(scores['RMSE']):.4f}")
        print("-" * 35)

    # 3. Initialize and Evaluate Models
    
    # (1) Random Forest
    rf = RandomForestRegressor(n_estimators=100, n_jobs=n_jobs, random_state=42)
    evaluate_model(rf, X, y, "Random Forest")
    
    # (2) XGBoost
    xgb = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, n_jobs=n_jobs, random_state=42)
    evaluate_model(xgb, X, y, "XGBoost")
    
    # (3) MLP (Neural Network)
    mlp = MLPRegressor(hidden_layer_sizes=(512, 128), activation='relu', solver='adam', max_iter=500, random_state=42)
    evaluate_model(mlp, X, y, "MLP (Neural Net)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ProThermo-T5: Protein Thermostability Training Pipeline")
    
    # Hardware Configuration
    parser.add_argument("--gpu_id", type=str, default="0", help="CUDA_VISIBLE_DEVICES ID (default: 0)")
    parser.add_argument("--n_jobs", type=int, default=10, help="Number of CPU cores for ML models (default: 10)")
    
    # Pipeline Toggles
    parser.add_argument("--skip_embedding", action="store_true", help="Skip extraction and only run training on existing .npy files")
    
    # Data & Model Configuration
    parser.add_argument("--data_path", type=str, default="final_unique_dataset_clean.csv", help="Path to input CSV dataset")
    parser.add_argument("--model_type", type=str, choices=["prott5", "esm2", "ankh"], default="prott5", help="Choose PLM to extract embeddings")
    parser.add_argument("--exp_name", type=str, default="Clean_18k", help="Experiment name (used for saving/loading .npy files)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for PLM inference (reduce if CUDA Out Of Memory)")

    args = parser.parse_args()

    # Apply GPU settings
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    
    # Dynamic experiment naming based on model choice
    full_exp_name = f"{args.exp_name}_{args.model_type.upper()}"

    # Step 1: Extract Embeddings (if not skipped)
    if not args.skip_embedding:
        success = extract_embeddings(args.data_path, full_exp_name, args.model_type, args.batch_size)
        if not success:
            exit(1)
            
    # Step 2: Train and Evaluate Models
    run_training_evaluation(full_exp_name, args.n_jobs)