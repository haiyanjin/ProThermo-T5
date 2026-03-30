import os
import re
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import T5Tokenizer, T5EncoderModel
from xgboost import XGBRegressor

class ProThermoPredictor:
    def __init__(self, xgb_model_path="ProThermo-T5.json", gpu_id="0"):
        """
        Initializes the ProThermo-T5 predictor by loading the ProtT5-XL PLM 
        and the trained XGBoost regressor.
        """
        print(f"\n{'='*40}")
        print("🚀 Initializing ProThermo-T5 Predictor")
        print(f"{'='*40}")

        # 1. Device Configuration
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"[INFO] Using Device: CUDA ({torch.cuda.get_device_name(0)})")
        else:
            self.device = torch.device("cpu")
            print("[WARN] No CUDA detected. Using CPU mode (Inference may be slow).")

        # 2. Load XGBoost Model
        if not os.path.exists(xgb_model_path):
            raise FileNotFoundError(f"[ERROR] XGBoost model not found at: {xgb_model_path}")
        
        print(f"[INFO] Loading XGBoost Model: {xgb_model_path}")
        self.xgb_model = XGBRegressor()
        self.xgb_model.load_model(xgb_model_path)

        # 3. Load ProtT5-XL Model
        print(f"[INFO] Loading ProtT5-XL-Half Model to {self.device}...")
        model_name = "Rostlab/prot_t5_xl_half_uniref50-enc"
        
        # Suppress legacy warnings
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False, legacy=False)
        self.plm_model = T5EncoderModel.from_pretrained(model_name).to(self.device)
        self.plm_model.eval()
        print("[SUCCESS] Predictor is ready!\n")

    def predict_sequence(self, sequence, ogt=34.0, condition=0.0):
        """
        Predicts the melting temperature (Tm) for a single protein sequence.
        """
        if not isinstance(sequence, str) or len(sequence.strip()) < 5:
            return np.nan
        
        # 1. Preprocess Sequence (Replace rare AAs and add spaces)
        seq_clean = " ".join(list(re.sub(r"[UZOB]", "X", sequence.strip().upper())))
        
        # 2. Tokenize
        ids = self.tokenizer.batch_encode_plus(
            [seq_clean], add_special_tokens=True, padding="longest", return_tensors="pt"
        )
        input_ids = ids['input_ids'].to(self.device)
        attention_mask = ids['attention_mask'].to(self.device)
        
        # 3. Extract Embeddings
        with torch.no_grad():
            embedding = self.plm_model(input_ids=input_ids, attention_mask=attention_mask)
            seq_len = (attention_mask[0] == 1).sum()
            # Mean pooling excluding the EOS token
            emb_vector = embedding.last_hidden_state[0, :seq_len-1].mean(dim=0).cpu().numpy()
            
        # 4. Concatenate Metadata (OGT and Condition)
        metadata = np.array([ogt, condition])
        final_input = np.concatenate([emb_vector, metadata]).reshape(1, -1)
        
        # 5. Predict Tm
        pred_tm = self.xgb_model.predict(final_input)[0]
        return float(pred_tm)

def run_batch_prediction(predictor, input_csv, output_csv, seq_column, ogt, condition):
    """
    Handles high-throughput batch prediction from a CSV file.
    """
    if not os.path.exists(input_csv):
        print(f"[ERROR] Input CSV not found: {input_csv}")
        return
        
    print(f"[INFO] Loading CSV: {input_csv}")
    df = pd.read_csv(input_csv)
    
    if seq_column not in df.columns:
        print(f"[ERROR] Column '{seq_column}' not found in CSV.")
        print(f"       Available columns: {list(df.columns)}")
        return
        
    print(f"[INFO] Starting Batch Prediction for {len(df)} sequences...")
    print(f"       -> OGT: {ogt} | Condition: {condition} (0.0=Lysate, 1.0=Cell)")
    
    predicted_tms = []
    
    # Iterate with progress bar
    for seq in tqdm(df[seq_column], desc="Predicting Tm"):
        try:
            tm = predictor.predict_sequence(str(seq), ogt=ogt, condition=condition)
            predicted_tms.append(tm)
        except Exception as e:
            # Handle potential errors gracefully without stopping the batch
            predicted_tms.append(np.nan)
            
    # Save Results
    df['Predicted_Tm'] = predicted_tms
    df.to_csv(output_csv, index=False)
    
    print(f"\n[SUCCESS] Batch Prediction Completed!")
    print(f"[SUCCESS] Results saved to: {output_csv}")
    print("\n[Preview - Top 5 Results]")
    print(df[[seq_column, 'Predicted_Tm']].head().to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ProThermo-T5: Protein Thermostability Predictor")
    
    # Hardware & Model Params
    parser.add_argument("--gpu_id", type=str, default="0", help="CUDA GPU ID (default: 0)")
    parser.add_argument("--xgb_model", type=str, default="ProThermo-T5.json", help="Path to trained XGBoost model")
    
    # Biological Metadata
    parser.add_argument("--ogt", type=float, default=34.0, help="Optimum Growth Temperature (default: 34.0)")
    parser.add_argument("--condition", type=float, default=0.0, help="Matrix Condition: 0.0 for Lysate, 1.0 for Cell (default: 0.0)")
    
    # Execution Modes
    parser.add_argument("--sequence", type=str, help="Single protein sequence to predict")
    
    # Batch Processing Params
    parser.add_argument("--csv_path", type=str, help="Path to input CSV for batch prediction")
    parser.add_argument("--seq_col", type=str, default="sequence", help="Column name containing sequences in CSV (default: 'sequence')")
    parser.add_argument("--out_csv", type=str, default="BPT_results.csv", help="Output CSV file name (default: BPT_results.csv)")

    args = parser.parse_args()

    # Ensure user provides either a sequence or a CSV
    if not args.sequence and not args.csv_path:
        print("[ERROR] Please provide either a single sequence (--sequence) OR a CSV file (--csv_path).")
        print("Example 1: python prediction.py --sequence MYEAVIGLEVLH...")
        print("Example 2: python prediction.py --csv_path input.csv --seq_col sequence")
        exit(1)

    # Initialize Predictor
    predictor = ProThermoPredictor(xgb_model_path=args.xgb_model, gpu_id=args.gpu_id)

    # Mode 1: Single Sequence Prediction
    if args.sequence:
        print(f"[INFO] Predicting single sequence...")
        tm = predictor.predict_sequence(args.sequence, ogt=args.ogt, condition=args.condition)
        print("-" * 40)
        print(f"🧬 Sequence Length : {len(args.sequence)} AAs")
        print(f"🌡️ Predicted Tm   : {tm:.2f} °C")
        print("-" * 40)

    # Mode 2: Batch CSV Prediction
    if args.csv_path:
        run_batch_prediction(
            predictor=predictor, 
            input_csv=args.csv_path, 
            output_csv=args.out_csv, 
            seq_column=args.seq_col, 
            ogt=args.ogt, 
            condition=args.condition
        )