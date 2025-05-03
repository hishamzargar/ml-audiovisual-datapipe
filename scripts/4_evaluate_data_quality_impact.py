import pathlib
import pandas as pd
import numpy as np
import librosa
import soundfile as sf # Good for reading WAV directly

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

from tqdm.auto import tqdm
import sys
import time

from typing import Optional
import traceback


# Manifests relative to project root
ORIGINAL_MANIFEST_REL = "data/development_subset_manifest.csv" # The first 100 samples
HQ_MANIFEST_REL = "data/processed/high_quality_manifest.csv" # The filtered ~57 samples

# Feature Extraction Parameters
TARGET_SR = 16000 # Should match the SR of preprocessed audio
N_MFCC = 13 # Number of MFCCs to extract

# Model Training Parameters
TEST_SIZE = 0.25
RANDOM_STATE = 42

def extract_audio_features(audio_path: pathlib.Path, target_sr: int, n_mfcc: int) -> Optional[np.ndarray]:
    """
    Loads audio and extracts MFCC features, returning the mean MFCC vector.
    Returns None if loading or feature extraction fails.
    """
    try:
        #Use librosa.load - handles various formats including MP4 audio stream
        y, sr = librosa.load(str(audio_path), sr=target_sr, mono=True)

        if len(y) == 0:
             print(f"Warning: Empty audio signal in {audio_path}. Skipping.", file=sys.stderr)
             return None
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

        # Aggregate features (mean across time)
        mean_mfccs = np.mean(mfccs, axis=1)
        return mean_mfccs
    except Exception as e:
        print(f"Error extracting features from {audio_path}: {e}", file=sys.stderr)
        return None
    
def prepare_data(manifest_df: pd.DataFrame, project_root: pathlib.Path, path_column: str):
    """Extracts features and labels from the manifest."""
    print(f"Preparing data for {len(manifest_df)} entries...")
    X_list = []
    y_list = []
    filepaths_list = []

    if path_column not in manifest_df.columns:
        print(f"Error: Specified path column '{path_column}' not found in manifest.", file=sys.stderr)
        return None, None, None
    if 'speaker_id' not in manifest_df.columns:
        print("Error: 'speaker_id' column not found in manifest.", file=sys.stderr)
        return None, None, None
    
    for index, row in tqdm(manifest_df.iterrows(), total=len(manifest_df), desc="Extracting Features"):
        relative_audio_path = row[path_column]
        speaker_id = row['speaker_id']
        abs_audio_path = (project_root / relative_audio_path).resolve()

        features = extract_audio_features(abs_audio_path, TARGET_SR, N_MFCC)

        if features is not None:
            X_list.append(features)
            y_list.append(speaker_id)
            filepaths_list.append(relative_audio_path)
        
    if not X_list:
        print("Error: No features could be extracted.", file=sys.stderr)
        return None, None, None
    
    X = np.array(X_list)
    # Encode string labels (speaker_id) to integers
    le = LabelEncoder()
    y = le.fit_transform(y_list)

    print(f"Data prepared: X shape={X.shape}, y shape={y.shape}, {len(le.classes_)} unique speakers.")
    # Add this to check distribution
    unique_labels, counts = np.unique(y, return_counts=True)
    print(f"  - Samples per speaker: min={counts.min()}, max={counts.max()}, avg={counts.mean():.2f}")
    return X, y, le

def train_evaluate(X: np.ndarray, y: np.ndarray, test_size_val: float):
    """Trains and evaluates a simple classifier."""
    results = {'accuracy': 0.0, 'f1_weighted': 0.0}
    n_samples = len(X)
    unique_classes, counts = np.unique(y, return_counts=True)
    n_classes = len(unique_classes)

    print(f"  Attempting evaluation with {n_samples} samples and {n_classes} unique speakers.")
    if n_samples < 4 or n_classes < 2: # Need enough samples for split and at least 2 classes
        print("  Warning: Insufficient data or classes (<2) for training/evaluation.", file=sys.stderr)
        return results

    can_stratify = np.all(counts >= 2) # Check if all classes have at least 2 samples

    try:
        # Split data - stratify if possible
        if can_stratify:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size_val, random_state=RANDOM_STATE, stratify=y
            )
            print(f"  Split: {len(X_train)} train / {len(X_test)} test (Stratified).")
        else:
             print("  Warning: Could not stratify train/test split (some speakers have < 2 samples). Splitting without stratification.", file=sys.stderr)
             X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size_val, random_state=RANDOM_STATE
             )
             print(f"  Split: {len(X_train)} train / {len(X_test)} test (Not Stratified).")

        # Check training set classes
        train_classes, train_counts = np.unique(y_train, return_counts=True)
        print(f"  Training set has {len(train_classes)} unique speakers.")
        # Check if any class in train set has only 1 sample (might cause issues for some models/metrics)
        if np.any(train_counts < 2):
             print(f"  Warning: {np.sum(train_counts < 2)} speaker(s) have only 1 sample in the training set.", file=sys.stderr)

        # --- Scale features ---
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # --- Add Checks Before Fit ---
        print(f"  X_train_scaled shape: {X_train_scaled.shape}, dtype: {X_train_scaled.dtype}")
        print(f"  y_train shape: {y_train.shape}, dtype: {y_train.dtype}")
        # Check for NaNs or Infs
        if np.isnan(X_train_scaled).any() or np.isinf(X_train_scaled).any():
            print("  Error: NaNs or Infs found in scaled training data! Cannot fit model.", file=sys.stderr)
            return results
        # Check for zero variance features (after scaling, std dev should be ~1 unless feature was constant)
        if np.any(np.isclose(X_train_scaled.std(axis=0), 0)):
             print("  Warning: Features with zero variance found after scaling in training data.", file=sys.stderr)
        # --- End Checks ---


        # --- Train model ---
        model = SVC(random_state=RANDOM_STATE, class_weight='balanced', probability=False) # probability=False might be slightly faster/more stable
        print("  Training SVC model...")
        model.fit(X_train_scaled, y_train) # <<< This is the critical call
        print("  Model fitting complete.")


        # --- Predict and Evaluate ---
        print("  Evaluating model...")
        y_pred = model.predict(X_test_scaled) # <<< This fails if fit didn't complete
        results['accuracy'] = accuracy_score(y_test, y_pred)
        results['f1_weighted'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    except Exception as e:
        # --- Modified Exception Print ---
        print(f"  Error during model training or evaluation: {type(e).__name__} - {e}", file=sys.stderr) # Print type and message
        print("  Traceback:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr) # Print the full traceback
        # --- End Modification ---

    return results


    
if __name__ == "__main__":
    overall_start_time = time.time()
    project_root = pathlib.Path(__file__).resolve().parent.parent

    # Construct absolute paths
    original_manifest_path = (project_root / ORIGINAL_MANIFEST_REL).resolve()
    hq_manifest_path = (project_root / HQ_MANIFEST_REL).resolve()

    # --- Load Manifests ---
    print("Loading manifests...")
    if not original_manifest_path.is_file() or not hq_manifest_path.is_file():
         print("Error: One or both manifest files not found. Ensure previous steps ran correctly.", file=sys.stderr)
         sys.exit(1)

    df_orig = pd.read_csv(original_manifest_path)
    df_hq = pd.read_csv(hq_manifest_path)
    print(f"Loaded original manifest with {len(df_orig)} entries.")
    print(f"Loaded high-quality manifest with {len(df_hq)} entries.")

    # --- Process and Evaluate ORIGINAL Subset ---
    print("\n--- Evaluating ORIGINAL Subset ---")
    X_orig, y_orig, _ = prepare_data(df_orig, project_root, path_column='filepath')
    if X_orig is not None:
        results_orig = train_evaluate(X_orig, y_orig, TEST_SIZE)
        print(f"Results on ORIGINAL ({len(df_orig)} samples):")
        print(f"  Accuracy: {results_orig['accuracy']:.4f}")
        print(f"  F1 Weighted: {results_orig['f1_weighted']:.4f}")
    else:
        print("Could not prepare data for original subset.")
        results_orig = {'accuracy': 0.0, 'f1_weighted': 0.0}


    # --- Process and Evaluate HIGH-QUALITY Subset ---
    print("\n--- Evaluating HIGH-QUALITY Subset ---")
    X_hq, y_hq, _ = prepare_data(df_hq, project_root, path_column='filepath_audio')
    if X_hq is not None:
        results_hq = train_evaluate(X_hq, y_hq, TEST_SIZE)
        print(f"Results on HIGH-QUALITY ({len(df_hq)} samples):")
        print(f"  Accuracy: {results_hq['accuracy']:.4f}")
        print(f"  F1 Weighted: {results_hq['f1_weighted']:.4f}")
    else:
        print("Could not prepare data for high-quality subset.")
        results_hq = {'accuracy': 0.0, 'f1_weighted': 0.0}


    # --- Comparison ---
    print("\n--- Comparison ---")
    print(f"Original Set Accuracy:      {results_orig['accuracy']:.4f}")
    print(f"High-Quality Set Accuracy:  {results_hq['accuracy']:.4f}")
    print(f"Accuracy Change:            {results_hq['accuracy'] - results_orig['accuracy']:+.4f}\n")

    print(f"Original Set F1 Weighted:   {results_orig['f1_weighted']:.4f}")
    print(f"High-Quality Set F1 Weighted:{results_hq['f1_weighted']:.4f}")
    print(f"F1 Change:                  {results_hq['f1_weighted'] - results_orig['f1_weighted']:+.4f}")

    overall_end_time = time.time()
    print(f"\nTotal evaluation script time: {overall_end_time - overall_start_time:.2f} seconds.")
    print("Script finished.")