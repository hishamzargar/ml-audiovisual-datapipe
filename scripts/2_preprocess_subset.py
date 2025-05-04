import pathlib
import pandas as pd
import cv2
import librosa
import soundfile as sf
import numpy as np
from tqdm.auto import tqdm
import sys
import os

# --- Configuration ---
MANIFEST_PATH = pathlib.Path("../data/development_subset_manifest2.csv")
OUTPUT_BASE_DIR = pathlib.Path("../data/processed/attempt2_standardized_subset")

#MANIFEST_PATH = pathlib.Path("../data/development_subset_manifest.csv") #Old path
#OUTPUT_BASE_DIR = pathlib.Path("../data/processed/standardized_subset") #Old Path

# Target format parameters
TARGET_SR = 16000 # Target audio sample rate (Hz)
TARGET_FPS = 25 # Target video frame rate
TARGET_WIDTH = 224 # Target video width
TARGET_HEIGHT = 224 # Target video height
TARGET_AUDIO_CHANNELS = 1 # Target audio channels (Mono)
VIDEO_CODEC = 'mp4v'

def preprocess_file(relative_filepath_str: str, output_dir: pathlib.Path, project_root: pathlib.Path):
    """
    Loads one audio-visual file, preprocesses it (standardize sr, fps, resolution),
    and saves the results (video as .mp4, audio as .wav).
    Returns True if successful, False otherwise.
    """
    input_filepath = project_root / relative_filepath_str
    base_filename = input_filepath.stem     # Get filename without extension

    # Define output paths
    speaker_id = input_filepath.parent.parent.name
    utterance_id = input_filepath.parent.name
    output_subdir = output_dir / speaker_id / utterance_id
    output_subdir.mkdir(parents=True, exist_ok=True)

    output_video_path = output_subdir / f"{base_filename}_video.mp4"
    output_audio_path = output_subdir / f"{base_filename}_audio.wav"

    # --- Audio Processing ---
    try:
        # Load audio
        y, sr_org = librosa.load(str(input_filepath), sr=None, mono=False)

        #Convert to mono if necesssary
        if y.ndim > 1 and y.shape[0] == 2: # Check if stereo
             y_mono = librosa.to_mono(y)
        elif y.ndim == 1:
            y_mono = y # Already mono
        else:
            # Handle unexpected shapes (e.g., more than 2 channels) - skip or log error
            print(f"Warning: Skipping audio for {input_filepath} due to unexpected shape {y.shape}", file=sys.stderr)
            return False # Indicate failure for this file
        
        # Resample if necessary
        if sr_org != TARGET_SR:
            y_resampled = librosa.resample(y_mono, orig_sr=sr_org, target_sr=TARGET_SR)
        else:
            y_resampled = y_mono

        # Save processed audio
        sf.write(str(output_audio_path), y_resampled, TARGET_SR, subtype='PCM_16')

    except Exception as e:
        print(f"Error processing audio for {input_filepath}: {e}", file=sys.stderr)
        return False # Indicate failure
    

    # --- Video Processing ---
    try:
        cap = cv2.VideoCapture(str(input_filepath))
        if not cap.isOpened():
            print(f"Error: Could not open video file {input_filepath}", file=sys.stderr)
            return False
        
        # Initialize VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
        writer = cv2.VideoWriter(
            str(output_video_path),
            fourcc,
            TARGET_FPS,
            (TARGET_WIDTH, TARGET_WIDTH)
        )
        if not writer.isOpened():
            print(f"Error: Could not open VideoWriter for {output_video_path}", file=sys.stderr)
            cap.release()
            return False
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Resize frame
            # Use INTER_AREA for shrinking, INTER_LINEAR or INTER_CUBIC for enlarging
            resized_frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)

            #Write frame
            writer.write(resized_frame)
            frame_count += 1

        # Release resources
        cap.release()
        writer.release()

        if frame_count == 0:
            print(f"Warning: No frames processed for video {input_filepath}", file=sys.stderr)
            # Optionally delete the empty output video file os.remove(output_video_path)
            return False
        
    except Exception as e:
        print(f"Error processing video for {input_filepath}: {e}", file=sys.stderr)
        # Clean up potentially partially written file if writer was opened
        if 'writer' in locals() and writer.isOpened():
             writer.release()
        # Optionally delete the file if it exists os.remove(output_video_path)
        return False # Indicate failure
    
    return True


if __name__ == "__main__":
    project_root = pathlib.Path(__file__).resolve().parent.parent
    manifest_full_path = (project_root / "data" / "development_subset_attempt2.csv").resolve()
    output_base_full_path = (project_root / "data" / "processed" / "attempt2_standardized_subset").resolve()

    print(f"Project root identified as: {project_root}")
    print(f"Loading manifest from: {manifest_full_path}")
    if not manifest_full_path.is_file():
        print(f"Error: Manifest file not found at {manifest_full_path}", file=sys.stderr)
        print("Please ensure the manifest exists and the script is run from the project root or check paths.")
        sys.exit(1)
    
    manifest_df = pd.read_csv(manifest_full_path)
    print(f"Found {len(manifest_df)} files in manifest.")
    print(f"Output directory: {output_base_full_path}")
    output_base_full_path.mkdir(parents=True, exist_ok=True)

    success_count = 0
    fail_count = 0

    print("\nStarting preprocessing...")
    # Use tqdm for progress bar over DataFrame rows
    for index, row, in tqdm(manifest_df.iterrows(), total=manifest_df.shape[0], desc="Processing_files"):
        filepath = row['filepath']
        if preprocess_file(filepath, output_base_full_path, project_root):
            success_count += 1
        else:
            fail_count += 1
            print(f"Failed processing: {filepath}") # Add more context if needed

    print("\nPreprocessing finished.")
    print(f"Successfully processed: {success_count} files.")
    print(f"Failed to process: {fail_count} files.")
    if fail_count > 0:
        print("Check logs for error details.")




    

        






