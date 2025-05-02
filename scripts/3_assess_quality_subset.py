import pathlib
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
from tqdm.auto import tqdm
import sys
import os
import torch
import torchaudio
import soundfile as sf

# --- Configuration ---
# Directory containing the standardized video/audio files
PROCESSED_DATA_DIR_REL = "data/processed/standardized_subset"
# File to save quality scores
QUALITY_SCORES_PATH_REL = "data/processed/quality_scores.csv"

# MediaPipe Face Detection parameters
MIN_DETECTION_CONFIDENCE = 0.5
# Frame sampling (process every Nth frame to speed up, 1 = process all)
FRAME_SAMPLING_STRIDE = 1

# VAD Parameters
VAD_SAMPLE_RATE = 16000 # Silero VAD expects 16kHz

# Moved outside function for efficiency, loaded once
VAD_MODEL_AVAILABLE = False
vad_model = None
vad_utils = None
try:
    # Determine device: prioritize CUDA, then MPS, fallback to CPU
    if torch.cuda.is_available():
        vad_device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): # Check for MPS
        vad_device = "mps"
    else:
        vad_device = "cpu"

    print(f"Determined compute device for VAD: {vad_device}")
    print(f"Loading Silero VAD model (PyTorch version) on device: {vad_device}")

    # Using torch.hub to load the PyTorch model (will download on first run if needed)
    vad_model, vad_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                      model='silero_vad',
                                      force_reload=False,
                                      onnx=False, # IMPORTANT: Use PyTorch model for MPS
                                      )

    # Move the model to the selected device
    vad_model.to(vad_device)
    VAD_MODEL_AVAILABLE = True # Flag that VAD is ready
    print("Silero VAD model loaded and moved to device successfully.")

except Exception as e:
    print(f"Error loading Silero VAD model or moving to device '{vad_device}': {e}", file=sys.stderr)
    print("Please ensure you have torch and torchaudio installed and internet connection for first download.", file=sys.stderr)
    vad_model = None # Ensure it's None if loading failed

def find_processed_files(processed_dir: pathlib.Path) -> list[tuple[pathlib.Path, pathlib.Path]]:
    """
    Scans the directory for processed video files (_video.mp4)
    and finds their corresponding audio files (_audio.wav).
    Returns a list of tuples: [(video_path, audio_path), ...].
    """
    pairs = []
    print(f"Scanning for processed file pairs in: {processed_dir.resolve()}")
    video_files = list(processed_dir.rglob('*_video.mp4'))
    print(f"Found {len(video_files)} processed video files.")

    for video_path in video_files:
        expected_audio_path = video_path.with_name(video_path.stem.replace('_video', '_audio')).with_suffix('.wav')
        if expected_audio_path.is_file():
            pairs.append((video_path, expected_audio_path))
        else:
            print(f"Warning: Corresponding audio file not found for {video_path}", file=sys.stderr)

    print(f"Found {len(pairs)} matching video/audio pairs.")
    if not pairs:
         print(f"Warning: No processed file pairs found in {processed_dir}", file=sys.stderr)
    return pairs

def calculate_video_quality(video_path: pathlib.Path, face_detector) -> dict:
    """
    Calculates video quality metrics for a single video file.
    Metrics: face_presence_ratio, avg_blur_score (Laplacian variance on face).
    """
    metrics = {
        'face_presence_ratio': 0.0,
        'avg_blur_score': np.nan    # Use NaN if no face detected for blur calc
    }
    total_frames = 0
    frames_with_face = 0
    total_laplacian_variance = 0.0
    frames_with_face_for_blur_calc = 0

    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error opening video file: {video_path}", file=sys.stderr)
            return metrics # Return default metrics

        while cap.isOpened():
            frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) # Current frame number
            ret, frame = cap.read()
            if not ret:
                break # End of video

            # Process only every Nth frame if stride > 1
            if frame_pos % FRAME_SAMPLING_STRIDE != 0:
                continue

            total_frames += 1

            # Convert frame to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detector.process(frame_rgb)

            face_detected_in_frame = False
            if results.detections:
                # Consider face detected if any detection meets confidence threshold
                for detection in results.detections:
                    if detection.score[0] >= MIN_DETECTION_CONFIDENCE:
                        face_detected_in_frame = True
                        # --- Calculate Blur on Face ---
                        try:
                            # Get bounding box relative coordinates
                            bboxC = detection.location_data.relative_bounding_box
                            ih, iw, _ = frame.shape
                            # Calculate absolute pixel coordinates (handle potential off-by-one errors)
                            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                                         int(bboxC.width * iw), int(bboxC.height * ih)
                            x = max(0, x)
                            y = max(0, y)

                            # Crop face region (ensure width/height are positive)
                            if w>0 and h>0:
                                face_roi = frame[y:y+h, x:x+w]
                                # Convert face ROI to grayscale for Laplacian
                                face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                                # Calculate Laplacian variance
                                lap_var = cv2.Laplacian(face_gray, cv2.CV_64F).var()
                                total_laplacian_variance += lap_var
                                frames_with_face_for_blur_calc += 1
                        except Exception as e_blur:
                            print(f"Warning: Error calculating blur for {video_path} frame {frame_pos}: {e_blur}", file=sys.stderr)
                        break # Only process first confident face for blur in this frame
            if face_detected_in_frame:
                frames_with_face += 1
        
        cap.release()

        # --- Final Metric Calculation ---
        if total_frames > 0:
            metrics['face_presence_ratio'] = frames_with_face / total_frames

        if frames_with_face_for_blur_calc > 0:
            metrics['avg_blur_score'] = total_laplacian_variance / frames_with_face_for_blur_calc

    except Exception as e:
        print(f"Error processing video {video_path}: {e}", file=sys.stderr)
        if 'cap' in locals() and cap.isOpened():
            cap.release() # Ensure release on error

    return metrics

def calculate_audio_quality(audio_path: pathlib.Path, vad_model_local, vad_utils_local, device) -> dict: # Added device argument
    """Calculates audio quality metrics (speech ratio using Silero VAD)."""
    metrics = {'speech_ratio': 0.0}
    if not VAD_MODEL_AVAILABLE or vad_model_local is None: # Check if VAD model is usable
         # print("Skipping audio quality check: VAD model not loaded.", file=sys.stderr) # Reduce verbosity
         return metrics

    try:
        # Read audio using soundfile (more direct for WAV) ensure it's float32
        wav, sr = sf.read(str(audio_path), dtype='float32')
        if sr != VAD_SAMPLE_RATE:
            print(f"Warning: Audio SR mismatch for VAD {audio_path}. Expected {VAD_SAMPLE_RATE}, got {sr}. Skipping VAD.", file=sys.stderr)
            return metrics

        # Convert to torch tensor AND move to the correct device
        audio_tensor = torch.from_numpy(wav).to(device) # Move tensor to device

        # Get speech timestamps
        get_speech_timestamps_func = vad_utils_local[0] # Extract function from tuple
        # Ensure model is also on the correct device (should be already, but good practice)
        speech_timestamps = get_speech_timestamps_func(audio_tensor, vad_model_local.to(device), sampling_rate=VAD_SAMPLE_RATE)

        # Calculate total speech duration
        speech_duration_samples = sum([ts['end'] - ts['start'] for ts in speech_timestamps])
        total_duration_samples = len(wav) # Use original wav length

        if total_duration_samples > 0:
            metrics['speech_ratio'] = speech_duration_samples / total_duration_samples

    except Exception as e:
        print(f"Error processing audio for VAD {audio_path}: {e}", file=sys.stderr)
    return metrics

if __name__ == "__main__":
    project_root = pathlib.Path(__file__).resolve().parent.parent
    processed_dir_abs = (project_root / PROCESSED_DATA_DIR_REL).resolve()
    output_quality_path = (project_root / QUALITY_SCORES_PATH_REL).resolve()

    print(f"Project root: {project_root}")
    print(f"Processing directory: {processed_dir_abs}")

    # --- Find Files ---
    processed_file_pairs = find_processed_files(processed_dir_abs)
    if not processed_file_pairs:
        print("No processed videos found to assess quality. Exiting.")
        sys.exit(0)

    # --- Initialize Detector ---
    print(f"\nInitializing MediaPipe Face Detection...")
    mp_face_detection = mp.solutions.face_detection
    # VAD model is loaded globally above
    with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=MIN_DETECTION_CONFIDENCE) as face_detector:

        # --- Process Files ---
        quality_results = []
        print(f"\nCalculating quality metrics for {len(processed_file_pairs)} file pairs...")

        for video_path, audio_path in tqdm(processed_file_pairs, desc="Assessing Quality"):
            # Calculate metrics
            video_metrics = calculate_video_quality(video_path, face_detector)
            # Pass the determined vad_device to the audio function
            audio_metrics = calculate_audio_quality(audio_path, vad_model, vad_utils, vad_device)

            # Combine results
            relative_video_path = str(video_path.relative_to(project_root))
            quality_results.append({
                'filepath_video': relative_video_path,
                'filepath_audio': str(audio_path.relative_to(project_root)),
                **video_metrics, # Unpack video metrics dict
                **audio_metrics  # Unpack audio metrics dict
            })

        # --- Summarize Results ---
        print("\nFinished video quality assessment.")
        if quality_results: # Check if list is not empty before creating DataFrame
            quality_df = pd.DataFrame(quality_results)
            print("\nQuality Scores Summary:")
            print(quality_df.describe())
            # --- Save results (implement saving later) ---
            output_quality_path.parent.mkdir(parents=True, exist_ok=True)
            quality_df.to_csv(output_quality_path, index=False)
            print(f"\nQuality scores saved to: {output_quality_path}")
        else:
            print("\nNo quality results generated.")

    print("\nScript finished.")


     
