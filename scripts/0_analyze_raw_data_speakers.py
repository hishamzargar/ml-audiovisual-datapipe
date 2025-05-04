import pathlib
from collections import Counter
import pandas as pd
from tqdm.auto import tqdm # Add tqdm for progress
import sys

# --- Configuration ---
# Directory containing the unzipped raw mp4 files
RAW_DATA_ROOT_REL = "data/raw/dev/mp4" # Relative to project root

# --- Parameters for Selecting Subset ---
NUM_TOP_SPEAKERS = 20 # How many speakers with the most clips to select
MIN_CLIPS_PER_SPEAKER_FOR_CONSIDERATION = 10 # Initial filter before selecting top N
OUTPUT_MANIFEST_REL = "data/development_subset_attempt2.csv" # Filename for the new manifest
# --- End Configuration ---


def create_targeted_manifest(data_root_abs: pathlib.Path,
                             output_manifest_abs: pathlib.Path,
                             num_speakers: int,
                             min_clips_initial: int,
                             project_root: pathlib.Path):
    """
    Scans raw data, identifies top N speakers with >= min_clips_initial,
    and creates a manifest CSV containing all clips for only those selected speakers.
    """
    print(f"Scanning for .mp4 files under: {data_root_abs}")
    if not data_root_abs.is_dir():
        print(f"Error: Raw data directory not found at {data_root_abs}", file=sys.stderr)
        sys.exit(1)

    all_mp4_files = list(data_root_abs.rglob('*.mp4'))
    n_found = len(all_mp4_files)
    print(f"Found {n_found} total .mp4 files.")
    if n_found == 0: sys.exit("Error: No .mp4 files found.")

    # --- Count clips per speaker ---
    speaker_counts = Counter()
    print("Counting clips per speaker...")
    for file_path in tqdm(all_mp4_files, desc="Counting Clips"):
        try:
            speaker_id = file_path.parent.parent.name
            if speaker_id.startswith("id"): speaker_counts[speaker_id] += 1
        except IndexError: continue # Ignore paths not matching structure

    print(f"\nFound {len(speaker_counts)} unique speaker IDs.")

    # --- Select Top N Speakers ---
    qualifying_speakers = {
        speaker: count
        for speaker, count in speaker_counts.items()
        if count >= min_clips_initial
    }
    num_qualifying = len(qualifying_speakers)
    print(f"Found {num_qualifying} speakers with >= {min_clips_initial} clips.")

    if num_qualifying == 0:
        sys.exit(f"Error: No speakers found with at least {min_clips_initial} clips.")

    # Sort by count descending and select top N
    # Use most_common() method of Counter
    top_speakers_list = [speaker for speaker, count in speaker_counts.most_common(num_speakers)]

    if len(top_speakers_list) < num_speakers:
         print(f"Warning: Only found {len(top_speakers_list)} speakers meeting criteria, using all of them.", file=sys.stderr)
         num_speakers = len(top_speakers_list) # Adjust number if fewer found

    print(f"\nSelected top {num_speakers} speakers:")
    selected_speakers_set = set(top_speakers_list) # Use set for faster lookup
    # Print selected speakers and their counts (optional)
    for speaker_id in top_speakers_list:
         print(f"  - {speaker_id}: {speaker_counts[speaker_id]} clips")


    # --- Create Manifest for Selected Speakers ---
    manifest_data = []
    print(f"\nGathering file paths for selected {num_speakers} speakers...")
    for file_path in tqdm(all_mp4_files, desc="Creating Manifest"):
        try:
            speaker_id = file_path.parent.parent.name
            # Check if this speaker is one of the selected ones
            if speaker_id in selected_speakers_set:
                utterance_id = file_path.parent.name
                relative_path = file_path.relative_to(project_root) # Path relative to project root
                manifest_data.append({
                    'filepath': str(relative_path),
                    'speaker_id': speaker_id,
                    'utterance_id': utterance_id
                })
        except IndexError: continue # Ignore paths not matching structure

    if not manifest_data:
         sys.exit("Error: Failed to collect any file paths for the selected speakers.")

    manifest_df = pd.DataFrame(manifest_data)
    print(f"\nCreated manifest with {len(manifest_df)} total entries for {num_speakers} speakers.")

    # --- Save Manifest ---
    try:
        output_manifest_abs.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        manifest_df.to_csv(output_manifest_abs, index=False)
        print(f"\nNew manifest for Attempt 2 saved to: {output_manifest_abs}")
    except Exception as e:
        print(f"\nError saving manifest file: {e}", file=sys.stderr)


if __name__ == "__main__":
    project_root = pathlib.Path(__file__).resolve().parent.parent
    raw_data_abs = (project_root / RAW_DATA_ROOT_REL).resolve()
    output_manifest_abs = (project_root / OUTPUT_MANIFEST_REL).resolve()

    create_targeted_manifest(
        data_root_abs=raw_data_abs,
        output_manifest_abs=output_manifest_abs,
        num_speakers=NUM_TOP_SPEAKERS,
        min_clips_initial=MIN_CLIPS_PER_SPEAKER_FOR_CONSIDERATION,
        project_root=project_root
    )
    print("\nScript finished.")