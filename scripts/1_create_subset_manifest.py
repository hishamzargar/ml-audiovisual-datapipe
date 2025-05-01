import pathlib
import random
import pandas as pd
import sys

#define root directory
RAW_DATA_ROOT = pathlib.Path("../data/raw/dev/mp4")

#path for manifest file
MANIFEST_DIR = pathlib.Path("../data")
MANIFEST_FILENAME = "development_subset_manifest.csv"
MANIFEST_PATH = MANIFEST_DIR / MANIFEST_FILENAME

#sample size
SAMPLE_SIZE = 100


def create_subset_manifest(data_root: pathlib.Path, output_path: pathlib.Path, n_samples: int):
    """
    Scans a directory for .mp4 files, randomly samples N files,
    extracts speaker/utterance IDs from the path, and saves to a CSV manifest.
    """
    print(f"Scanning for .mp4 files under: {data_root.resolve()}")

    if not data_root.is_dir():
        print(f"Error: Raw data directory not found at {data_root.resolve()}", file=sys.stderr)
        sys.exit(1)

    # Recursively find all .mp4 files
    all_mp4_files = list(data_root.rglob('*.mp4'))
    n_found = len(all_mp4_files)
    print(f"Found {n_found} .mp4 files.")

    if n_found == 0:
        print("Error: No .mp4 files found. Please check the RAW_DATA_ROOT path.", file=sys.stderr)
        sys.exit(1)
    
    # Adjust sample size if fewer files are found than requested
    actual_sample_size = min(n_samples, n_found)
    if actual_sample_size < n_samples:
        print(f"Warning: Requested {n_samples} samples, but only found {n_found}. Using all found files.")

    # Randomly sample the file paths
    print(f"Randomly sampling {actual_sample_size} files...")
    sampled_paths = random.sample(all_mp4_files, actual_sample_size)

    mainfest_data = []
    print("Extracting information and building manifest...")
    for file_path in sampled_paths:
        try:
            # Assumes path structure like .../mp4/id<speaker_id>/<utterance_id>/<filename>.mp4
            speaker_id = file_path.parent.parent.name
            utterance_id = file_path.parent.name
            # Store relative path from project root for better portability
            relative_path = file_path.relative_to(pathlib.Path.cwd())
            mainfest_data.append({
                "filepath": str(relative_path),
                "speaker_id": speaker_id,
                "utterance_id": utterance_id
            })
        except IndexError:
            print(f"Warning: Could not parse speaker/utterance ID for path: {file_path}. Skipping.", file=sys.stderr)
            continue # Skip files not matching the expected structure
    if not mainfest_data:
        print("Error: No valid file paths could be processed. Manifest not created.", file=sys.stderr)
        sys.exit(1)
    
    # Create a pandas DataFrame
    manifest_df = pd.DataFrame(mainfest_data)

    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the DataFrame to CSV
    try:
        manifest_df.to_csv(output_path, index=False)
        print("-" * 30)
        print(f"Successfully created manifest file with {len(manifest_df)} entries.")
        print(f"Manifest saved to: {output_path.resolve()}")
        print("-" * 30)
        print("Sample entries:")
        print(manifest_df.head())
        print("-" * 30)
    except Exception as e:
        print(f"Error saving manifest file to {output_path}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    # Assuming the script is run from the project root directory (ml-audiovisual-datapipe/)
    # Adjust RAW_DATA_ROOT if running from scripts/ directory itself
    project_root = pathlib.Path.cwd()
    raw_data_full_path = project_root / "data" / "raw" / "dev" / "mp4"
    manifest_full_path = project_root / "data" / MANIFEST_FILENAME

    create_subset_manifest(raw_data_full_path, manifest_full_path, SAMPLE_SIZE)
