# ML Audiovisual Data Pipeline

This project demonstrates the design, implementation, and evaluation of an end-to-end pipeline for curating high-quality audio-visual (AV) data, specifically targeting "talking head" videos suitable for downstream Machine Learning tasks like AI avatar generation or lip-sync model training.

The pipeline focuses on automated preprocessing, quality assessment based on calculated metrics, data filtering, and quantitative evaluation of the curation process's impact using a proxy task. This project was developed to showcase skills relevant to ML Engineer - Data roles, emphasizing data-centric AI principles and MLOps practices.

## Pipeline Stages Implemented

1.  **Setup:** Project initialization using Git for version control and a Python virtual environment (`venv`) for dependency management.
2.  **Data Analysis & Subsetting:** Scripts to analyze raw data distribution (e.g., clips per speaker) and create specific data subsets defined by manifest files (CSV).
3.  **Preprocessing:** Standardization of AV data:
    * **Video:** Resizing frames to a consistent resolution (224x224), converting to a standard frame rate (25 FPS).
    * **Audio:** Resampling to a consistent sample rate (16kHz), converting to mono channel.
4.  **Quality Assessment:** Automated calculation of quality metrics for each processed file:
    * **Video Metrics (using MediaPipe):**
        * `face_presence_ratio`: Percentage of frames with a confidently detected face.
        * `avg_blur_score`: Average Laplacian variance of the detected face region (higher is sharper).
    * **Audio Metrics (using Silero VAD):**
        * `speech_ratio`: Percentage of audio duration containing speech.
5.  **Filtering:** Application of configurable thresholds to the calculated quality metrics to filter out low-quality samples and generate a manifest of high-quality data.
6.  **Evaluation Framework:** Implementation of a proxy task (Speaker Identification using mean MFCC features + SVC classifier) to quantitatively measure the performance difference between models trained on an initial subset versus the curated, high-quality subset.

## Technology Stack

* **Language:** Python 3.9+
* **Core Libraries:** pathlib, pandas, numpy
* **AV Processing:** opencv-python, librosa, soundfile, mediapipe
* **ML/Evaluation:** scikit-learn, torch/torchaudio (for Silero VAD via torch.hub)
* **Utilities:** tqdm
* **Version Control:** Git

## Attempt 1: Random Subset & Diagnosis

### Process:

* An initial **random subset of 100 samples** was selected from a downloaded part of the VoxCeleb2 dataset (`scripts/1_create_subset_manifest.py`).
* This subset was processed through the standardization pipeline (`scripts/2_preprocess_subset.py`).
* Quality metrics were calculated, and filtering was applied (`scripts/3_assess_quality_subset.py`).
* The evaluation framework (Speaker ID) was run on both the original 100 samples and the filtered subset (`scripts/4_evaluate_data_quality_impact.py`).

### Results & Diagnosis:

* The pipeline stages executed, but the Speaker ID evaluation yielded **0.00% accuracy and F1-score** for both datasets.
* **Diagnosis:** Analysis revealed an extreme class imbalance specific to this random subset – **90 unique speakers within only 100 samples**, with most speakers having only 1 sample. This prevented the speaker ID model from learning effectively and caused issues with stratified train/test splitting.

### Learning:

* This attempt highlighted the critical importance of **data sampling strategy** and ensuring the data subset structure is suitable for the chosen downstream evaluation task. A robust pipeline includes diagnostics to identify such issues.

## Attempt 2: Targeted Sampling & Validation

### Process:

* **Targeted Subsetting:** The available raw data was analyzed to identify speakers with many clips (`scripts/0_analyze_raw_data_speakers.py`). The **top 20 speakers** were selected, and a new manifest (`data/development_subset_attempt2.csv`) was created containing all their clips (**10,000 samples** total, 500 per speaker).
* **Preprocessing:** Script 2 was run on this new 10k manifest, saving standardized outputs to `data/processed/attempt2_standardized_subset/`.
* **Quality Assessment & Filtering:** Script 3 was run using the Attempt 2 standardized data and the following thresholds: `face_presence >= 0.99`, `avg_blur_score >= 50`, `speech_ratio >= 0.9`. This resulted in a filtered high-quality manifest (`data/processed/attempt2_high_quality_manifest.csv`) containing **6,408 samples** across the 20 speakers.
* **Evaluation:** Script 4 was run again, comparing the Speaker ID model (MFCCs + SVC) performance on the full 10k targeted subset vs. the 6.4k filtered high-quality subset.

### Results:

| Dataset         | Samples | Speakers | Accuracy | F1 Weighted |
| :-------------- | ------: | -------: | -------: | ----------: |
| Original (A2)   | 10,000  |       20 |   0.8584 |      0.8577 |
| High-Quality (A2)|  6,408  |       20 |   0.8714 |      0.8728 |
| **Improvement** |         |          | **+0.0130** | **+0.0151** |

### Conclusion (Attempt 2):

* The targeted sampling strategy created a dataset suitable for the speaker ID task, resulting in significant non-zero accuracy.
* The evaluation clearly demonstrated that **applying the automated quality filtering pipeline improved model performance** (+1.3% Acc, +1.5% F1) despite reducing the dataset size by ~36%.
* This successfully validates the effectiveness of the implemented data curation pipeline.

## Overall Conclusion

This project successfully implemented an end-to-end pipeline for processing, assessing, and filtering audio-visual data based on configurable quality metrics. Through two attempts, it demonstrated not only the pipeline's functionality but also the ability to diagnose data-related issues (Attempt 1) and quantitatively validate the positive impact of data curation on a downstream ML task (Attempt 2).

## Future Work & Potential Extensions

* Implement more sophisticated quality metrics (e.g., audio SNR, face landmark stability, image brightness/contrast).
* Explore different feature extraction methods (e.g., pre-trained embeddings like VGGish, ResNet features).
* Experiment with different proxy evaluation tasks.
* Integrate robust Data Version Control (e.g., DVC) to track dataset versions formally.
* Scale the pipeline using distributed computing (Spark, Ray, Dask) and workflow orchestration (Airflow, Prefect).
* Incorporate Human-in-the-Loop (HITL) validation for quality metrics or threshold tuning.
* Perform visual/auditory inspection of edge cases identified by automated metrics.

## Repository Structure
├── data/
│   ├── raw/                    # Raw data (requires manual download, .gitignored)
│   │   └── dev/mp4/            # Example structure for unzipped VoxCeleb part
│   ├── processed/              # Processed data outputs
│   │   ├── standardized_subset/       # Standardized files (Attempt 1, .gitignored)
│   │   ├── attempt2_standardized_subset/ # Standardized files (Attempt 2, .gitignored)
│   │   ├── quality_scores.csv          # Quality metrics (Attempt 1)
│   │   ├── high_quality_manifest.csv   # Filtered manifest (Attempt 1)
│   │   ├── attempt2_quality_scores.csv # Quality metrics (Attempt 2)
│   │   └── attempt2_high_quality_manifest.csv # Filtered manifest (Attempt 2)
│   ├── development_subset_manifest.csv       # Initial random subset manifest (Attempt 1)
│   └── development_subset_attempt2.csv     # Targeted subset manifest (Attempt 2)
├── notebooks/                  # Jupyter notebooks for exploration and analysis
│   └── 01_initial_data_exploration.ipynb # Example notebook
├── scripts/                    # Python scripts for pipeline stages
│   ├── 0_analyze_raw_data_speakers.py # Speaker analysis & Attempt 2 manifest creation
│   ├── 1_create_subset_manifest.py    # Attempt 1 random manifest creation
│   ├── 2_preprocess_subset.py         # Standardization (needs config change per attempt)
│   ├── 3_assess_quality_subset.py     # Quality assessment & filtering (needs config change per attempt)
│   └── 4_evaluate_data_quality_impact.py # Evaluation framework (needs config change per attempt)
├── .gitignore
├── README.md                   # This file
└── requirements.txt            # Python dependencies

## How to Run (Attempt 2)
1.  **Clone:** `git clone <repository_url>`
2.  **Obtain Data:** Download at least one part (e.g., `dev aa`) of the **VoxCeleb2 video (`.mp4`) dataset** through the official request process ([Oxford VGGFace Website](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html)). Unzip it and place the contents such that you have the structure `data/raw/dev/mp4/id*/*/*.mp4` within the project directory. *(Note: Raw data is required but not included in the repo)*.
3.  **Setup Environment:**
    ```bash
    cd ml-audiovisual-datapipe
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    # Optional: Install libsndfile for robust audio backend (macOS example)
    # brew install libsndfile
    # pip uninstall soundfile librosa && pip install librosa # Reinstall after libsndfile
    ```
4.  **Run Analysis & Create Manifest (Attempt 2):**
    ```bash
    python scripts/0_analyze_raw_data_speakers.py
    ```
    *(This generates `data/development_subset_attempt2.csv`)*
5.  **Run Preprocessing (Attempt 2):**
    * *Ensure configuration in `scripts/2_preprocess_subset.py` points to `development_subset_attempt2.csv` and `attempt2_standardized_subset`.*
    ```bash
    python scripts/2_preprocess_subset.py
    ```
    *(This can take a long time depending on subset size)*
6.  **Run Quality Assessment & Filtering (Attempt 2):**
    * *Ensure configuration in `scripts/3_assess_quality_subset.py` points to `attempt2_standardized_subset`, `attempt2_quality_scores.csv`, and `attempt2_high_quality_manifest.csv`. Verify filter thresholds.*
    ```bash
    python scripts/3_assess_quality_subset.py
    ```
    *(This generates `data/processed/attempt2_high_quality_manifest.csv`)*
7.  **Run Evaluation (Attempt 2):**
    * *Ensure configuration in `scripts/4_evaluate_data_quality_impact.py` points to `development_subset_attempt2.csv` and `attempt2_high_quality_manifest.csv`.*
    ```bash
    python scripts/4_evaluate_data_quality_impact.py
    ```