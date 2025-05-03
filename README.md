# ML Audiovisual Data Pipeline Project

This project focuses on building a pipeline for curating and evaluating high-quality talking head video data, suitable for tasks like training AI avatars, lip-sync models, or other audiovisual machine learning applications.

## Pipeline Stages Implemented:

1.  **Setup:** Project initialization with Git and Python virtual environment (`venv`).
2.  **Data Subsetting:** Creation of an initial working subset from a larger dataset.
3.  **Preprocessing:** Standardization of video (resolution, FPS) and audio (sample rate, mono channel).
4.  **Quality Assessment:** Calculation of automated metrics for video (face presence, face blurriness) and audio (speech ratio via VAD).
5.  **Filtering:** Application of configurable thresholds to quality metrics to select high-quality samples.
6.  **Evaluation Framework:** Implementation of a proxy task (Speaker Identification using MFCCs + SVC) to quantitatively assess the impact of data curation.

## Attempt 1: Random Subset Evaluation

### Process:

* A random subset of **100 samples** was initially selected from a portion of the VoxCeleb2 dataset (`scripts/1_create_subset_manifest.py`).
* These samples were preprocessed to 224x224 resolution, 25 FPS, 16kHz mono audio (`scripts/2_preprocess_subset.py`).
* Quality metrics (face presence, blurriness via Laplacian variance, speech ratio via Silero VAD) were calculated (`scripts/3_assess_quality_subset.py`).
* Filtering thresholds were applied (`face_presence >= 0.99`, `avg_blur_score >= 50`, `speech_ratio >= 0.9`), resulting in a high-quality subset of **57 samples** (`data/processed/high_quality_manifest.csv`).
* A Speaker Identification model (SVC on mean MFCCs) was trained and evaluated on both the original 100-sample subset and the filtered 57-sample subset (`scripts/4_evaluate_data_quality_impact.py`).

### Results & Diagnosis:

* The preprocessing and quality assessment pipelines executed successfully.
* The Speaker ID evaluation yielded **0.00 accuracy and F1-score** for both the original and high-quality subsets.
* **Diagnosis:** Diagnostic prints revealed that both subsets contained a very high number of unique speakers relative to the sample size (e.g., 90 speakers in 100 samples, 54 speakers in 57 samples), with most speakers having only 1 sample (avg ~1.1 samples/speaker). This extreme lack of samples per class prevented the classifier from learning meaningful patterns for speaker identification, even after data quality filtering. Stratification during train/test split also failed due to this issue.

### Learnings & Next Steps:

Attempt 1 successfully demonstrated the construction of the data processing and quality assessment pipeline. However, it highlighted that the initial *random sampling strategy* was insufficient for the chosen *speaker identification proxy task*.

The next attempt will involve modifying the data subset selection strategy to ensure sufficient samples per speaker before evaluating the impact of quality filtering.

## How to Run (Attempt 1):

1.  Setup environment: `python3 -m venv venv`, `source venv/bin/activate`, `pip install -r requirements.txt`
2.  (If raw data selected) Run `python scripts/1_create_subset_manifest.py`
3.  Run `python scripts/2_preprocess_subset.py`
4.  Run `python scripts/3_assess_quality_subset.py` (Adjust thresholds inside if desired)
5.  Run `python scripts/4_evaluate_data_quality_impact.py`
