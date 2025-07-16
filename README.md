# AI_Face_Recognistion

AI Face Recognition Evaluation

This project performs face recognition using InsightFace with ONNXRuntime. It includes steps for dataset preparation, feature extraction, model training using SVM, and evaluation.

Directory Structure

├── dataset/                # (Optional) Raw dataset folder
├── models/                 # Saved models (e.g. SVM classifier)
├── scripts/                # All Python scripts
│   ├── extract_embeddings.py
│   ├── evaluate_baseline.py
│   ├── fine_tunning.py
│   ├── postfine_tunning.py
│   └── split_dataset.py
├── train/                  # Training images organized by person
├── test/                   # Testing images organized by person
├── train_embedding.npy     # Extracted embeddings for training set
├── train_labels.npy        # Corresponding labels
├── test_embedding.npy      # Extracted embeddings for testing set
├── test_labels.npy         # Corresponding labels
├── svm_classifier.pkl      # Trained SVM model
└── requirements.txt        # Project requirements

Steps

1. Install requirements

pip install -r requirements.txt

install all the requirements to run scripts

2. Extract Embeddings

python scripts/extract_embeddings.py

Saves .npy files for embeddings and labels.

3. Baseline Evaluation

python scripts/evaluate_baseline.py

Generates:

ROC Curve (baseline_roc_curve.png)

Similarity Distribution (baseline_similarity_distribution.png)

Precision-Recall Curve (baseline_precision_recall.png)

4. Fine-Tuning with SVM

python scripts/fine_tunning.py

Trains a linear SVM and outputs:

Accuracy & classification report

Confusion matrix (confusion_matrix.png)

Saved model (svm_classifier.pkl)

5. Post-Fine-Tuning Evaluation

python scripts/postfine_tunning.py

Evaluates model performance again using ROC and similarity plots:

ROC Curve (post_tuning_roc_curve.png)

Similarity Distribution (post_tuning_similarity_distribution.png)

Precision-Recall Curve (post_tuning_precision_recall.png)

Notes

CPUExecutionProvider is used by default for compatibility.

Images should be clear and face-visible for proper embedding.