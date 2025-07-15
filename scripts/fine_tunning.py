import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load train and test embeddings and labels
train_embeddings = np.load("train_embedding.npy")
train_labels = np.load("train_labels.npy")
test_embeddings = np.load("test_embedding.npy")
test_labels = np.load("test_labels.npy")

print(" Data loaded:")
print("  - Train samples:", len(train_labels))
print("  - Test samples:", len(test_labels))

# Train SVM classifier
print("\n Training SVM classifier...")
clf = SVC(kernel='linear', probability=True)
clf.fit(train_embeddings, train_labels)

joblib.dump(clf, "svm_classifier.pkl")

# Predict on test set
print("\n Predicting on test set...")
pred_labels = clf.predict(test_embeddings)

# Accuracy
acc = accuracy_score(test_labels, pred_labels)
print(f"\n Accuracy: {acc:.2f}")

# Classification report
print("\n Classification Report:")
print(classification_report(test_labels, pred_labels))

# Confusion matrix
cm = confusion_matrix(test_labels, pred_labels, labels=np.unique(train_labels))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=np.unique(train_labels), yticklabels=np.unique(train_labels), cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
