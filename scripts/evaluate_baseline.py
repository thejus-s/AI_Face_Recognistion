import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import itertools
from tqdm import tqdm

# Load test embeddings and labels
embeddings = np.load("test_embedding.npy")
labels = np.load("test_labels.npy")

# Sanity check
print(f"Loaded embeddings: {len(embeddings)}")
print(f"Unique labels: {set(labels)}")

# Generate all possible pairs
pairs = list(itertools.combinations(range(len(embeddings)), 2))

true_labels = []
similarities = []

for i, j in tqdm(pairs, desc="Computing similarities"):
    emb1 = embeddings[i].reshape(1, -1)
    emb2 = embeddings[j].reshape(1, -1)
    
    sim = cosine_similarity(emb1, emb2)[0][0]
    similarities.append(sim)
    
    # Same person = 1, Different person = 0
    true_labels.append(1 if labels[i] == labels[j] else 0)

# ROC Curve
fpr, tpr, thresholds = roc_curve(true_labels, similarities)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})', color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Baseline ROC Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("baseline_roc_curve.png")
plt.show()

# Similarity distribution
same = [sim for sim, label in zip(similarities, true_labels) if label == 1]
diff = [sim for sim, label in zip(similarities, true_labels) if label == 0]

plt.figure(figsize=(6, 4))
plt.hist(same, bins=50, alpha=0.6, label='Same Person', color='green')
plt.hist(diff, bins=50, alpha=0.6, label='Different People', color='red')
plt.xlabel("Cosine Similarity")
plt.ylabel("Frequency")
plt.title("Similarity Score Distribution")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("baseline_similarity_distribution.png")
plt.show()
