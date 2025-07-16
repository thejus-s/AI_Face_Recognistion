import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm

# Load embeddings and labels
embeddings = np.load("test_embedding.npy")
labels = np.load("test_labels.npy")

# Compute all possible pairs
pairs = list(itertools.combinations(range(len(embeddings)), 2))
true_labels = []
similarities = []

print(f"\n Generating similarity scores for {len(pairs)} pairs...\n")
for i, j in tqdm(pairs, desc="Computing similarities"):
    emb1 = embeddings[i].reshape(1, -1)
    emb2 = embeddings[j].reshape(1, -1)
    sim = cosine_similarity(emb1, emb2)[0][0]
    similarities.append(sim)
    true_labels.append(int(labels[i] == labels[j]))

# ROC Curve
fpr, tpr, thresholds = roc_curve(true_labels, similarities)
roc_auc = auc(fpr, tpr)

precision, recall, thresholds_pr = precision_recall_curve(true_labels, similarities)
avg_precision = average_precision_score(true_labels, similarities)

plt.figure(figsize=(6, 6))
plt.plot(recall, precision, color='purple', lw=2, label=f'PR Curve (AP = {avg_precision:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Post-Tuning Precision-Recall Curve')
plt.legend(loc="lower left")
plt.grid(True)
plt.tight_layout()
plt.savefig("post_tuning_precision_recall_curve.png")
plt.show()

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='green', lw=2, label=f'Post-Tuning ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Post-Tuning ROC Curve')
plt.legend(loc="lower right")
plt.grid()
plt.tight_layout()
plt.savefig("post_tuning_roc_curve.png")
plt.show()

# Similarity Distributions
same = [sim for sim, label in zip(similarities, true_labels) if label == 1]
diff = [sim for sim, label in zip(similarities, true_labels) if label == 0]

plt.figure(figsize=(6, 4))
plt.hist(same, bins=50, alpha=0.6, label='Same Person', color='blue')
plt.hist(diff, bins=50, alpha=0.6, label='Different Person', color='orange')
plt.xlabel("Cosine Similarity")
plt.ylabel("Frequency")
plt.title("Post-Tuning Similarity Distribution")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("post_tuning_similarity_distribution.png")
plt.show()
