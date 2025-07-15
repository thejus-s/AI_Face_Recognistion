import os
import cv2
import numpy as np
from tqdm import tqdm
from insightface.app import FaceAnalysis

# Initialize the InsightFace model
model = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
model.prepare(ctx_id=0, det_size=(320, 320))  # Optimize for performance

def extract_embedding(data_dir):
    embeddings = []
    labels = []

    print(f"\n Extracting embeddings from: {data_dir}\n")

    for person_name in tqdm(os.listdir(data_dir), desc=f"Processing {data_dir}"):
        person_path = os.path.join(data_dir, person_name)

        if not os.path.isdir(person_path):
            continue  # Skip files

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            print(f"Processing: {img_path}")

            # Read and validate image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Could not read image: {img_path}")
                continue

            try:
                # Resize image for consistency
                img = cv2.resize(img, (640, 640))

                # Detect faces
                faces = model.get(img)
                if faces:
                    face = faces[0]
                    emb = face.embedding
                    embeddings.append(emb)
                    labels.append(person_name)
                    print(f"  Face detected and embedding extracted.")
                else:
                    print(f"  No face found in: {img_path}")

            except Exception as e:
                print(f"    Error processing {img_path}: {e}")
                continue

    return np.array(embeddings), np.array(labels)

# Extract for both train and test
train_embs, train_labels = extract_embedding("train")
test_embs, test_labels = extract_embedding("test")

# Save to disk
np.save("train_embedding.npy", train_embs)
np.save("train_labels.npy", train_labels)
np.save("test_embedding.npy", test_embs)
np.save("test_labels.npy", test_labels)

print("\n Embedding extraction complete. Files saved to .npy format.")