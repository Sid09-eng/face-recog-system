import os
import cv2 # type: ignore
import numpy as np # type: ignore
from sklearn.decomposition import PCA # type: ignore
from sklearn.neighbors import KNeighborsClassifier # type: ignore
from sklearn.preprocessing import LabelEncoder # type: ignore
from joblib import dump # type: ignore

DATASET_DIR = 'dataset'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

def load_images(dataset_dir):
    X, y = [], []
    for person in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, person)
        if not os.path.isdir(person_dir): continue
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            img = cv2.resize(img, (100, 100))
            X.append(img.flatten())
            y.append(person)
    return np.array(X), np.array(y)

print("Loading images...")
X, y = load_images(DATASET_DIR)
print(f"Loaded {len(X)} images.")

print("Encoding labels...")
le = LabelEncoder()
y_enc = le.fit_transform(y)

print("Applying PCA...")
pca = PCA(n_components=100, whiten=True, random_state=42)
X_pca = pca.fit_transform(X)

print("Training classifier...")
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_pca, y_enc)

print("Saving model...")
dump(pca, os.path.join(MODEL_DIR, 'pca_model.joblib'))
dump(knn, os.path.join(MODEL_DIR, 'knn_classifier.joblib'))
dump(le, os.path.join(MODEL_DIR, 'label_encoder.joblib'))
print("Training complete.")
