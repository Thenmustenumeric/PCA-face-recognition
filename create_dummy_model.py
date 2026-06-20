import os
import pickle
import numpy as np
from sklearn.decomposition import PCA

MODEL_DIR = "models"
PCA_MODEL_PATH = os.path.join(MODEL_DIR, "pca_model.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

print("Membuat data dummy...")

X = np.random.rand(100, 10000)

print("Melatih PCA...")

pca = PCA(n_components=50)
pca.fit(X)

with open(PCA_MODEL_PATH, "wb") as f:
    pickle.dump(pca, f)

print("Model berhasil disimpan:")
print(PCA_MODEL_PATH)