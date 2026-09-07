import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Load data hasil cleaning (BUKAN yang sudah di-MinMax, PCA butuh StandardScaler sendiri)
df = pd.read_csv("tiktok_cleaned.csv")

kolom_numerik = [
    "video_duration_sec",
    "video_view_count",
    "video_like_count",
    "video_share_count",
    "video_download_count",
    "video_comment_count",
]

X = df[kolom_numerik]

# 2. Standarisasi (PCA sensitif terhadap skala)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================================================
# A. PCA dengan jumlah komponen BEBAS (di sini dipilih 3)
# =========================================================
n_bebas = 3
pca_bebas = PCA(n_components=n_bebas)
X_pca_bebas = pca_bebas.fit_transform(X_scaled)

df_pca_bebas = pd.DataFrame(
    X_pca_bebas, columns=[f"PC{i+1}" for i in range(n_bebas)]
)

print(f"PCA Bebas (n_components = {n_bebas})")
print("Explained Variance Ratio:", pca_bebas.explained_variance_ratio_)
print("Total Variance Explained:", sum(pca_bebas.explained_variance_ratio_))

df_pca_bebas.to_csv("hasil_pca_bebas.csv", index=False)

# =========================================================
# B. PCA dengan jumlah komponen OPTIMAL
#    (dipilih n minimum yang mencapai >= 95% varians kumulatif)
# =========================================================
pca_full = PCA().fit(X_scaled)
cum_var = np.cumsum(pca_full.explained_variance_ratio_)

threshold = 0.95
n_optimal = int(np.argmax(cum_var >= threshold) + 1)

print("\nMencari n_components optimal")
for i, v in enumerate(cum_var, start=1):
    print(f"n={i} -> Cumulative Variance: {v:.4f}")
print(f"n_components optimal (>= {threshold*100:.0f}% variance): {n_optimal}")

pca_optimal = PCA(n_components=n_optimal)
X_pca_optimal = pca_optimal.fit_transform(X_scaled)

df_pca_optimal = pd.DataFrame(
    X_pca_optimal, columns=[f"PC{i+1}" for i in range(n_optimal)]
)

df_pca_optimal.to_csv("hasil_pca_optimal.csv", index=False)

# =========================================================
# C. Visualisasi untuk screenshot laporan
# =========================================================
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cum_var) + 1), cum_var, marker="o")
plt.axhline(y=threshold, color="r", linestyle="--", label=f"{threshold*100:.0f}% variance")
plt.xlabel("Jumlah Komponen")
plt.ylabel("Cumulative Explained Variance")
plt.title("Scree Plot - Menentukan n_components Optimal")
plt.legend()
plt.grid(True)
plt.savefig("scree_plot_pca.png")
plt.show()

if n_bebas >= 2:
    plt.figure(figsize=(8, 6))
    plt.scatter(df_pca_bebas["PC1"], df_pca_bebas["PC2"], alpha=0.6)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"PCA Bebas: PC1 vs PC2 ({n_bebas} komponen)")
    plt.savefig("pca_bebas_scatter.png")
    plt.show()

print("\nSelesai. File tersimpan: hasil_pca_bebas.csv, hasil_pca_optimal.csv")
print("Gambar tersimpan: scree_plot_pca.png, pca_bebas_scatter.png")
