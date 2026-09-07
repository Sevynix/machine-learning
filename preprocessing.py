import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("tiktok_dataset.csv")
print(df.shape)
print(df.head())

print(df.isnull().sum())
print()
print((df.isnull().sum() / len(df) * 100).round(2))

kolom_missing = df.columns[df.isnull().any()].tolist()
print("Kolom yang missing:", kolom_missing)

mask = df[kolom_missing].isnull()
baris_missing_semua = (mask.sum(axis=1) == len(kolom_missing)).sum()
print("Baris yang missing di SEMUA kolom sekaligus:", baris_missing_semua)

before = df.shape[0]
df_clean = df.dropna()
after = df_clean.shape[0]

print("Sebelum drop:", before)
print("Sesudah drop:", after)
print("Baris dihapus:", before - after)
print("Missing tersisa:", df_clean.isnull().sum().sum())

kolom_numerik = ["video_duration_sec", "video_view_count", "video_like_count",
                  "video_share_count", "video_download_count", "video_comment_count"]

z_scores = np.abs(stats.zscore(df_clean[kolom_numerik]))
outliers_z = df_clean[(z_scores > 3).any(axis=1)]
print("Jumlah outlier (Z-score):", len(outliers_z))

Q1 = df_clean[kolom_numerik].quantile(0.25)
Q3 = df_clean[kolom_numerik].quantile(0.75)
IQR = Q3 - Q1
batas_bawah = Q1 - 1.5 * IQR
batas_atas = Q3 + 1.5 * IQR

outliers_iqr = df_clean[((df_clean[kolom_numerik] < batas_bawah) | (df_clean[kolom_numerik] > batas_atas)).any(axis=1)]
print("Jumlah outlier (IQR):", len(outliers_iqr))

plt.figure(figsize=(6,4))
sns.boxplot(y=df_clean["video_view_count"])
plt.title("Boxplot video_view_count")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(y=df_clean["video_comment_count"])
plt.title("Boxplot video_comment_count")
plt.show()

df_drop_outlier = df_clean[(z_scores <= 3).all(axis=1)]
print("Ukuran sebelum:", df_clean.shape)
print("Ukuran sesudah hapus outlier:", df_drop_outlier.shape)

df_median = df_clean.copy()
for col in kolom_numerik:
    col_z = np.abs(stats.zscore(df_median[col]))
    df_median[col] = np.where(col_z > 3, df_median[col].median(), df_median[col])
print("\nContoh hasil ganti median (video_comment_count):")
print(df_median["video_comment_count"].describe())

df_capped = df_clean.copy()
for col in kolom_numerik:
    df_capped[col] = np.where(df_capped[col] < batas_bawah[col], batas_bawah[col], df_capped[col])
    df_capped[col] = np.where(df_capped[col] > batas_atas[col], batas_atas[col], df_capped[col])
print("\nContoh hasil capping IQR (video_comment_count):")
print(df_capped["video_comment_count"].describe())

plt.figure(figsize=(6,4))
sns.boxplot(y=df_capped["video_comment_count"])
plt.title("Boxplot video_comment_count (setelah capping IQR)")
plt.show()

df_capped.to_csv("tiktok_cleaned.csv", index=False)
print("Dataset akhir disimpan:", df_capped.shape)