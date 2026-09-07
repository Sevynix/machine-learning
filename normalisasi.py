import pandas as pd
from sklearn import preprocessing

df = pd.read_csv("tiktok_cleaned.csv")
print(df.shape)
print(df.head())

kolom_numerik = ["video_duration_sec", "video_view_count", "video_like_count", "video_share_count", "video_download_count", "video_comment_count"]

print("Sebelum normalisasi:")
print(df[kolom_numerik].describe())

df_normalized = df.copy()
min_max_scaler = preprocessing.MinMaxScaler()
df_normalized[kolom_numerik] = min_max_scaler.fit_transform(df[kolom_numerik])

print("Sesudah normalisasi:")
print(df_normalized[kolom_numerik].describe())
print(df_normalized[kolom_numerik].head())

df_normalized.to_csv("tiktok_normalized.csv", index=False)
print("Selesai, disimpan sebagai tiktok_normalized.csv")