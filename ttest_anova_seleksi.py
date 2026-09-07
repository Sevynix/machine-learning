import pandas as pd
from scipy.stats import ttest_ind, f_oneway

df = pd.read_csv("tiktok_cleaned.csv")
print("Ukuran dataset:", df.shape)
print(df.head())

kolom_numerik = [
    "video_duration_sec", "video_view_count", "video_like_count",
    "video_share_count", "video_download_count", "video_comment_count"
]

alpha = 0.05
hasil = []

target_ttest = "claim_status"
grup_ttest = df[target_ttest].dropna().unique()

if len(grup_ttest) == 2:
    grup_a = df[df[target_ttest] == grup_ttest[0]]
    grup_b = df[df[target_ttest] == grup_ttest[1]]

    for kolom in kolom_numerik:
        t_stat, p_val = ttest_ind(
            grup_a[kolom].dropna(),
            grup_b[kolom].dropna(),
            equal_var=False
        )
        signifikan = "Ya" if p_val < alpha else "Tidak"

        print(f"\n[T-TEST] {kolom} vs {target_ttest}")
        print(f"T-Statistic: {t_stat}")
        print(f"P-Value: {p_val}")
        print(f"Signifikan: {signifikan}")

        hasil.append({
            "fitur": kolom,
            "target": target_ttest,
            "jenis_uji": "T-Test",
            "statistic": t_stat,
            "p_value": p_val,
            "signifikan (p<0.05)": signifikan
        })

target_anova = "author_ban_status"
grup_anova = df[target_anova].dropna().unique()

for kolom in kolom_numerik:
    grup_list = [df[df[target_anova] == g][kolom].dropna() for g in grup_anova]
    f_stat, p_val = f_oneway(*grup_list)
    signifikan = "Ya" if p_val < alpha else "Tidak"

    print(f"\n[ANOVA] {kolom} vs {target_anova}")
    print(f"F-Statistic: {f_stat}")
    print(f"P-Value: {p_val}")
    print(f"Signifikan: {signifikan}")

    hasil.append({
        "fitur": kolom,
        "target": target_anova,
        "jenis_uji": "ANOVA",
        "statistic": f_stat,
        "p_value": p_val,
        "signifikan (p<0.05)": signifikan
    })

df_hasil = pd.DataFrame(hasil)
df_hasil.to_csv("hasil_ttest_anova.csv", index=False)
print("\nHasil seleksi fitur T-Test & ANOVA disimpan ke hasil_ttest_anova.csv")
print(df_hasil)
