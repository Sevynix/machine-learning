import pandas as pd
from scipy.stats import chi2_contingency

df = pd.read_csv("tiktok_cleaned.csv")
print("Ukuran dataset:", df.shape)
print(df.head())

target = "claim_status"

fitur_kategorik = ["verified_status", "author_ban_status"]

alpha = 0.05
hasil = []

for fitur in fitur_kategorik:
    contingency_table = pd.crosstab(df[fitur], df[target])

    chi2, p, dof, expected = chi2_contingency(contingency_table)

    signifikan = "Ya" if p < alpha else "Tidak"

    print(f"\nFitur: {fitur}")
    print(f"Chi-Square Value: {chi2}")
    print(f"P-Value: {p}")
    print(f"Degree of Freedom: {dof}")
    print(f"Signifikan terhadap {target}: {signifikan}")

    hasil.append({
        "fitur": fitur,
        "target": target,
        "chi2_value": chi2,
        "p_value": p,
        "degree_of_freedom": dof,
        "signifikan (p<0.05)": signifikan
    })

df_hasil = pd.DataFrame(hasil)
df_hasil.to_csv("hasil_chi_square.csv", index=False)
print("\nHasil seleksi fitur Chi-Square disimpan ke hasil_chi_square.csv")
print(df_hasil)
