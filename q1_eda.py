"""Soru 1 - Veriyi tanıma: sınıf dağılımı ve arızalı/arızasız karşılaştırma."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from common import *

df = load()
print("Boyut:", df.shape, "| Eksik değer:", int(df.isna().sum().sum()))
vc = df[TARGET].value_counts()
print(vc, "\nArıza oranı: %.2f%%" % (100 * df[TARGET].mean()))
print("Arıza türü sayıları:\n", df[LEAK].sum())
print("Type bazında arıza oranı:\n", df.groupby("Type")[TARGET].agg(["mean", "count"]))

# --- Şekil 1: sınıf dağılımı + arıza oranı (tool wear aralığına göre) + Type
fig, ax = plt.subplots(1, 3, figsize=(11, 3.1))
bars = ax[0].bar(["Arızasız (0)", "Arızalı (1)"], vc.values, color=["#5B8DB8", "#D1495B"])
for b, v in zip(bars, vc.values):
    ax[0].text(b.get_x() + b.get_width() / 2, v, f"{v}\n(%{100*v/len(df):.1f})", ha="center", va="bottom", fontsize=8)
ax[0].set_ylim(0, 11500); ax[0].set_title("Machine failure dağılımı", fontsize=10)

bins = pd.cut(df["Tool wear [min]"], [-1, 50, 100, 150, 200, 260])
r = df.groupby(bins, observed=True)[TARGET].mean() * 100
ax[1].bar(range(len(r)), r.values, color="#D1495B")
ax[1].set_xticks(range(len(r))); ax[1].set_xticklabels(["0-50", "51-100", "101-150", "151-200", "201+"], fontsize=8)
ax[1].set_title("Takım aşınmasına göre arıza oranı (%)", fontsize=10)
t = df.groupby("Type")[TARGET].mean() * 100
ax[2].bar(t.index, t.values, color="#5B8DB8")
ax[2].set_title("Ürün tipine göre arıza oranı (%)", fontsize=10)
plt.tight_layout(); plt.savefig(FIG / "q1_dagilim.png", dpi=170); plt.close()

# --- Şekil 2: 5 sayısal değişken, arızalı vs arızasız boxplot
fig, ax = plt.subplots(1, 5, figsize=(11, 2.9))
for a, c in zip(ax, NUM):
    data = [df.loc[df[TARGET] == k, c] for k in (0, 1)]
    bp = a.boxplot(data, patch_artist=True, widths=.6, flierprops=dict(markersize=2))
    for p, col in zip(bp["boxes"], ["#5B8DB8", "#D1495B"]):
        p.set_facecolor(col)
    a.set_xticklabels(["Arızasız", "Arızalı"], fontsize=8)
    a.set_title(SHORT[c], fontsize=9)
plt.tight_layout(); plt.savefig(FIG / "q1_boxplot.png", dpi=170); plt.close()

# --- Tablo: ortalama, medyan, Cohen d, Mann-Whitney p
rows = []
for c in NUM:
    a, b = df.loc[df[TARGET] == 0, c], df.loc[df[TARGET] == 1, c]
    sp = (((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2)) ** .5
    rows.append({"Değişken": SHORT[c], "Arızasız ort.": a.mean(), "Arızalı ort.": b.mean(),
                 "Arızasız med.": a.median(), "Arızalı med.": b.median(),
                 "Cohen d": (b.mean() - a.mean()) / sp,
                 "Mann-Whitney p": stats.mannwhitneyu(a, b).pvalue})
tab = pd.DataFrame(rows).round(3)
tab.to_csv(RES / "q1_karsilastirma.csv", index=False)
print(tab.to_string(index=False))
