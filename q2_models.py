"""Soru 2 - Logistic Regression, KNN, Decision Tree karşılaştırması."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
from common import *

df = load()
Xtr, Xte, ytr, yte = split(df)
print("Girdiler:", list(Xtr.columns))
print("Eğitim:", Xtr.shape, "arıza %.2f%%" % (100 * ytr.mean()),
      "| Test:", Xte.shape, "arıza %.2f%%" % (100 * yte.mean()))

# Ölçekleme: LR ve KNN mesafe/gradyan temelli -> StandardScaler (yalnızca eğitimden öğrenilir, pipeline).
# Karar ağacı ölçekten etkilenmez, ham veri.
models = {
    "Dummy (hep arızasız)": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED)),
    "KNN (k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "Decision Tree": DecisionTreeClassifier(random_state=SEED),
}
rows, cms = [], {}
for name, m in models.items():
    m.fit(Xtr, ytr)
    p = m.predict(Xte)
    rows.append({"Model": name,
                 "Accuracy": accuracy_score(yte, p),
                 "Precision": precision_score(yte, p, zero_division=0),
                 "Recall": recall_score(yte, p),
                 "F1": f1_score(yte, p),
                 "Train Acc": accuracy_score(ytr, m.predict(Xtr))})
    cms[name] = confusion_matrix(yte, p)
res = pd.DataFrame(rows).round(3)
res.to_csv(RES / "q2_model_sonuclari.csv", index=False)
print(res.to_string(index=False))
for k, v in cms.items():
    print(k, "TN FP FN TP =", v.ravel())

fig, ax = plt.subplots(1, 3, figsize=(9, 2.6))
for a, (name, cm) in zip(ax, list(cms.items())[1:]):
    a.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            a.text(j, i, cm[i, j], ha="center", va="center", fontsize=11,
                   color="white" if cm[i, j] > cm.max() / 2 else "black")
    a.set_xticks([0, 1]); a.set_yticks([0, 1])
    a.set_xticklabels(["Tahmin 0", "Tahmin 1"], fontsize=8); a.set_yticklabels(["Gerçek 0", "Gerçek 1"], fontsize=8)
    a.set_title(name, fontsize=9)
plt.tight_layout(); plt.savefig(FIG / "q2_confusion.png", dpi=170); plt.close()

# --- Ek: tek bölmenin şansına bağımlılığı görmek için 5-kat çapraz doğrulama (yalnızca eğitim kümesi)
from sklearn.model_selection import StratifiedKFold, cross_validate
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
out = []
for name, m in list(models.items())[1:]:
    r = cross_validate(m, Xtr, ytr, cv=cv, scoring=["accuracy", "precision", "recall", "f1"])
    out.append({"Model": name, **{k: f"{r['test_'+k].mean():.3f} ± {r['test_'+k].std():.3f}"
                                  for k in ["accuracy", "precision", "recall", "f1"]}})
cvdf = pd.DataFrame(out); cvdf.to_csv(RES / "q2_cv_sonuclari.csv", index=False)
print("\n5-kat CV (eğitim kümesi):\n", cvdf.to_string(index=False))
