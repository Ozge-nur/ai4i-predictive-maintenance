"""Soru 3 - Hata maliyeti, recall artırma ve aşırı öğrenme analizi (seçilen model: Decision Tree)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, precision_recall_curve)
from common import *

df = load()
Xtr, Xte, ytr, yte = split(df)
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)


def add_feats(X):
    """Fiziksel anlamlı türetilmiş değişkenler (arıza türü etiketi KULLANILMAZ)."""
    X = X.copy()
    X["Power [W]"] = X["Torque [Nm]"] * X["Rotational speed [rpm]"] * 2 * np.pi / 60
    X["Temp diff [K]"] = X["Process temperature [K]"] - X["Air temperature [K]"]
    X["Wear x Torque"] = X["Tool wear [min]"] * X["Torque [Nm]"]
    return X


def report(name, y, p):
    tn, fp, fn, tp = confusion_matrix(y, p).ravel()
    return {"Model": name, "Accuracy": accuracy_score(y, p), "Precision": precision_score(y, p, zero_division=0),
            "Recall": recall_score(y, p), "F1": f1_score(y, p), "FN (kaçan)": fn, "FP (yanlış alarm)": fp}


rows = []
# --- (a/b) Temel ağaç ve recall artırma adımları -------------------------------------------
base = DecisionTreeClassifier(random_state=SEED).fit(Xtr, ytr)
rows.append(report("1) Temel DT (budanmamış)", yte, base.predict(Xte)))

dt_d = DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=SEED).fit(Xtr, ytr)
rows.append(report("2) Budanmış DT (depth=5, leaf>=5)", yte, dt_d.predict(Xte)))

dt_w = DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, class_weight="balanced", random_state=SEED).fit(Xtr, ytr)
rows.append(report("3) + class_weight='balanced'", yte, dt_w.predict(Xte)))

# eşik düşürme: eşik yalnızca eğitim verisi çapraz-doğrulama tahminleriyle seçilir (test sızıntısı yok)
oof = cross_val_predict(DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=SEED),
                        Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
prec, rec, thr = precision_recall_curve(ytr, oof)
f2 = 5 * prec * rec / (4 * prec + rec + 1e-12)          # recall'a 2 kat ağırlık veren F2
t_best = thr[np.argmax(f2[:-1])]
print("Seçilen eşik (CV, F2 maks.): %.3f" % t_best)
rows.append(report(f"4) Budanmış DT + eşik={t_best:.2f}", yte, (dt_d.predict_proba(Xte)[:, 1] >= t_best).astype(int)))

# türetilmiş özellikler
Xtr2, Xte2 = add_feats(Xtr), add_feats(Xte)
dt_f = DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=SEED).fit(Xtr2, ytr)
rows.append(report("5) Budanmış DT + türetilmiş özellik", yte, dt_f.predict(Xte2)))
dt_fw = DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, class_weight="balanced", random_state=SEED).fit(Xtr2, ytr)
rows.append(report("6) Özellik + class_weight", yte, dt_fw.predict(Xte2)))

res = pd.DataFrame(rows)
res.iloc[:, 1:5] = res.iloc[:, 1:5].round(3)
res.to_csv(RES / "q3_recall_artirma.csv", index=False)
print(res.to_string(index=False))
print("\nTüretilmiş özellik önemleri:")
print(pd.Series(dt_f.feature_importances_, Xtr2.columns).sort_values(ascending=False).round(3).head(6))

# --- (c) Aşırı öğrenme: derinliğe göre eğitim vs CV başarısı --------------------------------
depths = list(range(1, 21))
tr_f1, cv_f1 = [], []
for d in depths:
    r = cross_validate(DecisionTreeClassifier(max_depth=d, random_state=SEED), Xtr, ytr, cv=cv,
                       scoring="f1", return_train_score=True)
    tr_f1.append(r["train_score"].mean()); cv_f1.append(r["test_score"].mean())
sweep = pd.DataFrame({"depth": depths, "train_f1": tr_f1, "cv_f1": cv_f1}).round(3)
sweep.to_csv(RES / "q3_derinlik_taramasi.csv", index=False)
print(sweep.to_string(index=False))

# 5-kat CV ile üç modelin kararlılığı (tek bölmenin şansına bağımlılığı azaltmak için)
fig, ax = plt.subplots(1, 2, figsize=(10, 3))
ax[0].plot(depths, tr_f1, "o-", label="Eğitim F1", color="#D1495B", ms=3)
ax[0].plot(depths, cv_f1, "s-", label="CV F1", color="#5B8DB8", ms=3)
ax[0].set_xlabel("max_depth"); ax[0].set_ylabel("F1 (arıza sınıfı)"); ax[0].legend(); ax[0].set_title("Aşırı öğrenme: derinlik taraması", fontsize=10)
ax[1].plot(rec[:-1], prec[:-1], color="#5B8DB8")
ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision"); ax[1].set_title("Eşik seçimi: Precision–Recall (CV)", fontsize=10)
i = np.argmax(f2[:-1]); ax[1].scatter(rec[i], prec[i], color="#D1495B", zorder=3, label=f"seçilen eşik={t_best:.2f}"); ax[1].legend()
plt.tight_layout(); plt.savefig(FIG / "q3_overfit_pr.png", dpi=170); plt.close()

# --- Temel DT: eğitim vs test boşluğu
print("\nTemel DT  Eğitim acc/recall: %.3f/%.3f | Test acc/recall: %.3f/%.3f" % (
    accuracy_score(ytr, base.predict(Xtr)), recall_score(ytr, base.predict(Xtr)),
    accuracy_score(yte, base.predict(Xte)), recall_score(yte, base.predict(Xte))))
