# AI4I 2020 Kestirimci Bakım – Arıza Tahmini

Veri seti: [AI4I 2020 Predictive Maintenance (UCI #601)](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) – `data/ai4i2020.csv` (10.000 kayıt).

## Yapı
```
data/ai4i2020.csv          ham veri
src/common.py              veri yükleme, özellik seçimi, stratified train/test bölme
src/q1_eda.py              Soru 1: sınıf dağılımı, arızalı/arızasız karşılaştırma
src/q2_models.py           Soru 2: Logistic Regression, KNN, Decision Tree
src/q3_evaluation.py       Soru 3: recall artırma, eşik seçimi, aşırı öğrenme analizi
figures/  results/         üretilen grafikler ve tablolar (CSV)
report/rapor.pdf|docx      3 sayfalık rapor (her soru 1 sayfa)
```

## Çalıştırma
```bash
pip install -r requirements.txt
cd src
python q1_eda.py && python q2_models.py && python q3_evaluation.py
```

## Önemli tercihler
- TWF, HDF, PWF, OSF, RNF, UDI ve Product ID model girdilerinden çıkarılmıştır.
- %80/%20 stratified bölme, `random_state=42`; ölçekleme yalnızca LR ve KNN için, pipeline içinde (sızıntı yok).
- Q3'te karar eşiği yalnızca eğitim kümesi CV tahminleriyle seçilmiştir; test kümesi seçim için kullanılmamıştır.
