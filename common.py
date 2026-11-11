"""Ortak yardımcı fonksiyonlar: veri yükleme, özellik seçimi, train/test bölme."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ai4i2020.csv"
FIG = ROOT / "figures"
RES = ROOT / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

TARGET = "Machine failure"
LEAK = ["TWF", "HDF", "PWF", "OSF", "RNF"]   # arıza türü göstergeleri (sızıntı)
IDS = ["UDI", "Product ID"]                    # tanımlayıcılar
NUM = ["Air temperature [K]", "Process temperature [K]",
       "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
SHORT = {"Air temperature [K]": "Hava sıc. (K)",
         "Process temperature [K]": "Proses sıc. (K)",
         "Rotational speed [rpm]": "Dönme hızı (rpm)",
         "Torque [Nm]": "Tork (Nm)",
         "Tool wear [min]": "Takım aşınması (dk)"}
SEED = 42


def load():
    return pd.read_csv(DATA)


def get_xy(df):
    """TWF/HDF/PWF/OSF/RNF, UDI ve Product ID çıkarılır; Type one-hot kodlanır."""
    X = df.drop(columns=LEAK + IDS + [TARGET])
    X = pd.get_dummies(X, columns=["Type"], drop_first=False, dtype=int)
    return X, df[TARGET]


def split(df):
    """%80 eğitim / %20 test, sınıf oranını koruyan (stratify) bölme."""
    X, y = get_xy(df)
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
