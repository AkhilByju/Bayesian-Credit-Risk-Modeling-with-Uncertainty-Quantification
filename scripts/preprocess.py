# scripts/preprocess.py
from __future__ import annotations

from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split

from credit_risk.data import load_raw_credit_default, get_target_and_features
from credit_risk.features import build_preprocessor


def main():
    raw_path = Path("data/raw/credit_default.xls")  
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = 42
    test_size = 0.15
    val_size = 0.15 

    df = load_raw_credit_default(raw_path)
    X, y = get_target_and_features(df)

    # First split off test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # Then split train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size, random_state=seed, stratify=y_trainval
    )

    pre = build_preprocessor(X_train)
    X_train_t = pre.fit_transform(X_train)
    X_val_t = pre.transform(X_val)
    X_test_t = pre.transform(X_test)

    # Convert transformed matrices to DataFrames with generated feature names
    feature_names = []
    try:
        feature_names = pre.get_feature_names_out().tolist()
    except Exception:
        width = (X_train_t.shape[1] if len(getattr(X_train_t, "shape", ())) == 2 else 0)
        feature_names = [f"f{i}" for i in range(width)]

    import numpy as np

    def to_dense(a):
        # Handles scipy sparse matrices and numpy arrays
        return a.toarray() if hasattr(a, "toarray") else np.asarray(a)

    X_train_arr = to_dense(X_train_t)
    X_val_arr = to_dense(X_val_t)
    X_test_arr = to_dense(X_test_t)

    train_df = pd.DataFrame(X_train_arr, columns=feature_names)
    val_df = pd.DataFrame(X_val_arr, columns=feature_names)
    test_df = pd.DataFrame(X_test_arr, columns=feature_names)


    train_df["y"] = y_train.reset_index(drop=True).values
    val_df["y"] = y_val.reset_index(drop=True).values
    test_df["y"] = y_test.reset_index(drop=True).values

    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    print("✅ Wrote:")
    print(" - data/processed/train.csv")
    print(" - data/processed/val.csv")
    print(" - data/processed/test.csv")
    print(f"Shapes: train={train_df.shape}, val={val_df.shape}, test={test_df.shape}")


if __name__ == "__main__":
    main()
