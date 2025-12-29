# src/credit_risk/data.py
from __future__ import annotations

from pathlib import Path
import pandas as pd


RAW_DIR = Path("data/raw")


def load_raw_credit_default(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")

    if path.suffix.lower() in [".xls", ".xlsx"]:
        df = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    # Normalize column names
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]

    # --- UCI XLS quirk fix ---
    # Some versions include a "description" row as the first data row (strings like
    # "default payment next month" inside Y, and long text inside X* columns).
    # If the first Y value is not numeric, drop the first row.
    if "Y" in df.columns:
        first_y = df.loc[df.index[0], "Y"]
        try:
            float(first_y)  # works if numeric-like
        except Exception:
            df = df.iloc[1:].reset_index(drop=True)

    return df



def get_target_and_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    For the UCI 'Default of credit card clients' dataset, common formats are:
    - Excel version: columns X1..X23 and target 'Y'
    - Other versions: target 'default_payment_next_month'
    """

    # Drop common junk columns
    drop_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
    if "ID" in df.columns:
        drop_cols.append("ID")
    if "id" in df.columns:
        drop_cols.append("id")
    df = df.drop(columns=list(set(drop_cols)), errors="ignore")

    # Target detection (include 'Y' explicitly)
    possible_targets = [
        "Y",
        "y",
        "default_payment_next_month",
        "default.payment.next.month",
        "DEFAULT_PAYMENT_NEXT_MONTH",
        "default",
    ]
    target_col = next((c for c in possible_targets if c in df.columns), None)

    if target_col is None:
        raise KeyError(
            f"Could not find target column. Try one of {possible_targets}. "
            f"Columns seen (first 25): {list(df.columns)[:25]}"
        )

    y = pd.to_numeric(df[target_col], errors="raise").astype(int)
    X = df.drop(columns=[target_col])

    # Ensure numeric types where possible (Excel sometimes reads as object)
    for c in X.columns:
        try:
            X[c] = pd.to_numeric(X[c])
        except Exception:
            pass


    return X, y
