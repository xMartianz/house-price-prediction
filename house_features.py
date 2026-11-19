"""Feature engineering shared by the notebook and the Streamlit demo.

The fitted pipeline stored in models/house_price_pipeline.joblib references
`add_features` and `build_preprocessor` from this module, so the module must be
importable whenever the artifact is loaded.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# Columns where NaN means "the house does not have this" (per data_description).
NONE_MEANS_ABSENT = [
    "Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "PoolQC", "Fence", "MiscFeature", "MasVnrType",
]
# Numeric codes that are really categories.
NUMERIC_AS_CATEGORY = ["MSSubClass", "MoSold", "YrSold"]
DROP_COLUMNS = ["Id", "Utilities"]  # Id is a key; Utilities is almost constant.

QUALITY_MAP = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw Kaggle columns and add engineered features. Stateless."""
    df = df.copy()
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])

    for col in NONE_MEANS_ABSENT:
        if col in df.columns:
            df[col] = df[col].fillna("None")
    for col in NUMERIC_AS_CATEGORY:
        if col in df.columns:
            df[col] = df[col].astype("Int64").astype(str)

    # Ordinal quality scores (kept alongside the one-hot versions).
    for col in ["ExterQual", "KitchenQual", "BsmtQual", "HeatingQC", "FireplaceQu", "GarageQual"]:
        if col in df.columns:
            df[col + "_Score"] = df[col].map(QUALITY_MAP).astype(float)

    num = lambda c: pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["TotalSF"] = num("TotalBsmtSF") + num("1stFlrSF") + num("2ndFlrSF")
    df["TotalBath"] = (num("FullBath") + 0.5 * num("HalfBath")
                       + num("BsmtFullBath") + 0.5 * num("BsmtHalfBath"))
    df["TotalPorchSF"] = (num("OpenPorchSF") + num("EnclosedPorch") + num("3SsnPorch")
                          + num("ScreenPorch") + num("WoodDeckSF"))
    sold = pd.to_numeric(df["YrSold"], errors="coerce")
    df["HouseAge"] = (sold - num("YearBuilt")).clip(lower=0)
    df["YearsSinceRemod"] = (sold - num("YearRemodAdd")).clip(lower=0)
    df["QualxSF"] = num("OverallQual") * df["TotalSF"]
    df["HasGarage"] = (num("GarageArea") > 0).astype(int)
    df["HasBsmt"] = (num("TotalBsmtSF") > 0).astype(int)
    df["Has2ndFlr"] = (num("2ndFlrSF") > 0).astype(int)
    df["HasFireplace"] = (num("Fireplaces") > 0).astype(int)
    df["HasPool"] = (num("PoolArea") > 0).astype(int)

    # Log-transform strongly right-skewed areas (helps the linear model).
    for col in ["LotArea", "GrLivArea", "TotalSF", "1stFlrSF", "LotFrontage", "MasVnrArea"]:
        if col in df.columns:
            df["Log_" + col] = np.log1p(pd.to_numeric(df[col], errors="coerce"))
    return df


def build_preprocessor() -> Pipeline:
    """Feature engineering + imputation + scaling + one-hot encoding."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5,
                                 sparse_output=False)),
    ])
    encode = ColumnTransformer([
        ("num", numeric, make_column_selector(dtype_include="number")),
        ("cat", categorical, make_column_selector(dtype_exclude="number")),
    ], verbose_feature_names_out=False)
    return Pipeline([
        ("features", FunctionTransformer(add_features)),
        ("encode", encode),
    ])
