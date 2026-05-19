from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


class CustomModel:
    """
    HistGradientBoostingRegressor with rich temporal + cyclical features.

    Parameters
    ----------
    max_iter         : boosting iterations (default 2000)
    learning_rate    : shrinkage per tree (default 0.01)
    max_depth        : tree depth (default 6)
    min_samples_leaf : minimum leaf size (default 8)
    bias_mult        : post-prediction downward correction.
    random_state     : reproducibility seed
    """

    def __init__(
        self,
        max_iter: int = 2000,
        learning_rate: float = 0.01,
        max_depth: int = 6,
        min_samples_leaf: int = 8,
        bias_mult: float = 0.94,
        random_state: int = 42,
    ):
        self.bias_mult = bias_mult
        self._model = HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )

    # Feature engineering

    @staticmethod
    def _cyclical(series: pd.Series, period: float):
        angle = 2 * np.pi * series / period
        return np.sin(angle), np.cos(angle)

    @classmethod
    def _build_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        feats = pd.DataFrame(index=df.index)

        # Trend
        feats["trend"] = df["trend"]

        # Cyclical seasonal — sin/cos preserves Mon~Sun adjacency
        s, c = cls._cyclical(df["month"], 12)
        feats["month_sin"], feats["month_cos"] = s, c
        s, c = cls._cyclical(df["week_of_year"], 52)
        feats["woy_sin"], feats["woy_cos"] = s, c

        # Cyclical day-of-week — accept raw or reconstruct from one-hot
        if "dow" in df.columns:
            dow_raw = df["dow"]
        else:
            dow_cols = [c for c in df.columns if c.startswith("dow_")]
            if dow_cols:
                ohe = df[dow_cols].to_numpy()
                dow_raw = pd.Series(ohe.argmax(axis=1), index=df.index)
            else:
                dow_raw = pd.Series(np.zeros(len(df)), index=df.index)
        s, c = cls._cyclical(dow_raw, 7)
        feats["dow_sin"], feats["dow_cos"] = s, c

        # Binary flags
        feats["is_holiday"]      = df["is_holiday"]
        feats["is_school_break"] = df["is_school_break"]
        feats["is_weekend"]      = df["is_weekend"]
        feats["promo_flag"]      = df.get("promo_flag", pd.Series(0, index=df.index))

        # Interactions: multiplicative effects linear models cannot express
        feats["holiday_weekday"] = df["is_holiday"]      * (1 - df["is_weekend"])
        feats["school_weekday"]  = df["is_school_break"] * (1 - df["is_weekend"])

        # Lag / rolling features — NaN handled natively by HistGBR
        # same_dow_ewm = 0.5*lag7 + 0.3*lag14 + 0.15*lag21 + 0.05*lag28
        # trend7vs14   = roll7 - roll14  (demand momentum; tracks Nov→Dec descent)
        # level_ratio  = roll7 / roll28  (recent vs medium-term level)
        for col in (
            "lag7", "lag14",
            "roll7", "roll14", "roll28",
            "trend7vs14",
            "level_ratio",
            "same_dow_ewm",
        ):
            if col in df.columns:
                feats[col] = df[col]

        return feats

    # Public interface

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        Xf = self._build_features(X)
        self._model.fit(Xf, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Xf = self._build_features(X)
        raw = self._model.predict(Xf)
        return np.clip(raw * self.bias_mult, 0, None)
