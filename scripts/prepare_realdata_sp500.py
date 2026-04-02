from __future__ import annotations

from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import yfinance as yf

from grpca_gd.datasets.artifacts import DatasetArtifact, save_artifact
from grpca_gd.synthetic.graphs import normalized_laplacian


def _hash_config(payload: dict) -> str:
    data = repr(sorted(payload.items())).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _corr_graph(prices: pd.DataFrame, corr_threshold: float) -> np.ndarray:
    returns = prices.pct_change().dropna()
    corr = returns.corr().fillna(0.0).to_numpy()
    W = (np.abs(corr) >= corr_threshold).astype(float)
    np.fill_diagonal(W, 0.0)
    return W


def prepare_sp500_artifact(
    out_dir: Path,
    prices: pd.DataFrame | None = None,
    tickers: list[str] | None = None,
    start: str = "2018-01-01",
    end: str = "2023-01-01",
    corr_threshold: float = 0.3,
) -> None:
    if prices is None:
        if tickers is None:
            raise ValueError("tickers required when prices not provided")
        data = yf.download(tickers, start=start, end=end, progress=False)
        prices = data["Adj Close"].dropna()

    W = _corr_graph(prices, corr_threshold)
    L = normalized_laplacian(W)
    X = prices.pct_change().dropna().to_numpy()

    prep_cfg = {
        "start": start,
        "end": end,
        "corr_threshold": corr_threshold,
        "tickers": list(prices.columns),
    }
    artifact = DatasetArtifact(
        artifact_id="sp500_corr_v1",
        artifact_version="v1",
        dataset="sp500",
        graph_family="correlation",
        data_source="yfinance",
        prep_config_hash=_hash_config(prep_cfg),
        eval_protocol_id="default",
        X=X,
        L=L,
        metadata={"tickers": list(prices.columns), "start": start, "end": end},
    )
    save_artifact(artifact, out_dir)
