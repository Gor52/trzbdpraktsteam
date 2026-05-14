import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("STUDENT_FAST", "1")

import steam_reviews_app as m


def test_mask_phone():
    assert m.mask_phone("+79161234567") == "+791***4567"
    assert m.mask_phone("short") == "short"


def test_mask_name():
    msk = m.mask_name("Player_99")
    assert "*" in msk
    assert msk.endswith("99")
    assert m.mask_name("АБ") == "А*"


def test_public_id_to_user_id():
    uid = "e7a8f000-1111-2222-3333-444455556666"
    assert m.public_id_to_user_id(uid) == m.public_id_to_user_id(uid)
    assert 1 <= m.public_id_to_user_id(uid) <= 5000
    assert m.public_id_to_user_id(None) == 1
    assert m.public_id_to_user_id("anon") == 1


def test_sync_sqlite_skips_when_no_dataframe(monkeypatch):
    monkeypatch.setattr(m.STATE, "df", None)
    m.sync_sqlite_reviews_into_state_df()


def test_sentiment_ru_label():
    lab, c = m.sentiment_ru_label(0.82)
    assert lab == "позитивный"
    assert abs(c - 0.82) < 1e-6
    lab2, c2 = m.sentiment_ru_label(0.2)
    assert lab2 == "негативный"
    assert c2 > 0.79


def test_compute_binary_metrics():
    y_t = np.array([0, 1, 1, 0, 1])
    y_p = np.array([0, 1, 0, 0, 1])
    d = m.compute_binary_metrics(y_t, y_p)
    assert "accuracy" in d and "f1" in d
    assert 0 <= d["f1"] <= 1


def test_load_steam_dataframe(tmp_path):
    csv = tmp_path / "mini_steam.csv"
    csv.write_text(
        "date_posted,funny,helpful,hour_played,is_early_access_review,recommendation,review,title\n"
        "2019-01-01,0,0,1,False,Recommended,Great game,Game A\n"
        "2019-01-02,0,0,1,False,Not Recommended,Bad game,Game A\n"
        "2019-01-03,0,0,1,False,Recommended,Ok,Game B\n",
        encoding="utf-8",
    )
    df = m.load_steam_dataframe(csv, sample_n=200, random_state=0)
    assert len(df) >= 3
    assert set(["text", "rating", "product", "date", "user_id", "sentiment"]).issubset(df.columns)


def test_recommend_user_based():
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2, 3],
            "product": ["A", "B", "C", "A", "B", "A"],
            "rating": [5, 4, 5, 2, 2, 5],
            "text": ["x"] * 6,
            "sentiment": [1, 1, 1, 0, 0, 1],
            "date": pd.date_range("2020-01-01", periods=6),
        }
    )
    out = m.recommend_user_based(df, target_user_id=1, top_n=2)
    assert isinstance(out, list)
    assert len(out) >= 1


def test_proba_positive_respects_class_column_order():
    class _M:
        classes_ = np.array([1, 0])

        def predict_proba(self, X):
            return np.array([[0.82, 0.18]], dtype=float)

    out = m.proba_positive_sentiment(_M(), np.zeros((1, 4)))[0]
    assert abs(float(out) - 0.82) < 1e-6


def test_predict_sentiment_mock(monkeypatch):
    class _Emb:
        def encode(self, texts, show_progress_bar=False, **kwargs):
            return np.zeros((len(texts), 4), dtype=np.float32)

    class _Clf:
        classes_ = np.array([0, 1])

        def predict_proba(self, X):
            return np.array([[0.1, 0.9]], dtype=float)

    monkeypatch.setattr(m, "ensure_pipeline", lambda: None)
    m.STATE.models_tuned = {"Mock": _Clf()}
    m.STATE.best_model_name = "Mock"
    m.STATE.embedder = _Emb()
    label, conf, name = m.predict_sentiment("test review text")
    assert label == "позитивный"
    assert conf >= 0.5
    assert name == "Mock"
