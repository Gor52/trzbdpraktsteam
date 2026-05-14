import gc
import io
import json
import os
import platform
import re
import sqlite3
import sys
import uuid
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, session, url_for

from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "steam_reviews.csv"
DB_PATH = BASE_DIR / "steam_reviews.sqlite"
LOG_PATH = BASE_DIR / "app.log"
CACHE_DIR = BASE_DIR / "steam_app_cache"
REPORT_PNG = BASE_DIR / "student_report.png"

STEAM_QUICK = os.environ.get("STEAM_QUICK", "0") == "1"
SAMPLE_N = int(os.environ.get("STEAM_SAMPLE_N", "2000" if STEAM_QUICK else "4500"))
CLUSTER_SUBSAMPLE = int(os.environ.get("STEAM_CLUSTER_N", "700" if STEAM_QUICK else "1200"))
RANDOM_STATE = 42
ADMIN_KEY = os.environ.get("STEAM_ADMIN_KEY", "devsecret")
STUDENT_FAST = os.environ.get("STUDENT_FAST", "0") == "1"
SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "steam-coursework-secret-change-me")

os.makedirs(CACHE_DIR, exist_ok=True)

PIPELINE_CACHE = CACHE_DIR / "trained_pipeline.joblib"
PIPELINE_SIG = CACHE_DIR / "trained_pipeline.sig.json"


def _pipeline_cache_sig():
    csv_t = int(CSV_PATH.stat().st_mtime) if CSV_PATH.exists() else 0
    db_t = int(DB_PATH.stat().st_mtime) if DB_PATH.exists() else 0
    return {"sample_n": SAMPLE_N, "csv_mtime": csv_t, "db_mtime": db_t, "quick": STEAM_QUICK, "rs": RANDOM_STATE}


def _try_load_pipeline_cache():
    if STUDENT_FAST or os.environ.get("STEAM_FORCE_RETRAIN"):
        return False
    if not PIPELINE_CACHE.exists() or not PIPELINE_SIG.exists():
        return False
    try:
        want = _pipeline_cache_sig()
        got = json.loads(PIPELINE_SIG.read_text(encoding="utf-8"))
        if got != want:
            return False
        print("Загрузка готового пайплайна с диска (кэш) — быстро", flush=True)
        blob = joblib.load(PIPELINE_CACHE)
        for k, v in blob.items():
            setattr(STATE, k, v)
        STATE.embedder = SentenceTransformer(STATE.embedder_name)
        STATE.last_sqlite_sync_mtime = None
        print("Кэш загружен. Можно открывать сайт.", flush=True)
        return True
    except Exception as ex:
        print("Кэш повреждён, переобучение:", ex, flush=True)
        return False


def _save_pipeline_cache():
    if STUDENT_FAST:
        return
    try:
        blob = {
            "df": STATE.df,
            "X": STATE.X,
            "y": STATE.y,
            "X_train": STATE.X_train,
            "X_test": STATE.X_test,
            "y_train": STATE.y_train,
            "y_test": STATE.y_test,
            "models_tuned": STATE.models_tuned,
            "metrics_before": STATE.metrics_before,
            "metrics_after": STATE.metrics_after,
            "grid_results": STATE.grid_results,
            "best_model_name": STATE.best_model_name,
            "roc_curves": STATE.roc_curves,
            "conclusion": STATE.conclusion,
            "cluster_labels_km": STATE.cluster_labels_km,
            "cluster_labels_db": STATE.cluster_labels_db,
            "cluster_meta": STATE.cluster_meta,
            "pca2": STATE.pca2,
            "df_cluster": STATE.df_cluster,
        }
        joblib.dump(blob, PIPELINE_CACHE, compress=1)
        PIPELINE_SIG.write_text(json.dumps(_pipeline_cache_sig()), encoding="utf-8")
        print("Пайплайн сохранён в кэш:", PIPELINE_CACHE, flush=True)
    except Exception as ex:
        print("Не удалось сохранить кэш:", ex, flush=True)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def grid_search_n_jobs():
    if STUDENT_FAST:
        return 1
    s = os.environ.get("STEAM_GRID_JOBS", "").strip()
    if s:
        return int(s)
    if platform.system() == "Windows":
        return 1
    return -1


def rf_n_jobs():
    if platform.system() == "Windows":
        return 1
    return -1


def cat_thread_count():
    if platform.system() == "Windows":
        return 2
    return -1


def log_event(user_id, action, result):
    ts = datetime.now().isoformat(timespec="seconds")
    line = f"{ts}\tuser={user_id}\taction={action}\tresult={result}\n"
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line)
    except OSError:
        pass


def read_last_log_lines(n=50):
    if not LOG_PATH.exists():
        return []
    try:
        text = LOG_PATH.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return lines[-n:]


def mask_phone(phone):
    if phone is None or (isinstance(phone, float) and np.isnan(phone)):
        return ""
    s = str(phone)
    if len(s) >= 10:
        return s[:4] + "***" + s[-4:]
    return s


def mask_name(name):
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return ""
    s = str(name)
    if len(s) <= 3:
        return s[0] + "*" * (len(s) - 1) if s else ""
    return s[:2] + "*" * (len(s) - 4) + s[-2:]


def public_id_to_user_id(user_public_id) -> int:
    """Стабильный числовой user_id для merged df; совпадает с отзывами из SQLite."""
    if user_public_id is None:
        return 1
    s = str(user_public_id).strip()
    if not s or s == "anon":
        return 1
    h = int(hash(s))
    x = abs(h % 5000) % 5000
    return int(x + 1)


def sentiment_ru_label(proba_positive):
    p = float(np.clip(proba_positive, 0.0, 1.0))
    if p >= 0.5:
        return "позитивный", p
    return "негативный", 1.0 - p


def proba_positive_sentiment(mdl, X):
    """Вероятность метки sentiment=1. Колонки predict_proba идут в порядке mdl.classes_, не всегда [,1]."""
    X = np.asarray(X)
    pp = mdl.predict_proba(X)
    classes = np.asarray(mdl.classes_)
    if pp.ndim != 2 or pp.shape[1] < 2:
        return pp.ravel().astype(float)
    if len(classes) != pp.shape[1]:
        return pp[:, 1].astype(float)
    idx = int(np.where(classes == 1)[0][0])
    return pp[:, idx].astype(float)


def top_products_by_positive_share(df, min_reviews=15):
    g = df.groupby("product").agg(
        n=("sentiment", "count"),
        pos_share=("sentiment", "mean"),
    )
    g = g[g["n"] >= min_reviews].sort_values("pos_share", ascending=False)
    return g.reset_index()


def top_k_words_negative(df, k=5):
    neg = df[df["sentiment"] == 0]["text"].fillna("").str.lower()
    blob = " ".join(neg.tolist())
    words = re.findall(r"[a-zA-Z]{3,}", blob)
    vc = pd.Series(words).value_counts().head(k)
    return list(vc.items())


def load_steam_dataframe(csv_path, sample_n=SAMPLE_N, random_state=RANDOM_STATE):
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Нет файла: {csv_path}")

    raw = pd.read_csv(csv_path, nrows=min(120_000, 200_000), on_bad_lines="skip")
    if len(raw) > sample_n:
        raw = raw.sample(n=sample_n, random_state=random_state).reset_index(drop=True)

    df = pd.DataFrame()
    df["text"] = raw["review"].fillna("").astype(str).str.slice(0, 4000)
    df["product"] = raw["title"].fillna("Unknown game").astype(str)
    df["date"] = pd.to_datetime(raw["date_posted"], errors="coerce")
    df["date"] = df["date"].fillna(pd.Timestamp("2000-01-01"))

    rec = raw["recommendation"].astype(str)
    df["rating"] = np.where(rec.str.contains("Not"), 1, 5)
    df["sentiment"] = np.where(rec.str.contains("Not"), 0, 1).astype(np.int8)

    rng = np.random.default_rng(random_state)
    df["user_id"] = rng.integers(1, 420, size=len(df))
    df["user_name"] = "Player_" + df["user_id"].astype(str)
    df["phone"] = "+79" + pd.Series(rng.integers(10**8, 10**9 - 1, size=len(df))).astype(str)

    df = df.drop_duplicates(subset=["text", "product"]).reset_index(drop=True)
    return df


def init_sqlite(db_path):
    db_path = Path(db_path)
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS user_reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_public_id TEXT NOT NULL,
            text TEXT NOT NULL,
            rating INTEGER NOT NULL,
            product TEXT NOT NULL,
            created_at TEXT NOT NULL,
            sentiment_label TEXT,
            sentiment_conf REAL
        );
        """
    )
    con.commit()
    con.close()


def fetch_sqlite_reviews(db_path):
    if not Path(db_path).is_file():
        return pd.DataFrame()
    con = sqlite3.connect(db_path)
    try:
        q = pd.read_sql_query("SELECT * FROM user_reviews", con)
    finally:
        con.close()
    if q.empty:
        return q
    df = pd.DataFrame()
    df["text"] = q["text"].astype(str)
    df["rating"] = q["rating"].astype(int)
    df["product"] = q["product"].astype(str)
    df["date"] = pd.to_datetime(q["created_at"], errors="coerce")
    df["sentiment"] = np.where(q["rating"] >= 4, 1, 0).astype(np.int8)
    df["user_id"] = q["user_public_id"].astype(str).map(public_id_to_user_id).astype(int)
    df["user_name"] = "WebUser_" + q["user_public_id"].astype(str).str.slice(0, 8)
    df["phone"] = "+79" + (q["user_public_id"].astype(str).map(lambda s: str(abs(hash(s)) % 10**9)).str.zfill(9))
    return df


def insert_user_review(db_path, user_public_id, text, rating, product, sentiment_label, sentiment_conf):
    init_sqlite(db_path)
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO user_reviews
        (user_public_id, text, rating, product, created_at, sentiment_label, sentiment_conf)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_public_id,
            text[:8000],
            int(rating),
            product[:500],
            datetime.now().isoformat(timespec="seconds"),
            sentiment_label,
            float(sentiment_conf),
        ),
    )
    con.commit()
    con.close()


def compute_binary_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def build_user_item_matrix(df):
    u_codes, user_uniques = pd.factorize(df["user_id"])
    i_codes, item_uniques = pd.factorize(df["product"])
    R = np.zeros((len(user_uniques), len(item_uniques)), dtype=np.float64)
    for u, i, r in zip(u_codes, i_codes, df["rating"].astype(float).values):
        R[u, i] = r
    return R, user_uniques, item_uniques


def recommend_user_based(df, target_user_id, top_n=6, n_neighbors=40):
    if df.empty:
        return []
    R, user_uniques, item_uniques = build_user_item_matrix(df)
    matches = np.where(user_uniques == target_user_id)[0]
    if matches.size == 0:
        pop = df.groupby("product")["rating"].mean().sort_values(ascending=False).head(top_n)
        return [(str(k), float(v)) for k, v in pop.items()]

    u = int(matches[0])
    user_vec = R[u].copy()
    rated = user_vec > 0

    Rc = R.copy()
    cnt = (Rc > 0).sum(axis=1, keepdims=True)
    sm = Rc.sum(axis=1, keepdims=True)
    means = np.zeros_like(sm)
    ok = cnt > 0
    means[ok] = sm[ok] / cnt[ok]
    means = np.nan_to_num(means)
    Rc = np.where(R > 0, R - means, 0.0)

    sim = Rc @ Rc[u]
    norms = np.linalg.norm(Rc, axis=1) * (np.linalg.norm(Rc[u]) + 1e-9)
    cos = sim / (norms + 1e-9)
    neigh = np.argsort(-cos)[1 : n_neighbors + 1]

    scores = np.zeros(len(item_uniques))
    for v in neigh:
        w = max(cos[v], 0.0)
        if w <= 0:
            continue
        scores += w * R[v]

    mask_unseen = ~rated
    scores = np.where(mask_unseen, scores, -np.inf)
    order = np.argsort(-scores)[:top_n]
    out = []
    for j in order:
        if not np.isfinite(scores[j]) or scores[j] <= -1e8:
            continue
        out.append((str(item_uniques[j]), float(scores[j])))
    if len(out) < top_n:
        pop = df.groupby("product")["rating"].mean().sort_values(ascending=False)
        for p, val in pop.items():
            if len(out) >= top_n:
                break
            if all(p != x[0] for x in out):
                out.append((str(p), float(val)))
    return out[:top_n]


class PipelineState:
    def __init__(self):
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.embedder_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.embedder = None
        self.models_default = {}
        self.models_tuned = {}
        self.metrics_before = {}
        self.metrics_after = {}
        self.grid_results = {}
        self.best_model_name = ""
        self.cluster_labels_km = None
        self.cluster_labels_db = None
        self.cluster_meta = []
        self.pca2 = None
        self.df_cluster = None
        self.roc_curves = {}
        self.conclusion = ""
        self.last_sqlite_sync_mtime = None


STATE = PipelineState()


def encode_embeddings(texts, embedder, batch_size=64):
    if embedder is None:
        raise RuntimeError("SentenceTransformer недоступен")
    return np.asarray(embedder.encode(texts, batch_size=batch_size, show_progress_bar=False))


def grid_search_model(model, grid, X_tr, y_tr):
    nj = grid_search_n_jobs()
    gs = GridSearchCV(
        model,
        grid,
        scoring="f1",
        cv=2,
        n_jobs=nj,
        refit=True,
        verbose=1 if nj == 1 else 0,
    )
    gs.fit(X_tr, y_tr)
    return gs


def build_pipeline():
    global STATE
    if _try_load_pipeline_cache():
        return STATE

    print("\nЗагружаем steam_reviews.csv и SQLite", flush=True)
    init_sqlite(DB_PATH)
    df_csv = load_steam_dataframe(CSV_PATH, sample_n=SAMPLE_N)
    df_sql = fetch_sqlite_reviews(DB_PATH)
    if not df_sql.empty:
        df = pd.concat([df_csv, df_sql], ignore_index=True)
    else:
        df = df_csv
    df = df.drop_duplicates(subset=["text", "product"]).reset_index(drop=True)
    print("Строк для обучения:", len(df), flush=True)

    if SentenceTransformer is None:
        raise RuntimeError("Установите sentence-transformers и torch")

    cache_emb = CACHE_DIR / f"emb_{len(df)}_{SAMPLE_N}.npy"
    texts = df["text"].tolist()

    print("SentenceTransformer: векторизация отзывов", flush=True)
    STATE.embedder = SentenceTransformer(STATE.embedder_name)
    if cache_emb.exists() and not STUDENT_FAST:
        X = np.load(cache_emb)
        if X.shape[0] != len(df):
            X = encode_embeddings(texts, STATE.embedder)
            np.save(cache_emb, X)
    else:
        X = encode_embeddings(texts, STATE.embedder)
        if not STUDENT_FAST:
            np.save(cache_emb, X)

    y = df["sentiment"].values.astype(int)
    idx_all = np.arange(len(X))
    idx_train, idx_test = train_test_split(
        idx_all, test_size=0.22, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_test = X[idx_train], X[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]
    texts_train = [texts[i] for i in idx_train]

    STATE.df = df
    STATE.X = X
    STATE.y = y
    STATE.X_train, STATE.X_test = X_train, X_test
    STATE.y_train, STATE.y_test = y_train, y_test

    print("Обучение моделей (до подбора гиперпараметров)", flush=True)
    models_default = {}
    if XGBClassifier is not None:
        models_default["XGBoost"] = XGBClassifier(
            n_estimators=120,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.9,
            random_state=RANDOM_STATE,
            verbosity=0,
            n_jobs=rf_n_jobs(),
        )
    models_default["RandomForest"] = RandomForestClassifier(
        n_estimators=120, max_depth=18, random_state=RANDOM_STATE, n_jobs=rf_n_jobs()
    )
    models_default["LogisticRegression"] = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    models_default["MLPClassifier"] = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=120, random_state=RANDOM_STATE)
    if CatBoostClassifier is not None:
        models_default["CatBoost"] = CatBoostClassifier(
            depth=6,
            iterations=200,
            learning_rate=0.1,
            verbose=False,
            random_seed=RANDOM_STATE,
            thread_count=cat_thread_count(),
        )

    metrics_before = {}
    roc_curves = {}
    for name, mdl in models_default.items():
        mdl.fit(X_train, y_train)
        pred = mdl.predict(X_test)
        metrics_before[name] = compute_binary_metrics(y_test, pred)
        if hasattr(mdl, "predict_proba"):
            proba = mdl.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, proba)
            roc_curves[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(auc(fpr, tpr))}

    print(
        "Подсказка: GridSearch на Windows идёт в 1 процесс (не вешает ПК). "
        "Ускорить: set STEAM_GRID_JOBS=-1",
        flush=True,
    )
    if STEAM_QUICK:
        print("Режим STEAM_QUICK: маленькая выборка и сжатые сетки GridSearch.", flush=True)
        grids = {
            "LogisticRegression": (
                LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
                {"C": [0.5, 2.0], "penalty": ["l2"], "solver": ["lbfgs"]},
            ),
            "RandomForest": (
                RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=rf_n_jobs()),
                {"n_estimators": [80], "max_depth": [12, 18], "min_samples_leaf": [1, 2]},
            ),
            "MLPClassifier": (
                MLPClassifier(max_iter=150, random_state=RANDOM_STATE),
                {
                    "hidden_layer_sizes": [(64,), (128,)],
                    "alpha": [1e-4, 1e-3],
                    "learning_rate_init": [1e-3],
                },
            ),
        }
        if XGBClassifier is not None:
            grids["XGBoost"] = (
                XGBClassifier(random_state=RANDOM_STATE, verbosity=0, n_jobs=rf_n_jobs()),
                {"max_depth": [4, 6], "learning_rate": [0.1], "n_estimators": [80], "subsample": [0.9]},
            )
        if CatBoostClassifier is not None:
            grids["CatBoost"] = (
                CatBoostClassifier(verbose=False, random_seed=RANDOM_STATE, thread_count=cat_thread_count()),
                {"depth": [4, 6], "learning_rate": [0.1], "iterations": [100, 160]},
            )
    else:
        grids = {
        "LogisticRegression": (
            LogisticRegression(max_iter=4000, random_state=RANDOM_STATE),
            {"C": [0.25, 1.0, 4.0], "penalty": ["l2"], "solver": ["lbfgs", "liblinear"]},
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=rf_n_jobs()),
            {"n_estimators": [80, 160], "max_depth": [12, 24, None], "min_samples_leaf": [1, 2, 4]},
        ),
        "MLPClassifier": (
            MLPClassifier(max_iter=200, random_state=RANDOM_STATE),
            {
                "hidden_layer_sizes": [(64,), (128,), (128, 64)],
                "alpha": [1e-5, 1e-4, 1e-3],
                "learning_rate_init": [1e-3, 3e-3],
            },
        ),
        }
        if XGBClassifier is not None:
            grids["XGBoost"] = (
                XGBClassifier(random_state=RANDOM_STATE, verbosity=0, n_jobs=rf_n_jobs()),
                {"max_depth": [3, 6], "learning_rate": [0.05, 0.15], "n_estimators": [80, 160], "subsample": [0.85, 1.0]},
            )
        if CatBoostClassifier is not None:
            grids["CatBoost"] = (
                CatBoostClassifier(verbose=False, random_seed=RANDOM_STATE, thread_count=cat_thread_count()),
                {"depth": [4, 6], "learning_rate": [0.08, 0.15], "iterations": [120, 220]},
            )

    models_tuned = {}
    metrics_after = {}
    grid_results = {}
    for key, (base, grid) in grids.items():
        if key not in models_default:
            continue
        print(f"   GridSearch: {key} ...", flush=True)
        gs = grid_search_model(base, grid, X_train, y_train)
        best = gs.best_estimator_
        models_tuned[key] = best
        pred = best.predict(X_test)
        metrics_after[key] = compute_binary_metrics(y_test, pred)
        grid_results[key] = {"best_params": gs.best_params_, "best_score_cv_f1": float(gs.best_score_)}
        if hasattr(best, "predict_proba"):
            proba = best.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, proba)
            roc_curves[key + " (tuned)"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(auc(fpr, tpr))}

    best_name = max(metrics_after, key=lambda k: metrics_after[k]["f1"])
    STATE.models_tuned = models_tuned
    STATE.metrics_before = metrics_before
    STATE.metrics_after = metrics_after
    STATE.grid_results = grid_results
    STATE.best_model_name = best_name
    STATE.roc_curves = roc_curves
    STATE.models_default = {}
    del models_default
    gc.collect()

    best_f1 = metrics_after[best_name]["f1"]
    tmp = sorted(metrics_after.items(), key=lambda kv: kv[1]["f1"], reverse=True)
    second = tmp[1][0] if len(tmp) > 1 else ""
    STATE.conclusion = (
        "Итог: лучше всего отработала " + str(best_name) + " с F1=" + str(round(best_f1, 3))
        + ". Вторая по качеству — " + str(second)
        + ". Векторизация везде через SentenceTransformer."
    )
    print(STATE.conclusion, flush=True)

    print("KMeans + DBSCAN + PCA-2D", flush=True)
    gc.collect()
    rng = np.random.default_rng(RANDOM_STATE)
    csize = min(CLUSTER_SUBSAMPLE, len(X_train))
    sub_idx = rng.choice(np.arange(len(X_train)), size=csize, replace=False)
    Xc = np.asarray(X_train[sub_idx], dtype=np.float32, order="C")
    yc = y_train[sub_idx]
    texts_sub = [texts_train[i] for i in sub_idx]

    n_comp = int(min(32, max(3, Xc.shape[0] - 1), Xc.shape[1]))
    pca_red = PCA(n_components=n_comp, random_state=RANDOM_STATE, svd_solver="randomized")
    Xr = np.asarray(pca_red.fit_transform(Xc), dtype=np.float32)
    del Xc
    gc.collect()

    scaler = StandardScaler()
    Xs = np.asarray(scaler.fit_transform(Xr), dtype=np.float32)

    km = KMeans(n_clusters=6, random_state=RANDOM_STATE, n_init=10)
    lab_km = km.fit_predict(Xs)
    db = DBSCAN(eps=0.95, min_samples=5, metric="euclidean")
    lab_db = db.fit_predict(Xs)

    pca2 = np.asarray(Xr[:, :2], dtype=np.float32)
    del Xr, Xs, scaler, pca_red
    gc.collect()

    from sklearn.feature_extraction.text import CountVectorizer

    cv = CountVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 1))
    meta = []

    for cid in sorted(set(lab_km)):
        mask = lab_km == cid
        sub = [texts_sub[i] for i in np.flatnonzero(mask)]
        y_sub = yc[mask]
        pos_share = float(y_sub.mean()) if y_sub.size else 0.0
        try:
            mat = cv.fit_transform(sub)
            sums = np.asarray(mat.sum(axis=0)).ravel()
            feats = np.array(cv.get_feature_names_out())
            topw = feats[np.argsort(-sums)[:10]].tolist()
        except Exception:
            topw = []
        meta.append(
            {
                "algorithm": "KMeans",
                "cluster_id": int(cid),
                "top_words": topw,
                "positive_share": pos_share,
                "size": int(mask.sum()),
            }
        )

    for cid in sorted(x for x in set(lab_db) if x >= 0):
        mask = lab_db == cid
        if mask.sum() < 5:
            continue
        sub = [texts_sub[i] for i in np.flatnonzero(mask)]
        y_sub = yc[mask]
        pos_share = float(y_sub.mean()) if y_sub.size else 0.0
        try:
            mat = cv.fit_transform(sub)
            sums = np.asarray(mat.sum(axis=0)).ravel()
            feats = np.array(cv.get_feature_names_out())
            topw = feats[np.argsort(-sums)[:10]].tolist()
        except Exception:
            topw = []
        meta.append(
            {
                "algorithm": "DBSCAN",
                "cluster_id": int(cid),
                "top_words": topw,
                "positive_share": pos_share,
                "size": int(mask.sum()),
            }
        )

    STATE.cluster_labels_km = lab_km
    STATE.cluster_labels_db = lab_db
    STATE.pca2 = pca2
    STATE.cluster_meta = meta
    STATE.df_cluster = pd.DataFrame(
        {
            "x": pca2[:, 0],
            "y": pca2[:, 1],
            "kmeans": lab_km,
            "dbscan": lab_db,
            "sentiment": yc,
            "text": [t[:400] for t in texts_sub],
        }
    )

    print("Сохраняем student_report.png", flush=True)
    save_png_report(df, metrics_before, metrics_after, confusion_matrix(y_test, models_tuned[best_name].predict(X_test)))

    _save_pipeline_cache()
    try:
        STATE.last_sqlite_sync_mtime = DB_PATH.stat().st_mtime if DB_PATH.is_file() else None
    except OSError:
        STATE.last_sqlite_sync_mtime = None
    return STATE


def save_png_report(df, metrics_before, metrics_after, cm):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Отчет по steam отзывам", fontsize=14)

    axes[0, 0].hist(df["rating"], bins=[1, 2, 3, 4, 5, 6], edgecolor="black", color="steelblue", alpha=0.75)
    axes[0, 0].set_title("Распределение оценок (1/5 из Steam)")

    daily = df.groupby(df["date"].dt.date).size().reset_index(name="c")
    daily["date"] = pd.to_datetime(daily["date"])
    axes[0, 1].plot(daily["date"], daily["c"], color="darkorange", linewidth=1.6)
    axes[0, 1].set_title("Динамика числа отзывов (по дням)")
    axes[0, 1].tick_params(axis="x", rotation=35)

    names = list(metrics_after.keys())
    f1b = [metrics_before[n]["f1"] for n in names if n in metrics_before]
    f1a = [metrics_after[n]["f1"] for n in names]
    x = np.arange(len(names))
    w = 0.35
    axes[1, 0].bar(x - w / 2, [metrics_before[n]["f1"] if n in metrics_before else 0 for n in names], width=w, label="до GridSearch", color="cadetblue")
    axes[1, 0].bar(x + w / 2, f1a, width=w, label="после GridSearch", color="coral")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(names, rotation=25, ha="right")
    axes[1, 0].set_title("F1 до и после подбора")
    axes[1, 0].legend()

    im = axes[1, 1].imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[1, 1].text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    axes[1, 1].set_title("Матрица ошибок (лучшая модель)")
    axes[1, 1].set_xticks([0, 1])
    axes[1, 1].set_yticks([0, 1])
    axes[1, 1].set_xticklabels(["негатив", "позитив"])
    axes[1, 1].set_yticklabels(["негатив", "позитив"])
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(REPORT_PNG, dpi=140, bbox_inches="tight")
    plt.close(fig)


def sync_sqlite_reviews_into_state_df():
    """Подмешать новые строки из SQLite в STATE.df (графики, CF, облако) без переобучения."""
    if STATE.df is None or not DB_PATH.is_file():
        return
    try:
        mtime = DB_PATH.stat().st_mtime
    except OSError:
        return
    if STATE.last_sqlite_sync_mtime is not None and mtime <= STATE.last_sqlite_sync_mtime:
        return
    df_sql = fetch_sqlite_reviews(DB_PATH)
    if df_sql.empty:
        STATE.last_sqlite_sync_mtime = mtime
        return
    cols = list(STATE.df.columns)
    if not all(c in df_sql.columns for c in cols):
        STATE.last_sqlite_sync_mtime = mtime
        return
    merged = pd.concat([STATE.df, df_sql[cols]], ignore_index=True)
    STATE.df = merged.drop_duplicates(subset=["text", "product"], keep="last").reset_index(drop=True)
    STATE.last_sqlite_sync_mtime = mtime


def ensure_pipeline():
    if STATE.df is None:
        build_pipeline()
    sync_sqlite_reviews_into_state_df()


def predict_sentiment(text):
    ensure_pipeline()
    mdl = STATE.models_tuned.get(STATE.best_model_name)
    if mdl is None:
        raise RuntimeError("Модель не обучена")
    emb = STATE.embedder.encode([text[:4000]], show_progress_bar=False)
    proba = float(proba_positive_sentiment(mdl, np.asarray(emb))[0])
    label, conf = sentiment_ru_label(proba)
    return label, conf, STATE.best_model_name


def _plot_figure_json_safe(fig):
    import plotly.io as pio

    return json.loads(pio.to_json(fig))


def _cluster_scatter_lists(dfc, hover_slice: int = 120):
    x = dfc["x"].astype(float).tolist()
    y = dfc["y"].astype(float).tolist()
    colors = dfc["kmeans"].astype(int).tolist()
    hover_txt = (
        dfc["text"]
        .fillna("")
        .astype(str)
        .str.slice(0, hover_slice)
        .str.replace("<", " ", regex=False)
        .str.replace(">", " ", regex=False)
        .tolist()
    )
    customdata = [[int(c)] for c in colors]
    hover_labels = [f"Кластер {int(c)}<br>{h}" for c, h in zip(colors, hover_txt)]
    return x, y, colors, hover_labels, customdata


app = Flask(__name__)
app.secret_key = SECRET_KEY


@app.before_request
def _assign_user():
    if "user_public_id" not in session:
        session["user_public_id"] = str(uuid.uuid4())


@app.after_request
def _after(resp):
    uid = session.get("user_public_id", "anon")
    log_event(uid, f"{request.method} {request.path}", f"status={resp.status_code}")
    return resp


@app.route("/")
def index():
    ensure_pipeline()
    df = STATE.df
    uid = session.get("user_public_id")

    pos_share_table = top_products_by_positive_share(df).head(12)
    top_neg_words = top_k_words_negative(df, k=5)

    recs = recommend_user_based(df, public_id_to_user_id(uid), top_n=6)

    import plotly.graph_objects as go

    rcol = df["rating"].astype(int).clip(1, 5)
    counts = rcol.value_counts().reindex([1, 2, 3, 4, 5], fill_value=0).astype(int)
    rating_fig = go.Figure(
        data=[
            go.Bar(
                x=counts.index.tolist(),
                y=counts.values.tolist(),
                marker_color="#3a7ca5",
            )
        ]
    )
    rating_fig.update_layout(
        title="Распределение оценок",
        template="plotly_white",
        height=360,
        xaxis_title="Оценка",
        yaxis_title="Число отзывов",
        xaxis=dict(tickmode="linear", dtick=1, range=[0.5, 5.5]),
    )

    daily = df.groupby(df["date"].dt.date).size().reset_index(name="c")
    daily["date"] = pd.to_datetime(daily["date"])
    line_fig = go.Figure(data=[go.Scatter(x=daily["date"], y=daily["c"], mode="lines+markers", line=dict(color="#e67e22"))])
    line_fig.update_layout(title="Отзывы по дням", template="plotly_white", height=360)

    wc_b64 = ""
    try:
        from wordcloud import WordCloud

        wc = WordCloud(width=900, height=400, background_color="white", colormap="viridis")
        wc.generate(" ".join(df["text"].str.slice(0, 300).tolist()))
        buf = io.BytesIO()
        wc.to_image().save(buf, format="PNG")
        import base64

        wc_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        pass

    rows = []
    for name in STATE.metrics_after:
        b = STATE.metrics_before.get(name, {})
        a = STATE.metrics_after[name]
        rows.append(
            {
                "model": name,
                "acc_b": f"{b.get('accuracy', 0):.3f}",
                "f1_b": f"{b.get('f1', 0):.3f}",
                "acc_a": f"{a['accuracy']:.3f}",
                "f1_a": f"{a['f1']:.3f}",
            }
        )

    cluster_plot = {}
    if STATE.df_cluster is not None and len(STATE.df_cluster):
        dfc = STATE.df_cluster
        cx, cy, ccol, hover_labels, cdata = _cluster_scatter_lists(dfc, hover_slice=100)
        figc = go.Figure(
            data=[
                go.Scatter(
                    x=cx,
                    y=cy,
                    mode="markers",
                    marker=dict(size=7, color=ccol, colorscale="Turbo", showscale=True),
                    customdata=cdata,
                    text=hover_labels,
                    hovertemplate="%{text}<extra></extra>",
                )
            ]
        )
        figc.update_layout(title="Кластеры (KMeans) в PCA-2D", template="plotly_white", height=480)
        cluster_plot = _plot_figure_json_safe(figc)

    return render_template(
        "index.html",
        pos_share_table=pos_share_table,
        top_neg_words=top_neg_words,
        recs=recs,
        rating_plot=_plot_figure_json_safe(rating_fig),
        line_plot=_plot_figure_json_safe(line_fig),
        wc_b64=wc_b64,
        metric_rows=rows,
        conclusion=STATE.conclusion,
        best_model=STATE.best_model_name,
        cluster_plot=cluster_plot,
        user_id_display=str(uid)[:8],
    )


@app.route("/submit_review", methods=["POST"])
def submit_review():
    ensure_pipeline()
    text = request.form.get("text", "").strip()
    rating = int(request.form.get("rating", "5"))
    product = request.form.get("product", "").strip() or "Unknown game"
    rating = max(1, min(5, rating))
    uid = session.get("user_public_id", "anon")
    label, conf, _ = predict_sentiment(text)
    insert_user_review(DB_PATH, uid, text, rating, product, label, conf)
    log_event(uid, "submit_review", f"rating={rating};sentiment={label};conf={conf:.3f}")
    return redirect(url_for("index"))


@app.route("/predict_ajax", methods=["POST"])
def predict_ajax():
    data = request.get_json(force=True, silent=True) or {}
    text = str(data.get("text", ""))
    label, conf, mname = predict_sentiment(text)
    uid = session.get("user_public_id", "anon")
    log_event(uid, "predict_ajax", f"{label};{conf:.3f};model={mname}")
    return jsonify({"label": label, "confidence": conf, "model": mname})


@app.route("/products")
def products_page():
    ensure_pipeline()
    df = STATE.df
    g = df.groupby("product").agg(
        avg_rating=("rating", "mean"),
        n=("rating", "count"),
        pos_share=("sentiment", "mean"),
    ).reset_index().sort_values("n", ascending=False)
    log_event(session.get("user_public_id", "anon"), "products", f"rows={len(g)}")
    return render_template("products.html", table=g.head(80))


@app.route("/models")
def models_page():
    ensure_pipeline()
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=2, subplot_titles=("ROC-кривые (после подбора)", "F1 до / после"))
    for name, cur in STATE.roc_curves.items():
        if "(tuned)" not in name:
            continue
        base = name.replace(" (tuned)", "")
        fig.add_trace(go.Scatter(x=cur["fpr"], y=cur["tpr"], name=f"{base} AUC={cur['auc']:.3f}"), row=1, col=1)

    names = list(STATE.metrics_after.keys())
    fig.add_trace(
        go.Bar(name="до", x=names, y=[STATE.metrics_before[n]["f1"] if n in STATE.metrics_before else 0 for n in names]),
        row=1,
        col=2,
    )
    fig.add_trace(go.Bar(name="после", x=names, y=[STATE.metrics_after[n]["f1"] for n in names]), row=1, col=2)
    fig.update_layout(
        template="plotly_dark",
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        barmode="group",
    )
    plot_json = _plot_figure_json_safe(fig)
    rows = []
    for n in STATE.metrics_after:
        b, a = STATE.metrics_before.get(n, {}), STATE.metrics_after[n]
        rows.append(
            {
                "model": n,
                "acc_b": b.get("accuracy", 0),
                "pr_b": b.get("precision", 0),
                "rc_b": b.get("recall", 0),
                "f1_b": b.get("f1", 0),
                "acc_a": a["accuracy"],
                "pr_a": a["precision"],
                "rc_a": a["recall"],
                "f1_a": a["f1"],
                "best_params": str(STATE.grid_results.get(n, {}).get("best_params", {})),
            }
        )
    log_event(session.get("user_public_id", "anon"), "models", "view")
    return render_template("models.html", rows=rows, plot_json=plot_json, conclusion=STATE.conclusion)


@app.route("/clusters")
def clusters_page():
    ensure_pipeline()
    import plotly.graph_objects as go

    dfc = STATE.df_cluster
    if dfc is None or len(dfc) == 0:
        plot_json = {}
        return render_template("clusters.html", plot_json=plot_json, meta=STATE.cluster_meta or [])
    cx, cy, ccol, hover_labels, cdata = _cluster_scatter_lists(dfc, hover_slice=120)
    fig = go.Figure(
        go.Scatter(
            x=cx,
            y=cy,
            mode="markers",
            marker=dict(
                size=8,
                color=ccol,
                colorscale="Portland",
                showscale=True,
                colorbar=dict(title="кластер"),
            ),
            customdata=cdata,
            text=hover_labels,
            hovertemplate="%{text}<extra></extra>",
        )
    )
    fig.update_layout(title="Интерактивно: наведите на точку (кластер + фрагмент отзыва)", template="plotly_white", height=560)
    plot_json = _plot_figure_json_safe(fig)
    log_event(session.get("user_public_id", "anon"), "clusters", "view")
    return render_template("clusters.html", plot_json=plot_json, meta=STATE.cluster_meta)


@app.route("/api/cluster_samples/<int:cid>")
def api_cluster_samples(cid):
    ensure_pipeline()
    dfc = STATE.df_cluster
    if dfc is None or dfc.empty:
        return jsonify({"samples": []})
    sub = dfc[dfc["kmeans"] == cid]["text"].head(12).tolist()
    log_event(session.get("user_public_id", "anon"), "api_cluster_samples", f"cluster={cid};n={len(sub)}")
    return jsonify({"cluster": cid, "samples": sub})


@app.route("/admin/logs")
def admin_logs():
    key = request.args.get("key", "")
    if key != ADMIN_KEY:
        log_event(session.get("user_public_id", "anon"), "admin_logs", "denied")
        return "Доступ запрещён", 403
    lines = read_last_log_lines(50)
    log_event(session.get("user_public_id", "anon"), "admin_logs", "ok")
    return render_template("admin_logs.html", lines=lines)


def main():
    print(
        "Steam отзывы | SAMPLE_N=%s | STEAM_QUICK=%s"
        % (SAMPLE_N, STEAM_QUICK),
        flush=True,
    )
    print(
        "Первый запуск: эмбеддинги + GridSearch (долго). Повторный — из кэша steam_app_cache, быстро.",
        flush=True,
    )
    print(
        "Быстрая проверка: set STEAM_QUICK=1 | заново обучить: set STEAM_FORCE_RETRAIN=1",
        flush=True,
    )
    if not CSV_PATH.is_file():
        print("Нет файла steam_reviews.csv рядом со скриптом:", CSV_PATH, flush=True)
        sys.exit(1)
    build_pipeline()
    print("Сервер: http://127.0.0.1:5000/  (откройте в браузере)", flush=True)
    print("Логи админа (нужен key): /admin/logs?key=" + ADMIN_KEY, flush=True)
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    main()
