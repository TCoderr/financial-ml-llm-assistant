# app.py — Chat + Excel Upload + ML(XGBoost) + LLM(OpenRouter)
#           + Auto-Train + Anti-Cutoff + SQLite Memory
#           + Sohbette hisse adı geçince otomatik açıklama (cache)
# ---------------------------------------------------------------------------------------
# .env:
#   TRAINING_EXCEL=./Tum_Bilancolar.xlsx
#   FORCE_RETRAIN=false
#   OPENROUTER_API_KEY=or-xxxxxxxxxxxxxxxxxxxxxxxx
#   OPENROUTER_MODEL=openai/gpt-4o-mini
#   APP_URL=http://localhost:8001
#   APP_NAME=LLMTest-TR
# ---------------------------------------------------------------------------------------

import os, io, json, joblib, warnings, sqlite3
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

import xgboost as xgb
from openai import OpenAI

warnings.filterwarnings("ignore", category=FutureWarning)

# -------------------- ENV --------------------
load_dotenv()
TRAINING_EXCEL     = os.getenv("TRAINING_EXCEL", "./Tum_Bilancolar.xlsx")
FORCE_RETRAIN      = os.getenv("FORCE_RETRAIN", "false").lower() == "true"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL   = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
APP_URL            = os.getenv("APP_URL", "http://localhost:8001")
APP_NAME           = os.getenv("APP_NAME", "LLMTest-TR")

# -------------------- FastAPI --------------------
app = FastAPI(title="ML+LLM — Chat Upload (Combo + Memory)", version="2.4")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# -------------------- OpenRouter client --------------------
or_client: Optional[OpenAI] = None
if OPENROUTER_API_KEY:
    or_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        default_headers={"HTTP-Referer": APP_URL, "X-Title": APP_NAME},
    )

# -------------------- SQLite Memory --------------------
os.makedirs("out", exist_ok=True)
DB_PATH = "out/chat_memory.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            session_id TEXT,
            role TEXT,
            content TEXT,
            ts DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def save_msg(session_id: str, role: str, content: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)",
        (session_id, role, content)
    )
    conn.commit()
    conn.close()

def load_history(session_id: str, limit: int = 12) -> List[dict]:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT role, content FROM chat_history
        WHERE session_id = ?
        ORDER BY ts DESC
        LIMIT ?
    """, (session_id, limit)).fetchall()
    conn.close()
    rows.reverse()
    return [{"role": r, "content": c} for r, c in rows]

init_db()

# -------------------- In-memory cache (son Excel + tahminler) --------------------
LAST_DF = {}           # session_id -> DataFrame (ML_Proba/ML_Pred eklenmiş)
LAST_TICKER_COL = {}   # session_id -> str|None
LAST_DATE_COL = {}     # session_id -> str|None

# -------------------- ML paths --------------------
PREPROCESSOR_PATH = "out/preprocessor.joblib"
XGB_MODEL_PATH    = "out/xgb_model.json"
METRICS_PATH      = "out/ml_metrics.json"

# -------------------- ML helpers --------------------
TARGET_COL   = "Yükselme Tahmini"
EXCLUDE_COLS = {"Fiyat Değişimi", "Fiyat Değişimi (X)"}  # eğitim ve inference dışı
ID_KEYS      = ["şirket", "sirket", "ticker", "kod", "symbol", "isin"]

def to_binary(v):
    if pd.isna(v): return np.nan
    if isinstance(v, (int, np.integer)): return int(v > 0)
    if isinstance(v, float): return int(v > 0.0)
    s = str(v).strip().lower()
    if s in ["1","evet","true","pozitif","yükseldi","yukselme","y","yes"]: return 1
    if s in ["0","hayır","false","negatif","yükselmedi","n","no"]: return 0
    try: return int(float(s) > 0)
    except: return np.nan

def detect_date_col(cols):
    for c in cols:
        lc = c.lower()
        if any(k in lc for k in ["tarih","açıklanma","aciklanma","date","periyot"]):
            return c
    return None

def add_date_features(df, date_col):
    X = df.copy()
    if "periyot" in date_col.lower():
        def parse_p(v):
            s = str(v)
            if "/" in s:
                yy, qq = s.split("/", 1)
                return int(yy), int(qq)
            return None
        pq = X[date_col].apply(parse_p)
        X["_year"]    = [p[0] if p else np.nan for p in pq]
        X["_quarter"] = [p[1] if p else np.nan for p in pq]
        X.drop(columns=[date_col], inplace=True)
    else:
        X[date_col]   = pd.to_datetime(X[date_col], errors="coerce")
        X["_year"]    = X[date_col].dt.year
        X["_month"]   = X[date_col].dt.month
        X["_quarter"] = X[date_col].dt.quarter
        X.drop(columns=[date_col], inplace=True)
    return X

def split_xy_from_excel(df_raw: pd.DataFrame):
    if TARGET_COL not in df_raw.columns:
        raise RuntimeError(f"Hedef sütun '{TARGET_COL}' bulunamadı (eğitim dosyası).")
    y = df_raw[TARGET_COL].map(to_binary)
    mask = ~y.isna()
    y = y.loc[mask].astype(int)
    df = df_raw.loc[mask].copy()
    X = df.drop(columns=[TARGET_COL])

    X = X.drop(columns=[c for c in EXCLUDE_COLS if c in X.columns], errors="ignore")
    id_like = [c for c in X.columns if any(k in c.lower() for k in ID_KEYS)]
    X = X.drop(columns=id_like, errors="ignore")

    date_col = detect_date_col(df.columns)
    if date_col and date_col in X.columns:
        X = add_date_features(X, date_col)

    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if (not pd.api.types.is_numeric_dtype(X[c])) and (X[c].nunique() <= 50)]
    high_card = [c for c in X.columns if (not pd.api.types.is_numeric_dtype(X[c])) and (X[c].nunique() > 50)]
    X = X.drop(columns=high_card, errors="ignore")
    cat_cols = [c for c in cat_cols if c not in high_card]

    return X, y, num_cols, cat_cols

def build_preprocessor(num_cols, cat_cols):
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), cat_cols),
    ], remainder="drop")

def train_from_file(path: str):
    df_raw = pd.read_excel(path)
    if df_raw.empty: raise RuntimeError("Eğitim Excel'i boş.")
    X, y, num_cols, cat_cols = split_xy_from_excel(df_raw)
    pre = build_preprocessor(num_cols, cat_cols)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(sss.split(X, y))
    X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
    y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

    Xtr_np = pre.fit_transform(X_tr)
    Xte_np = pre.transform(X_te)

    pos, neg = int((y_tr==1).sum()), int((y_tr==0).sum())
    spw = (neg/pos) if pos>0 else 1.0

    tree_method = "gpu_hist"
    try:
        _ = xgb.DeviceQuantileDMatrix(Xtr_np, y_tr.values)
    except Exception:
        tree_method = "hist"

    model = xgb.XGBClassifier(
        n_estimators=800, max_depth=6, learning_rate=0.03,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        objective="binary:logistic", eval_metric="aucpr",
        tree_method=tree_method, scale_pos_weight=spw,
        random_state=42, n_jobs=0
    )
    model.fit(Xtr_np, y_tr.values)

    proba_te = model.predict_proba(Xte_np)[:,1]
    yhat_te  = (proba_te >= 0.5).astype(int)
    roc = roc_auc_score(y_te, proba_te)
    ap  = average_precision_score(y_te, proba_te)
    report = classification_report(y_te, yhat_te, output_dict=True)

    joblib.dump(pre, PREPROCESSOR_PATH)
    model.save_model(XGB_MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump({"roc_auc": float(roc), "avg_precision": float(ap), "classification_report": report},
                  f, ensure_ascii=False, indent=2)

def ensure_trained():
    need = FORCE_RETRAIN or (not os.path.exists(PREPROCESSOR_PATH) or not os.path.exists(XGB_MODEL_PATH))
    if need:
        if not os.path.exists(TRAINING_EXCEL):
            raise RuntimeError(f"Otomatik eğitim için TRAINING_EXCEL bulunamadı: {TRAINING_EXCEL}")
        train_from_file(TRAINING_EXCEL)

def load_artifacts():
    pre = joblib.load(PREPROCESSOR_PATH)
    model = xgb.XGBClassifier()
    model.load_model(XGB_MODEL_PATH)
    return pre, model

def find_col(cols, keys):
    return next((c for c in cols if any(k in c.lower() for k in keys)), None)

def build_user_text(row: dict, ticker_col=None, date_col=None):
    parts = []
    if ticker_col and ticker_col in row: parts.append(f"Ticker={row.get(ticker_col)}")
    if date_col   and date_col in row:   parts.append(f"Tarih={row.get(date_col)}")
    for label in [
        "Özkaynak Karlılığı","Aktif Karlılık","Net Kar Marjı","Brüt Kar Marjı","FAVÖK Marjı",
        "Kaldıraç Oranı","Aktif Devir Hızı","Özkaynak Devir Hızı","Kapanış Fiyatı","Fiyat Değişimi"
    ]:
        if label in row and pd.notna(row[label]):
            try: parts.append(f"{label}={float(row[label]):.3f}")
            except: parts.append(f"{label}={row[label]}")
    return "; ".join(parts)

# -------------------- LLM anti-cutoff helper --------------------
def llm_complete_full(or_client, model, system_prompt: str, history_msgs: List[dict],
                      user_text: str, temperature: float = 0.5,
                      max_tokens_per_call: int = 1000, max_rounds: int = 3) -> str:
    messages = [{"role": "system", "content": system_prompt}] + history_msgs + [
        {"role": "user", "content": user_text}
    ]
    out_parts = []
    rounds = 0

    while True:
        resp = or_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens_per_call,
        )
        choice = resp.choices[0]
        chunk = (choice.message.content or "").strip()
        if chunk:
            out_parts.append(chunk)

        finish = getattr(choice, "finish_reason", None) or getattr(choice, "finish_details", None)
        if str(finish).lower() != "length":
            break

        rounds += 1
        if rounds >= max_rounds:
            break

        messages.append({"role": "assistant", "content": chunk})
        messages.append({"role": "user", "content": "Devam et; kaldığın yerden, aynı başlık yapısıyla sürdür."})

    return "\n".join(out_parts).strip()

# -------------------- Startup --------------------
@app.on_event("startup")
def _startup():
    try:
        ensure_trained()
        print(">>> Model hazır (out/ altında).")
    except Exception as e:
        print(f">>> Uyarı: Otomatik eğitim yapılamadı: {e}")

# -------------------- Sağlık --------------------
@app.get("/")
def health():
    trained = os.path.exists(PREPROCESSOR_PATH) and os.path.exists(XGB_MODEL_PATH)
    return {"status": "ok", "trained": trained, "training_excel": TRAINING_EXCEL, "llm": bool(or_client)}

# --------------- Yardımcı: prompt'tan ticker yakalama ----------------
def find_ticker_mentions(session_id: str, text: str) -> List[str]:
    df = LAST_DF.get(session_id)
    tcol = LAST_TICKER_COL.get(session_id)
    if df is None or not tcol or tcol not in df.columns:
        return []
    # basit kelime eşleşmesi (noktalama temizle)
    chars = ",.;:!?()[]{}|/\\"
    clean = text.upper()
    for ch in chars:
        clean = clean.replace(ch, " ")
    words = {w.strip() for w in clean.split() if w.strip()}
    tickers = set(map(lambda x: str(x).upper(), df[tcol].dropna().unique()))
    return [w for w in words if w in tickers]

# -------------------- Tek endpoint: Chat + Excel + Memory --------------------
@app.post("/chat_combo")
async def chat_combo(
    session_id: str = Form(...),
    file: UploadFile = File(None),
    prompt: str = Form("")
):
    try:
        parts: List[str] = []

        # 1) Excel varsa: ML tahmini + açıklama
        if file is not None:
            ensure_trained()
            pre, model = load_artifacts()

            content = await file.read()
            df0 = pd.read_excel(io.BytesIO(content))
            if df0.empty:
                raise HTTPException(400, "Excel boş görünüyor.")

            if TARGET_COL in df0.columns:
                df0 = df0.drop(columns=[TARGET_COL])

            ticker_col = find_col(df0.columns, ID_KEYS)
            date_col_any = detect_date_col(df0.columns)

            df_tmp = df0.copy()
            df_tmp = df_tmp.drop(columns=[c for c in EXCLUDE_COLS if c in df_tmp.columns], errors="ignore")
            if date_col_any and date_col_any in df_tmp.columns:
                df_tmp = add_date_features(df_tmp, date_col_any)

            pre_num = list(pre.transformers[0][2])
            pre_cat = list(pre.transformers[1][2])
            expected = pre_num + pre_cat

            df_for_ml = df_tmp.drop(columns=[c for c in df_tmp.columns if any(k in c.lower() for k in ID_KEYS)],
                                    errors="ignore").copy()
            for c in expected:
                if c not in df_for_ml.columns:
                    df_for_ml[c] = np.nan
            df_for_ml = df_for_ml[expected].copy()

            X_np = pre.transform(df_for_ml)
            proba = model.predict_proba(X_np)[:,1]
            yhat  = (proba >= 0.5).astype(int)

            out = df0.copy()
            out["ML_Proba"] = proba
            out["ML_Pred"]  = yhat

            # cache’e koy
            LAST_DF[session_id] = out
            LAST_TICKER_COL[session_id] = ticker_col
            LAST_DATE_COL[session_id] = date_col_any

            ones = out[out["ML_Pred"]==1].copy()

            if len(ones) > 0:
                topn_names = []
                topn = ones.sort_values("ML_Proba", ascending=False).head(20)
                for _, r in topn.iterrows():
                    nm = str(r.get(ticker_col, "Seçim")) if ticker_col else "Seçim"
                    topn_names.append(nm)
                parts.append("Seçilen Hisseler\n\n" + "\n".join(topn_names))

                if or_client:
                    system_prompt = (
                        "Sadece verilen bilanço metriklerine dayanarak seçilen hisselerin neden yükselebileceğini açıkla. "
                        "Kaçın: 'genel bilgi veririm', 'güncel yorum yapamam', 'yorum yapamam' gibi ifadeler. "
                        "Başlıklar: Özet (1 cümle), Güçlü (2-3 madde), Zayıf (1-2 madde)."
                    )
                    blocks = []
                    explain_n = min(10, len(topn))
                    for i in range(explain_n):
                        r = topn.iloc[i]
                        user_text = build_user_text(r.to_dict(), ticker_col, date_col_any) or f"Ticker={str(r.get(ticker_col, '—'))}"
                        try:
                            txt = llm_complete_full(
                                or_client=or_client,
                                model=OPENROUTER_MODEL,
                                system_prompt=system_prompt,
                                history_msgs=[],  # açıklamada geçmişi kullanmıyoruz
                                user_text=user_text,
                                temperature=0.35,
                                max_tokens_per_call=1000,
                                max_rounds=3
                            )
                        except Exception as e:
                            txt = f"(LLM açıklama hatası: {e})"
                        title = str(r.get(ticker_col, "Seçim")) if ticker_col else "Seçim"
                        blocks.append(f"{title}\n{txt}")
                    parts.append("\n\n".join(blocks))
                else:
                    parts.append("(LLM anahtarı bulunamadı: Açıklama yazılamadı.)")
            else:
                parts.append("Seçilen Hisse bulunamadı.")

        # 2) Metin varsa: sohbet + (varsa) ticker bazlı açıklama
        if prompt.strip():
            save_msg(session_id, "user", prompt.strip())

            # a) Sohbet yanıtı (kısa + kesintisiz)
            if or_client:
                try:
                    history_msgs = load_history(session_id, limit=12)
                    chat_out = llm_complete_full(
                        or_client=or_client,
                        model=OPENROUTER_MODEL,
                        system_prompt="Kısa ama TAM ve kesintisiz yanıt ver.",
                        history_msgs=history_msgs,
                        user_text=prompt.strip(),
                        temperature=0.7,
                        max_tokens_per_call=1000,
                        max_rounds=3
                    )
                except Exception as e:
                    chat_out = f"(LLM sohbet hatası: {e})"
            else:
                chat_out = "(LLM anahtarı tanımlı değil — yalnızca ML tahmini çalıştı.)"
            parts.append(chat_out)

            # b) Prompt içinde hisse ismi geçtiyse: cache'ten veriyle açıklama
            mentions = find_ticker_mentions(session_id, prompt)
            if mentions and or_client and LAST_DF.get(session_id) is not None:
                df = LAST_DF[session_id]
                tcol = LAST_TICKER_COL.get(session_id)
                dcol = LAST_DATE_COL.get(session_id)
                explain_sys = (
                    "Bu bir yatırım tavsiyesi değildir. Sadece SAĞLANAN metrikleri özetle ve çıkarım yap. "
                    "Kaçın: 'genel bilgi veririm', 'güncel yorum yapamam', 'yorum yapamam' vb. "
                    "Şablon: Özet (1 cümle) • Güçlü (2-3 madde) • Zayıf (1-2 madde)."
                )
                for tk in mentions:
                    row = df.loc[df[tcol].astype(str).str.upper() == tk].head(1) if tcol else pd.DataFrame()
                    if row.empty:
                        continue
                    r = row.iloc[0].to_dict()
                    pred = int(r.get("ML_Pred", 0))
                    utext = build_user_text(r, tcol, dcol) or f"Ticker={tk}"
                    try:
                        txt = llm_complete_full(
                            or_client=or_client,
                            model=OPENROUTER_MODEL,
                            system_prompt=explain_sys,
                            history_msgs=[],    # açıklamada geçmiş yok
                            user_text=utext,
                            temperature=0.35,
                            max_tokens_per_call=1000,
                            max_rounds=3
                        )
                    except Exception as e:
                        txt = f"(LLM açıklama hatası: {e})"
                    reason = "" if pred == 1 else "\n(Not: Model bu hisse için 0 verdiği için ana listede yer almadı.)"
                    parts.append(f"{tk}\n{txt}{reason}")

            save_msg(session_id, "assistant", "\n\n".join(parts))

        if not parts:
            parts.append("Dosya veya mesaj gelmedi. Excel yükleyebilir ve/veya mesaj yazabilirsiniz.")

        return {"message": "\n\n".join(parts)}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# -------------------- CHAT UI --------------------
@app.get("/chat", response_class=HTMLResponse)
def chat_page():
    return """
<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8" />
  <title>Hisse Tahmin Uygulaması</title>
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <style>
    body { font-family: system-ui, Arial; margin:0; background:#fff; color:#111; }
    header { position: sticky; top:0; background:#fff; border-bottom:1px solid #eee; padding:12px 16px; }
    footer { position: sticky; bottom:0; background:#fff; border-top:1px solid #eee; padding:12px 16px; }
    .container { max-width:980px; margin:0 auto; }
    .messages { height: calc(100vh - 200px); overflow-y:auto; padding:16px; }
    .row { display:flex; margin:10px 0; }
    .user { justify-content:flex-end; }
    .assistant { justify-content:flex-start; }
    .bubble { max-width:80%; padding:10px 14px; border-radius:16px; box-shadow:0 1px 2px rgba(0,0,0,.06); white-space:pre-wrap; }
    .bubble.user { background:#2563eb; color:#fff; }
    .bubble.assistant { background:#f3f4f6; color:#111; }
    .controls { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
    textarea { flex:1; padding:10px 12px; border:1px solid #ddd; border-radius:12px; resize:vertical; min-height:48px; }
    button { padding:10px 14px; border:none; border-radius:12px; background:#2563eb; color:#fff; cursor:pointer; }
    .hint { color:#666; font-size:12px; margin-top:6px; }
    input[type=file] { padding:6px; }
  </style>
</head>
<body>
  <header>
    <div class="container">
      <strong>Hisse Tahmin Uygulaması</strong>
    </div>
  </header>

  <main class="container">
    <div id="messages" class="messages"></div>
  </main>

  <footer>
    <div class="container">
      <div class="controls">
        <input type="file" id="excelFile" accept=".xlsx,.xls">
        <textarea id="input" placeholder="Mesaj yazabilir ve/veya Excel seçip Gönder’e basabilirsiniz"></textarea>
        <button id="send" type="button">Gönder</button>
      </div>
      <div class="hint">Excel seçerseniz tahmin + açıklama; mesaj yazarsanız sohbet. İkisini de beraber kullanabilirsiniz.</div>
    </div>
  </footer>

<script>
  // Basit UUID
  function uuidv4() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
      const r = crypto.getRandomValues(new Uint8Array(1))[0] & 15;
      const v = c === 'x' ? r : (r & 0x3 | 0x8);
      return v.toString(16);
    });
  }

  window.addEventListener('DOMContentLoaded', () => {
    const messagesEl = document.getElementById('messages');
    const inputEl = document.getElementById('input');
    const sendBtn = document.getElementById('send');
    const excelInput = document.getElementById('excelFile');

    let sessionId = localStorage.getItem('session_id');
    if (!sessionId) { sessionId = uuidv4(); localStorage.setItem('session_id', sessionId); }

    const messages = [
      { role: "assistant", content: "Merhaba! Hisse Tahmini için Hisse Bilançolarının Bulunduğu Excel Dosyasını Yükleyin." },
    ];

    function render() {
      messagesEl.innerHTML = "";
      for (const m of messages) {
        const row = document.createElement('div');
        row.className = 'row ' + (m.role === 'user' ? 'user' : 'assistant');
        const bubble = document.createElement('div');
        bubble.className = 'bubble ' + (m.role === 'user' ? 'user' : 'assistant');
        bubble.textContent = m.content;
        row.appendChild(bubble);
        messagesEl.appendChild(row);
      }
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }
    render();

    async function send() {
      const file = excelInput.files[0];
      const text = inputEl.value.trim();
      if (!file && !text) return;

      const tag = file ? `(Excel: ${file.name})` : "";
      const combined = [text, tag].filter(Boolean).join(" — ");
      messages.push({ role: "user", content: combined || "(boş mesaj)" });
      render();

      // input’u hemen temizle
      inputEl.value = "";

      const form = new FormData();
      form.append('session_id', sessionId);
      if (file) form.append('file', file);
      form.append('prompt', text);

      try {
        const res = await fetch('/chat_combo', { method: 'POST', body: form });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || data.detail || ("HTTP "+res.status));
        messages.push({ role: "assistant", content: data.message || "Tamamlandı." });
      } catch (e) {
        messages.push({ role: "assistant", content: "(Hata: " + (e.message||e) + ")" });
      } finally {
        render();
        excelInput.value = "";  // tekrar yükleyebil
      }
    }

    sendBtn.addEventListener('click', send);
    inputEl.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
    });
  });
</script>
</body>
</html>
    """

# -------------- main --------------
if __name__ == "__main__":
    import uvicorn
    # PyCharm: Run/Debug -> Module name: uvicorn ; Parameters: app:app --reload --port 8001
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
