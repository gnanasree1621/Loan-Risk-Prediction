import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from datetime import date, datetime, timedelta
import os
from pymongo import MongoClient

st.set_page_config(
    page_title="LoanLens | Loan Risk Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)
try:
    MONGODB_URI = st.secrets["MONGODB_URI"]
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")

    db = client["loanlens"]
    predictions_collection = db["predictions"]

    st.success("MongoDB connected successfully!")

except Exception as e:
    st.warning(f"MongoDB connection failed: {e}")
MODEL_PATH = Path(__file__).parent / "loan_risk_model.pkl"
FEATURES = [
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "REGION_POPULATION_RELATIVE", "DAYS_BIRTH", "DAYS_EMPLOYED",
    "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "OWN_CAR_AGE", "CNT_CHILDREN",
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

:root { --navy:#101a33; --blue:#2563eb; --ink:#172033; --muted:#64748b; --line:#e2e8f0; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:#f4f7fc; color:var(--ink); }
[data-testid="stHeader"] { background:rgba(244,247,252,.88); }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#101a33 0%,#172d53 100%); }
[data-testid="stSidebar"] * { color:#edf2ff !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { padding:8px 10px; border-radius:10px; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background:rgba(255,255,255,.08); }
.hero {
  padding:30px 32px; border-radius:24px;
  background:linear-gradient(120deg,#101a33 0%,#173f73 55%,#2563eb 100%);
  color:white; margin:4px 0 24px; border:1px solid rgba(255,255,255,.12);
  box-shadow:0 14px 32px rgba(30,64,175,.12);
}
.hero h1 { font-family:'Manrope',sans-serif; font-size:2.15rem; line-height:1.2; font-weight:800; margin:7px 0 9px; color:white; }
.hero p { color:#dbeafe; margin:0; max-width:780px; font-size:1rem; line-height:1.6; }
.eyebrow { text-transform:uppercase; letter-spacing:.14em; font-size:.72rem; font-weight:800; color:#bfdbfe; }
.section-title { font-family:'Manrope',sans-serif; font-weight:800; color:#172033; font-size:1.22rem; margin:12px 0 5px; }
.subtle { color:#64748b; font-size:.92rem; }
div[data-testid="stMetric"] {
  background:white; border:1px solid #e3eaf5; padding:19px 21px;
  border-radius:18px; box-shadow:0 8px 24px rgba(15,23,42,.045);
}
div[data-testid="stMetricLabel"] { color:#64748b; font-weight:600; }
div[data-testid="stMetricValue"] { color:#13264a; font-family:'Manrope',sans-serif; font-weight:800; }
div[data-testid="stForm"] {
  background:white; border:1px solid #e3eaf5; border-radius:20px;
  padding:24px; box-shadow:0 8px 24px rgba(15,23,42,.035);
}
div[data-testid="stVerticalBlockBorderWrapper"] {
  border-radius:18px; border-color:#e3eaf5; background:rgba(255,255,255,.78);
}
.stButton > button, .stFormSubmitButton > button, [data-testid="stDownloadButton"] button {
  border-radius:12px; min-height:44px; font-weight:700;
  border:1px solid #d7e3f7; transition:all .15s ease;
}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
  background:linear-gradient(100deg,#1d4ed8,#2563eb) !important;
  color:white !important; border:0 !important; box-shadow:0 6px 16px rgba(37,99,235,.18);
}
.stButton > button:hover, .stFormSubmitButton > button:hover { transform:translateY(-1px); border-color:#93c5fd; }
div[data-testid="stTextInput"] input, div[data-testid="stNumberInput"] input,
div[data-testid="stDateInput"] input, div[data-testid="stSelectbox"] div {
  border-radius:10px;
}
div[data-testid="stAlert"] { border-radius:14px; }
[data-testid="stDataFrame"] { border:1px solid #e3eaf5; border-radius:12px; overflow:hidden; }
hr { border-color:#e2e8f0; }
footer { visibility:hidden; }
@media (max-width: 768px) {
  .hero { padding:22px; border-radius:18px; }
  .hero h1 { font-size:1.65rem; }
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def get_mongo_collection():
    """MongoDB is optional locally; configure MONGODB_URI in deployment secrets."""
    uri = st.secrets.get("MONGODB_URI", os.getenv("MONGODB_URI", ""))
    if not uri:
        return None
    from pymongo import MongoClient
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    db_name = st.secrets.get("MONGODB_DATABASE", os.getenv("MONGODB_DATABASE", "loanlens"))
    collection_name = st.secrets.get("MONGODB_COLLECTION", os.getenv("MONGODB_COLLECTION", "predictions"))
    return client[db_name][collection_name]

def get_collection_safe():
    try:
        return get_mongo_collection()
    except Exception as exc:
        st.warning(f"MongoDB is not connected. Showing session-only data. Details: {exc}")
        return None

def hero(kicker, title, description):
    st.markdown(
        f'<div class="hero"><div class="eyebrow">{kicker}</div>'
        f'<h1>{title}</h1><p>{description}</p></div>',
        unsafe_allow_html=True,
    )

def relative_days(event_date, application_date):
    return (event_date - application_date).days

def risk_label(prediction):
    return "Higher default risk" if int(prediction) == 1 else "Lower default risk"

def safe_records(collection, limit=500):
    if collection is None:
        return st.session_state.get("history", [])
    docs = list(collection.find({}, {"_id": 0}).sort("created_at", -1).limit(limit))
    for doc in docs:
        if "created_at" in doc and hasattr(doc["created_at"], "strftime"):
            doc["Date"] = doc["created_at"].strftime("%Y-%m-%d %H:%M")
        doc.pop("created_at", None)
    return docs

if "history" not in st.session_state:
    st.session_state.history = []

try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.info("Make sure loan_risk_model.pkl is in the same folder as app.py.")
    st.stop()

collection = get_collection_safe()

st.sidebar.markdown("## 🏦 LoanLens")
st.sidebar.caption("LOAN RISK INTELLIGENCE")
page = st.sidebar.radio("NAVIGATION", ["Dashboard", "Predict Risk", "Prediction History", "About"])
st.sidebar.markdown("---")
if collection is not None:
    st.sidebar.success("MongoDB connected")
else:
    st.sidebar.info("Session-only mode")
st.sidebar.caption("Educational prototype · Not a lending decision")

# ---------------- Dashboard ----------------
if page == "Dashboard":
    hero("CREDIT RISK ANALYSIS", "Loan Risk Prediction",
         "A clean workspace to estimate loan-default risk from applicant and financial indicators.")

    records = safe_records(collection)
    total = len(records)
    higher = sum(1 for r in records if r.get("Prediction") == "Higher default risk")
    lower = sum(1 for r in records if r.get("Prediction") == "Lower default risk")
    probs = [float(r["Default probability"]) for r in records if r.get("Default probability") is not None]
    avg_prob = sum(probs) / len(probs) if probs else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Applicants analyzed", total)
    c2.metric("Higher-risk predictions", higher)
    c3.metric("Lower-risk predictions", lower)
    c4.metric("Average default probability", f"{avg_prob:.1%}" if avg_prob is not None else "—")

    left, right = st.columns([1.1, 1])
    with left:
        with st.container(border=True):
            st.markdown("### 🧠 Predict loan risk")
            st.write("Enter applicant financial and profile details. No applicant name is required.")
            st.info("Use Predict Risk from the navigation menu to run an analysis.")
    with right:
        with st.container(border=True):
            st.markdown("### 📊 Recent activity")
            if records:
                df = pd.DataFrame(records[:5])
                cols = [c for c in ["Date", "Prediction", "Default probability"] if c in df.columns]
                st.dataframe(df[cols], use_container_width=True, hide_index=True)
            else:
                st.info("No predictions recorded yet. Run your first analysis from Predict Risk.")

    if records:
        st.markdown('<div class="section-title">Prediction distribution</div>', unsafe_allow_html=True)
        st.bar_chart(pd.Series([r.get("Prediction", "Unknown") for r in records]).value_counts())

# ---------------- Predict Risk ----------------
elif page == "Predict Risk":
    hero("APPLICANT ASSESSMENT", "Predict loan-default risk",
         "Enter the required applicant indicators. Calendar dates are converted to the day-based format expected by the model.")

    with st.form("loan_form"):
        st.markdown('<div class="section-title">1. Application and financial information</div>', unsafe_allow_html=True)
        c0, c1, c2 = st.columns(3)
        with c0:
            application_date = st.date_input("Loan application date", value=date.today(), max_value=date.today())
            annual_income = st.number_input("Annual income", min_value=0.0, value=180000.0, step=10000.0)
        with c1:
            credit_amount = st.number_input("Credit amount", min_value=0.0, value=500000.0, step=10000.0)
            annual_annuity = st.number_input("Annual annuity", min_value=0.0, value=25000.0, step=1000.0)
        with c2:
            goods_price = st.number_input("Goods price", min_value=0.0, value=450000.0, step=10000.0)
            region_population = st.number_input("Region population relative", min_value=0.0, value=0.018801, format="%.6f")

        st.markdown("---")
        st.markdown('<div class="section-title">2. Applicant profile</div>', unsafe_allow_html=True)
        st.caption("Choose normal calendar dates; the application converts them automatically. No name, phone number, or address is collected.")
        application_date = min(application_date, date.today())
        birth_default = min(date(1993, 1, 1), application_date - timedelta(days=365))
        employment_default = min(date(2018, 1, 1), application_date)
        registration_default = min(date(2015, 1, 1), application_date)
        id_default = min(date(2020, 1, 1), application_date)

        p1, p2 = st.columns(2)
        with p1:
            dob = st.date_input("Date of birth", value=birth_default, min_value=date(1920, 1, 1), max_value=application_date)
            employment_start = st.date_input("Employment start date", value=employment_default, min_value=date(1950, 1, 1), max_value=application_date)
        with p2:
            registration_date = st.date_input("Registration date", value=registration_default, min_value=date(1950, 1, 1), max_value=application_date)
            id_date = st.date_input("ID publication date", value=id_default, min_value=date(1950, 1, 1), max_value=application_date)

        p3, p4 = st.columns(2)
        with p3:
            car_age = st.number_input("Own car age (years)", min_value=0.0, value=5.0, step=1.0)
        with p4:
            children = st.number_input("Number of children", min_value=0, value=0, step=1)

        st.markdown("---")
        st.markdown('<div class="section-title">3. External assessment indicators</div>', unsafe_allow_html=True)
        st.caption("Use actual values when available. These sample defaults are for demonstration only.")
        e1, e2, e3 = st.columns(3)
        with e1:
            ext1 = st.number_input("External source 1", min_value=0.0, max_value=1.0, value=0.45, step=0.01, format="%.4f")
        with e2:
            ext2 = st.number_input("External source 2", min_value=0.0, max_value=1.0, value=0.60, step=0.01, format="%.4f")
        with e3:
            ext3 = st.number_input("External source 3", min_value=0.0, max_value=1.0, value=0.50, step=0.01, format="%.4f")

        submitted = st.form_submit_button("🔍 Predict loan risk", use_container_width=True, type="primary")

    if submitted:
        if dob >= application_date:
            st.error("Date of birth must be earlier than the application date.")
        elif any(d > application_date for d in [employment_start, registration_date, id_date]):
            st.error("Employment, registration, and ID publication dates must not be after the application date.")
        else:
            values = {
                "AMT_INCOME_TOTAL": annual_income,
                "AMT_CREDIT": credit_amount,
                "AMT_ANNUITY": annual_annuity,
                "AMT_GOODS_PRICE": goods_price,
                "REGION_POPULATION_RELATIVE": region_population,
                "DAYS_BIRTH": relative_days(dob, application_date),
                "DAYS_EMPLOYED": relative_days(employment_start, application_date),
                "DAYS_REGISTRATION": relative_days(registration_date, application_date),
                "DAYS_ID_PUBLISH": relative_days(id_date, application_date),
                "OWN_CAR_AGE": car_age,
                "CNT_CHILDREN": int(children),
                "EXT_SOURCE_1": ext1,
                "EXT_SOURCE_2": ext2,
                "EXT_SOURCE_3": ext3,
            }
            try:
                frame = pd.DataFrame([values], columns=FEATURES)
                prediction = int(model.predict(frame)[0])
                if not hasattr(model, "predict_proba"):
                    st.error("This model does not provide predict_proba, so a probability cannot be displayed.")
                else:
                    probability = float(model.predict_proba(frame)[0][1])
                    label = risk_label(prediction)
                    record = {
                        "Prediction": label,
                        "Default probability": probability,
                        "Annual income": annual_income,
                        "Credit amount": credit_amount,
                        "created_at": datetime.utcnow(),
                    }

                    if collection is not None:
                        try:
                            collection.insert_one(record.copy())
                        except Exception as exc:
                            st.error(f"Prediction succeeded but could not be saved to MongoDB: {exc}")
                            st.session_state.history.append({
                                **record,
                                "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                            })
                    else:
                        st.session_state.history.append({
                            **record,
                            "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        })

                    st.markdown("---")
                    st.markdown('<div class="section-title">Prediction result</div>', unsafe_allow_html=True)
                    r1, r2 = st.columns([1.1, 1])
                    with r1:
                        if prediction == 1:
                            st.error("### Higher default risk")
                            st.write("The model assigns this input to the higher-risk class. This is an estimate, not a final lending decision.")
                        else:
                            st.success("### Lower default risk")
                            st.write("The model assigns this input to the lower-risk class. This does not guarantee repayment or loan approval.")
                    with r2:
                        st.metric("Estimated default probability", f"{probability:.1%}")
                        st.progress(min(max(probability, 0.0), 1.0), text="Model-estimated probability for class 1")
                    with st.expander("View applicant details and model values"):
                        # Show friendly dates to the user, while preserving the numeric day values used by the model.
                        display_values = {
                            "Annual income": annual_income,
                            "Credit amount": credit_amount,
                            "Annual annuity": annual_annuity,
                            "Goods price": goods_price,
                            "Region population relative": region_population,
                            "Date of birth": dob.strftime("%d-%m-%Y"),
                            "Employment start date": employment_start.strftime("%d-%m-%Y"),
                            "Registration date": registration_date.strftime("%d-%m-%Y"),
                            "ID publication date": id_date.strftime("%d-%m-%Y"),
                            "Own car age (years)": car_age,
                            "Number of children": int(children),
                            "External source 1": ext1,
                            "External source 2": ext2,
                            "External source 3": ext3,
                        }
                        st.dataframe(
                            pd.DataFrame([display_values]),
                            use_container_width=True,
                            hide_index=True
                        )
                        with st.expander("Technical model values (relative days)"):
                            st.caption("The model requires relative-day numbers internally. These are kept technical and are not shown as the main applicant dates.")
                            st.dataframe(frame, use_container_width=True, hide_index=True)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                st.caption("Confirm that the model's feature names, order, preprocessing, and package versions match training.")

# ---------------- History ----------------
elif page == "Prediction History":
    hero("SAVED RECORDS", "Prediction history",
         "Review and export recorded predictions. MongoDB provides persistent storage when configured.")
    records = safe_records(collection)
    if records:
        df = pd.DataFrame(records)
        c1, c2 = st.columns(2)
        c1.metric("Recorded predictions", len(df))
        if "Default probability" in df.columns:
            c2.metric("Average default probability", f"{pd.to_numeric(df['Default probability'], errors='coerce').mean():.1%}")
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("⬇ Download history as CSV", data=df.to_csv(index=False).encode("utf-8"),
                           file_name="loan_prediction_history.csv", mime="text/csv")
        if collection is not None:
            st.caption("Records are stored in MongoDB. No applicant name or contact information is saved by this app.")
        else:
            st.caption("MongoDB is not configured. These records are temporary and will not persist after the session ends.")
    else:
        st.info("No predictions have been recorded yet. Go to Predict Risk and submit an applicant profile.")

# ---------------- About ----------------
elif page == "About":
    hero("PROJECT OVERVIEW", "About LoanLens",
         "A machine-learning prototype for exploring loan-default risk from structured applicant data.")
    a, b = st.columns(2)
    with a:
        with st.container(border=True):
            st.markdown("### 🎯 Objective")
            st.write("Estimate whether a set of applicant indicators resembles the lower-risk or higher-risk class learned by the trained model.")
            st.markdown("### 🧩 Modules")
            st.write("Dashboard · Predict Risk · Prediction History · About")
    with b:
        with st.container(border=True):
            st.markdown("### ⚙️ Technology stack")
            st.write("Python, Streamlit, Pandas, Joblib, XGBoost-compatible model and MongoDB for saved prediction records.")
            st.markdown("### 🔒 Privacy")
            st.write("The app does not ask for applicant names, phone numbers, or addresses. It stores model inputs only in memory for prediction and saves the selected summary fields to the database.")
            st.markdown("### ⚠️ Responsible use")
            st.write("This educational prototype is not a final lending decision. Validate the model before any real-world use.")
