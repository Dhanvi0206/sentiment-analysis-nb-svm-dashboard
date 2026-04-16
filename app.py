import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="💬",
    layout="wide"
)

# =====================================================
# PATHS
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# =====================================================
# LOAD MODELS
# =====================================================
try:
    nb = pickle.load(open(os.path.join(MODELS_DIR, "nb_model.pkl"), "rb"))
    svm = pickle.load(open(os.path.join(MODELS_DIR, "svm_model.pkl"), "rb"))
    tfidf = pickle.load(open(os.path.join(MODELS_DIR, "tfidf.pkl"), "rb"))
    metrics = pickle.load(open(os.path.join(MODELS_DIR, "metrics.pkl"), "rb"))
except Exception:
    st.error("❌ Models not found. Run training first.")
    st.stop()

# =====================================================
# SAFE METRICS
# =====================================================
nb_acc = metrics.get("nb_accuracy", 0)
svm_acc = metrics.get("svm_accuracy", 0)
cm_nb = metrics.get("cm_nb")
cm_svm = metrics.get("cm_svm")
nb_report = metrics.get("nb_report")
svm_report = metrics.get("svm_report")

# =====================================================
# HELPER FUNCTION (IMPORTANT FIX)
# =====================================================
def map_prediction(pred):
    """Handles both numeric and text labels"""
    if pred == 1 or str(pred).lower() == "positive":
        return "Positive"
    else:
        return "Negative"

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("📌 Navigation")
section = st.sidebar.radio(
    "Choose a section",
    ["🏠 Live Demo", "📊 Model Performance", "ℹ️ About Project"]
)

st.sidebar.markdown("---")
st.sidebar.write("Built with  using Python & Streamlit")

# =====================================================
# LIVE DEMO
# =====================================================
if section == "🏠 Live Demo":
    st.title("🔍 Live Sentiment Prediction")

    user_input = st.text_area(
        "Enter your sentence:",
        height=120,
        placeholder="Example: I absolutely loved this movie!"
    )

    if st.button("Predict Sentiment"):
        if not user_input.strip():
            st.warning("⚠️ Please enter some text.")
        else:
            vec = tfidf.transform([user_input])

            nb_pred = nb.predict(vec)[0]
            svm_pred = svm.predict(vec)[0]

            # FIXED mapping
            nb_result = map_prediction(nb_pred)
            svm_result = map_prediction(svm_pred)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Naive Bayes")
                if nb_result == "Positive":
                    st.success("🟢 Positive")
                else:
                    st.error("🔴 Negative")

            with col2:
                st.subheader("SVM (LinearSVC)")
                if svm_result == "Positive":
                    st.success("🟢 Positive")
                else:
                    st.error("🔴 Negative")

# =====================================================
# MODEL PERFORMANCE
# =====================================================
elif section == "📊 Model Performance":
    st.title("📈 Model Performance Dashboard")

    st.subheader("🎯 Accuracy Overview")
    st.write(f"Naive Bayes: {nb_acc:.4f}")
    st.progress(float(nb_acc))

    st.write(f"SVM: {svm_acc:.4f}")
    st.progress(float(svm_acc))

    st.markdown("---")

    col1, col2 = st.columns([1, 2])

    with col1:
        acc_df = pd.DataFrame({
            "Model": ["Naive Bayes", "SVM"],
            "Accuracy": [nb_acc, svm_acc]
        }).set_index("Model")
        st.dataframe(acc_df)

    with col2:
        fig, ax = plt.subplots()
        ax.bar(["Naive Bayes", "SVM"], [nb_acc, svm_acc])
        ax.set_ylim(0, 1)
        st.pyplot(fig)

    st.markdown("---")

    st.subheader("📌 Confusion Matrices")

    col_cm1, col_cm2 = st.columns(2)

    with col_cm1:
        fig_nb, ax_nb = plt.subplots()
        sns.heatmap(cm_nb, annot=True, fmt="d", cmap="Blues")
        st.pyplot(fig_nb)

    with col_cm2:
        fig_svm, ax_svm = plt.subplots()
        sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Greens")
        st.pyplot(fig_svm)

# =====================================================
# ABOUT
# =====================================================
else:
    st.title("ℹ️ Project Overview")
    st.write("""
    This project performs sentiment analysis using:
    - Naive Bayes
    - Support Vector Machine (SVM)

    Text is converted using TF-IDF and evaluated using:
    - Accuracy
    - Precision
    - Recall
    - F1-score
    """)

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.markdown(
    "<div style='text-align:center;'>Built with  using Streamlit</div>",
    unsafe_allow_html=True
)