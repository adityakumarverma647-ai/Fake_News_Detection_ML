import streamlit as st
import pickle
import re
import time


# =============================================================

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# LOAD MODEL AND VECTORIZER
# ============================================================

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    with open("vectorizer.pkl", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    return model, vectorizer


try:
    model, vectorizer = load_model()
except FileNotFoundError:
    st.error(
        "Model files not found. Please run train_model.py first "
        "to generate model.pkl and vectorizer.pkl."
    )
    st.stop()


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background: linear-gradient(
            135deg,
            #0f172a 0%,
            #111827 50%,
            #020617 100%
        );
        color: #f8fafc;
    }

    /* Remove default top padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Hero section */
    .hero {
        padding: 35px;
        border-radius: 24px;
        background: linear-gradient(
            135deg,
            #1e293b,
            #172554
        );
        border: 1px solid #334155;
        margin-bottom: 30px;
        box-shadow: 0 15px 40px rgba(0,0,0,0.25);
    }

    .hero h1 {
        font-size: 3rem;
        margin-bottom: 10px;
        color: #ffffff;
    }

    .hero p {
        font-size: 1.1rem;
        color: #cbd5e1;
        line-height: 1.7;
    }

    .badge {
        display: inline-block;
        padding: 7px 14px;
        border-radius: 20px;
        background: #1d4ed8;
        color: white;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 15px;
    }

    /* Cards */
    .card {
        background: rgba(30, 41, 59, 0.85);
        border: 1px solid #334155;
        border-radius: 20px;
        padding: 25px;
        margin-bottom: 20px;
    }

    .card h3 {
        color: #ffffff;
        margin-bottom: 8px;
    }

    .card p {
        color: #cbd5e1;
        line-height: 1.6;
    }

    /* Prediction */
    .prediction-real {
        padding: 30px;
        border-radius: 20px;
        background: rgba(22, 101, 52, 0.25);
        border: 1px solid #22c55e;
        text-align: center;
        margin-top: 20px;
    }

    .prediction-fake {
        padding: 30px;
        border-radius: 20px;
        background: rgba(127, 29, 29, 0.25);
        border: 1px solid #ef4444;
        text-align: center;
        margin-top: 20px;
    }

    .prediction-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: white;
    }

    .confidence {
        font-size: 1.1rem;
        color: #cbd5e1;
        margin-top: 8px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #020617;
        border-right: 1px solid #1e293b;
    }

    section[data-testid="stSidebar"] h2 {
        color: white;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        padding: 12px;
        font-weight: 700;
    }

    /* Text area */
    textarea {
        border-radius: 15px !important;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 25px;
        margin-top: 40px;
        color: #94a3b8;
        border-top: 1px solid #334155;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 📰 Fake News AI")

    st.markdown("---")

    st.markdown("### 🤖 Model")

    st.info(
        "Machine Learning model trained using "
        "TF-IDF and Logistic Regression."
    )

    st.markdown("### 📊 Performance")

    st.metric(
        label="Model Accuracy",
        value="95.46%"
    )

    st.markdown("---")

    st.markdown("### 🛠️ Technologies")

    st.markdown(
        """
        - Python
        - Pandas
        - NumPy
        - Scikit-learn
        - TF-IDF
        - Logistic Regression
        - Streamlit
        """
    )

    st.markdown("---")

    st.caption(
        "AI-powered educational project for detecting "
        "potentially fake news."
    )


# ============================================================
# HERO SECTION
# ============================================================

# st.markdown(
#     """
#     <div class="hero">

#         <div class="badge">AI / MACHINE LEARNING PROJECT</div>

#         <h1>📰 Fake News Detector</h1>

#         <p>
#             Analyze news content using Natural Language Processing
#             and Machine Learning to determine whether an article
#             is likely to be <b>REAL</b> or <b>FAKE</b>.
#         </p>

#     </div>
#     """,
#     unsafe_allow_html=True
# )


# ============================================================
# INTRODUCTION CARDS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="card">
            <h3>🧠 AI Powered</h3>
            <p>
                Uses machine learning and natural language
                processing to analyze news content.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="card">
            <h3>⚡ Fast Prediction</h3>
            <p>
                Enter an article and receive a prediction
                within seconds.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <div class="card">
            <h3>📊 95.46% Accuracy</h3>
            <p>
                Logistic Regression model trained on the
                WELFake news dataset.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# NEWS INPUT
# ============================================================

st.markdown("## 🔍 Analyze News")

st.markdown(
    "Paste a news headline or article below and let the model analyze it."
)

news_text = st.text_area(
    "News Article",
    height=250,
    placeholder=(
        "Paste the headline or full news article here..."
    ),
    label_visibility="collapsed"
)


# ============================================================
# BUTTON
# ============================================================

analyze_button = st.button(
    "🔎 Analyze News",
    type="primary"
)


# ============================================================
# PREDICTION
# ============================================================

if analyze_button:

    if not news_text.strip():

        st.warning(
            "⚠️ Please enter some news text before analyzing."
        )

    elif len(news_text.strip()) < 20:

        st.warning(
            "⚠️ Please enter a longer piece of news text "
            "for a more meaningful prediction."
        )

    else:

        with st.spinner("Analyzing news content..."):
            time.sleep(1)

            # Convert text to TF-IDF
            text_vector = vectorizer.transform([news_text])

            # Prediction
            prediction = model.predict(text_vector)[0]

            # Prediction probability
            probabilities = model.predict_proba(text_vector)[0]

            confidence = max(probabilities) * 100

        # WELFake:
        # 0 = Fake
        # 1 = Real

        if prediction == 1:

            st.markdown(
                f"""
                <div class="prediction-real">

                    <div class="prediction-title">
                        ✅ REAL NEWS
                    </div>

                    <div class="confidence">
                        Model Confidence: {confidence:.2f}%
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.success(
                "The model classified this news as REAL."
            )

        else:

            st.markdown(
                f"""
                <div class="prediction-fake">

                    <div class="prediction-title">
                        🚨 FAKE NEWS
                    </div>

                    <div class="confidence">
                        Model Confidence: {confidence:.2f}%
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.error(
                "The model classified this news as FAKE."
            )

        # ====================================================
        # PROBABILITY DETAILS
        # ====================================================

        st.markdown("### 📊 Prediction Probability")

        probability_col1, probability_col2 = st.columns(2)

        fake_probability = probabilities[0] * 100
        real_probability = probabilities[1] * 100

        with probability_col1:

            st.metric(
                "🚨 Fake Probability",
                f"{fake_probability:.2f}%"
            )

            st.progress(
                int(fake_probability)
            )

        with probability_col2:

            st.metric(
                "✅ Real Probability",
                f"{real_probability:.2f}%"
            )

            st.progress(
                int(real_probability)
            )


# ============================================================
# HOW IT WORKS
# ============================================================

st.markdown("---")

st.markdown("## ⚙️ How It Works")

step1, step2, step3, step4 = st.columns(4)

with step1:
    st.markdown(
        """
        <div class="card">
            <h3>01</h3>
            <h3>📝 Input</h3>
            <p>News text is provided to the application.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with step2:
    st.markdown(
        """
        <div class="card">
            <h3>02</h3>
            <h3>🔤 TF-IDF</h3>
            <p>
                Text is converted into numerical
                features using TF-IDF.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with step3:
    st.markdown(
        """
        <div class="card">
            <h3>03</h3>
            <h3>🤖 ML Model</h3>
            <p>
                Logistic Regression analyzes the
                processed text.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with step4:
    st.markdown(
        """
        <div class="card">
            <h3>04</h3>
            <h3>🎯 Result</h3>
            <p>
                The system predicts whether the news
                is REAL or FAKE.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# IMPORTANT DISCLAIMER
# ============================================================

st.markdown("---")

st.warning(
    "⚠️ This application is an educational machine learning "
    "project. Predictions are model-based and should not be "
    "treated as definitive proof that a news article is real "
    "or fake. Always verify important information using "
    "reliable sources."
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        <b>Fake News Detection using Machine Learning</b>
        <br>
        Built with Python • Scikit-learn • TF-IDF • Streamlit
        <br><br>
        AI / ML Project
    </div>
    """,
    unsafe_allow_html=True
)