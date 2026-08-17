# 📰 Fake News Detection Using Machine Learning

A Machine Learning and Natural Language Processing (NLP) application that analyzes news content and predicts whether it is **Real** or **Fake**.

The project uses **TF-IDF Vectorization** to convert news text into numerical features and a **Logistic Regression** classifier to perform the final prediction. A Streamlit web application provides an interactive interface for users to enter news content and receive predictions.

---

## 🎯 Project Objective

The main objective of this project is to develop an AI/ML-based system capable of identifying potentially fake news using textual features from news articles.

The system demonstrates a complete Machine Learning workflow:

**Dataset → Data Cleaning → Text Processing → TF-IDF → Model Training → Evaluation → Prediction → Web Application**

---

## 🚀 Features

* 📰 News article analysis
* 🤖 Machine Learning-based prediction
* 🔤 TF-IDF text vectorization
* 📊 Logistic Regression classification
* 🎯 Real/Fake prediction
* 📈 Prediction probability
* 📋 Classification report
* ⚡ Fast predictions
* 🌐 Interactive Streamlit web interface
* 🎨 Professional dark-themed UI
* ⚠️ Educational prediction disclaimer

---

## 📊 Dataset

This project uses the **WELFake Dataset** obtained from Kaggle.

The dataset contains approximately **72,000 news articles** with labels indicating whether the news is real or fake.

### Dataset Columns

| Column       | Description          |
| ------------ | -------------------- |
| `title`      | News article title   |
| `text`       | Full news article    |
| `label`      | Target variable      |
| `Unnamed: 0` | Dataset index column |

### Label Meaning

```text
0 → Fake News
1 → Real News
```

---

## 🧠 Machine Learning Methodology

### 1. Data Loading

The dataset is loaded using Pandas.

### 2. Data Cleaning

Rows containing missing values in the title, text, or label columns are removed.

### 3. Text Combination

The news title and article text are combined into a single text feature.

### 4. TF-IDF Vectorization

**Term Frequency-Inverse Document Frequency (TF-IDF)** converts textual information into numerical features that can be processed by the Machine Learning model.

The project uses:

* English stop-word removal
* Unigram and bigram features
* Maximum document frequency filtering
* Up to 100,000 features

### 5. Train-Test Split

The dataset is divided into:

* **80% Training Data**
* **20% Testing Data**

### 6. Model Training

The classification model used is:

**Logistic Regression**

### 7. Model Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

---

## 📈 Model Performance

The trained Logistic Regression model achieved:

### **95.46% Accuracy**

Classification performance:

| Class    | Precision | Recall | F1-Score |
| -------- | --------: | -----: | -------: |
| Fake (0) |      0.96 |   0.95 |     0.95 |
| Real (1) |      0.95 |   0.96 |     0.96 |

### Confusion Matrix

```text
[[6625  381]
 [ 269 7033]]
```

The model was trained using **57,229 samples** and evaluated on **14,308 test samples** after data cleaning.

---

## 🖥️ Application Workflow

```text
User enters news
       ↓
Text preprocessing
       ↓
TF-IDF Vectorization
       ↓
Logistic Regression Model
       ↓
Prediction Probability
       ↓
REAL / FAKE Result
```

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **TF-IDF**
* **Logistic Regression**
* **Streamlit**
* **Pickle**
* **GitHub**

---

## 📁 Project Structure

```text
Fake_News_Detection/
│
├── app.py
├── train_model.py
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
├── README.md
│
├── dataset/
│   └── WELFake_Dataset.csv
│
├── screenshots/
│
└── assets/
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Open the project folder

```bash
cd Fake_News_Detection
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

### Windows

```bash
.venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

If the trained model files are already available, run:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🧪 Training the Model

To retrain the Machine Learning model:

```bash
python train_model.py
```

This generates:

```text
model.pkl
vectorizer.pkl
```

These files are then used by the Streamlit application for predictions.

---

## 📷 Screenshots

Add project screenshots inside the `screenshots/` folder.

Recommended screenshots:

```text
screenshots/
├── home.png
├── real_prediction.png
├── fake_prediction.png
└── probability.png
```

Then add them to this README when the screenshots are ready.

---

## 🎓 Learning Outcomes

Through this project, I gained practical experience in:

* Data preprocessing
* Exploratory understanding of datasets
* Natural Language Processing
* TF-IDF feature extraction
* Text classification
* Logistic Regression
* Model evaluation
* Confusion matrix analysis
* Model serialization using Pickle
* Streamlit application development
* Integrating Machine Learning with a web interface

---

## ⚠️ Disclaimer

This project is developed for **educational and demonstration purposes**.

The prediction generated by the Machine Learning model should not be considered definitive proof that a news article is real or fake. Important information should always be verified through reliable and trusted sources.

---

## 👨‍💻 Author

**Aditya Kumar Verma**

B.Tech — Computer Science & Engineering (AI)

---

## ⭐ Project Highlights

**Dataset:** WELFake
**Model:** Logistic Regression
**Feature Extraction:** TF-IDF
**Accuracy:** **95.46%**
**Interface:** Streamlit
**Project Type:** AI / Machine Learning / NLP
