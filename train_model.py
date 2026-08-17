import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ==========================================
# 1. LOAD DATASET
# ==========================================

DATA_PATH = "dataset/WELFake_Dataset.csv"

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset loaded successfully!")
print(f"Total records: {len(df)}")
print("\nColumns:")
print(df.columns.tolist())


# ==========================================
# 2. CHECK REQUIRED COLUMNS
# ==========================================

required_columns = ["title", "text", "label"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Required column '{column}' not found in dataset.")


# ==========================================
# 3. REMOVE MISSING VALUES
# ==========================================

df = df.dropna(subset=["title", "text", "label"])

print(f"\nRecords after removing missing values: {len(df)}")


# ==========================================
# 4. COMBINE TITLE AND TEXT
# ==========================================

df["content"] = df["title"].astype(str) + " " + df["text"].astype(str)


# ==========================================
# 5. FEATURES AND TARGET
# ==========================================

X = df["content"]
y = df["label"]


# ==========================================
# 6. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nData split completed.")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# ==========================================
# 7. TF-IDF VECTORIZATION
# ==========================================

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7,
    max_features=100000,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF transformation completed.")


# ==========================================
# 8. TRAIN LOGISTIC REGRESSION MODEL
# ==========================================

print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_tfidf, y_train)

print("Model training completed.")


# ==========================================
# 9. MODEL PREDICTION
# ==========================================

y_pred = model.predict(X_test_tfidf)


# ==========================================
# 10. MODEL EVALUATION
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n========================================")
print("MODEL PERFORMANCE")
print("========================================")

print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ==========================================
# 11. SAVE MODEL
# ==========================================

print("\nSaving trained model...")

with open("model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)


# ==========================================
# 12. SAVE TF-IDF VECTORIZER
# ==========================================

print("Saving TF-IDF vectorizer...")

with open("vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)


# ==========================================
# 13. COMPLETION MESSAGE
# ==========================================

print("\n========================================")
print("TRAINING COMPLETED SUCCESSFULLY!")
print("========================================")

print("Created files:")
print("1. model.pkl")
print("2. vectorizer.pkl")