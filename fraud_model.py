import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Step 1: Load dataset (replace with your dataset path)
df = pd.read_csv("creditcard.csv")  # Download from Kaggle

# Step 2: Split features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Step 3: Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 4: Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 5: Evaluate
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("📊 Report:\n", classification_report(y_test, y_pred))

# Step 6: Save model
joblib.dump(model, "fraud_model.pkl")
print("✅ Model saved as fraud_model.pkl")

# Step 7: Demo prediction
sample = X_test.iloc[0].values.reshape(1, -1)
print("🔎 Sample Prediction (0=Not Fraud, 1=Fraud):", model.predict(sample))
