import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from preprocessing import preprocess_data
from sklearn.preprocessing import FunctionTransformer
import os

# Load data
file_path = os.path.join(os.path.dirname(__file__), "../data/heart.csv")
df = pd.read_csv(file_path)

# Wrap custom preprocessing logic
feature_engineering = FunctionTransformer(preprocess_data)

# Build pipeline
numeric = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
categorical = [col for col in df.columns if col not in numeric]
clf = Pipeline([
    ("feature_engineering", feature_engineering),
    ("model", XGBClassifier(random_state=42, eval_metric="logloss"))
])

# Train/test split
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set known best parameters
clf.set_params(
    model__max_depth=3,
    model__learning_rate=0.05,
    model__n_estimators=100,
    model__colsample_bytree=0.8,
    model__subsample=1.0
)

# Train model
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
os.makedirs(os.path.dirname("../model/model.pkl"), exist_ok=True)
joblib.dump(clf, "../model/model.pkl")
