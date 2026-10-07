from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import joblib

SAVED_MODEL = "data/model.pkl"

# Load dataset
X, y = load_iris(return_X_y=True)

# Train simple model
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Save model
joblib.dump(model, SAVED_MODEL)
print(f"✅ Model trained and saved as {SAVED_MODEL}")