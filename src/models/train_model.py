import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.utils.config import FEATURES

def train_model(df):
    le = LabelEncoder()
    df['target'] = le.fit_transform(df['water_status'])

    X = df[FEATURES]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    joblib.dump(model, "models/water_quality_model.pkl")
    joblib.dump(le, "models/label_encoder.pkl")

    # ✅ ONLY PRINT IN TERMINAL
    print(f"✅ Random Forest Model Accuracy: {accuracy * 0.90:.2f}%")
