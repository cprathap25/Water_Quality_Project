import joblib

model = joblib.load("models/water_quality_model.pkl")
le = joblib.load("models/label_encoder.pkl")

def predict_water_quality(data):
    pred = model.predict(data)
    return le.inverse_transform(pred)[0]
