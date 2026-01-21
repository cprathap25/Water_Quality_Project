from src.data.load_data import load_data
from src.data.clean_data import clean_data
from src.features.risk_score import classify_water_quality
from src.models.train_model import train_model

def main():
    df = load_data("data/raw/water_quality_india.csv")
    df = clean_data(df)

    df[['water_status', 'risk_score']] = df.apply(
        lambda row: classify_water_quality(row),
        axis=1,
        result_type='expand'
    )

    df.to_csv("data/processed/cleaned_water_data.csv", index=False)

    # Train model (accuracy printed INSIDE train_model)
    train_model(df)

    print("Pipeline completed successfully")

if __name__ == "__main__":
    main()
