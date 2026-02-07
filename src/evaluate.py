import joblib
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from preprocess import load_data

def evaluate_model():
    # Load model
    model = joblib.load("model/random_forest_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    
    # Load data
    X_scaled, y, _ = load_data()
    
    # Evaluate on full dataset (for simplicity)
    y_pred = model.predict(X_scaled)
    
    mae = mean_absolute_error(y, y_pred)
    rmse = root_mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.2f}")
    
    # # Feature importance
    # features = ['NDVI', 'GNDVI', 'NDWI', 'SAVI', 'soil_moisture', 'temperature', 'rainfall']
    # importances = model.feature_importances_
    
    # for feat, imp in zip(features, importances):
    #     print(f"{feat}: {imp:.3f}")

    # Feature importance
    features = ['NDVI', 'GNDVI', 'NDWI', 'SAVI', 'soil_moisture', 'temperature', 'rainfall', 'Pesticide_Amount_L_per_ha']
    importances = model.feature_importances_

    for feat, imp in zip(features, importances):
        print(f"{feat}: {imp:.3f}")


if __name__ == "__main__":
    evaluate_model()
