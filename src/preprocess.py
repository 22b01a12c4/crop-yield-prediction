import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path="data/crop_data.csv", crop="Rice", use_saved_scaler=None):
    df = pd.read_csv(path)
    
    # Drop unnecessary columns
    df = df.drop(columns=[col for col in df.columns if "Unnamed" in col], errors='ignore')
    
    # Filter by crop type
    df = df[df['crop_type'] == crop].copy()
    
    # Handle missing values
    df.fillna(df.median(numeric_only=True), inplace=True)
    
    # Select features and target
    features = ['NDVI', 'GNDVI', 'NDWI', 'SAVI', 'soil_moisture', 'temperature', 'rainfall', 'Pesticide_Amount_L_per_ha']
    target = 'yield'
    
    X = df[features]
    y = df[target]
    
    # Feature scaling
    if use_saved_scaler:
        X_scaled = use_saved_scaler.transform(X)
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, y, scaler
    
    return X_scaled, y
