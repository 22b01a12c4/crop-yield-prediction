from flask import Flask, render_template, request, redirect, url_for
import joblib

app = Flask(__name__)

# Load model and scaler once
model = joblib.load("model/random_forest_model.pkl")
scaler = joblib.load("model/scaler.pkl")

features = [
    'NDVI', 'GNDVI', 'NDWI', 'SAVI',
    'soil_moisture', 'temperature',
    'rainfall', 'Pesticide_Amount_L_per_ha'
]

feature_full_forms = {
    'NDVI': 'Normalized Difference Vegetation Index',
    'GNDVI': 'Green Normalized Difference Vegetation Index',
    'NDWI': 'Normalized Difference Water Index',
    'SAVI': 'Soil Adjusted Vegetation Index'
}

@app.route("/", methods=["GET", "POST"])
def index():
    # GET request → NO prediction
    prediction = None

    if request.method == "POST":
        try:
            input_data = [float(request.form[feat]) for feat in features]
            X_scaled = scaler.transform([input_data])
            prediction = round(model.predict(X_scaled)[0], 2)

            # Redirect to GET with prediction
            return redirect(url_for("index", pred=prediction))

        except Exception as e:
            prediction = f"Error: {e}"

    # GET after redirect
    pred = request.args.get("pred")
    if pred is not None:
        prediction = pred

    return render_template(
        "index.html",
        prediction=prediction,
        features=features,
        feature_full_forms=feature_full_forms
    )

if __name__ == "__main__":
    app.run(debug=True)
