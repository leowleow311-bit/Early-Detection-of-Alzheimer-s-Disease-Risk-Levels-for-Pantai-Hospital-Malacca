from flask import Flask, request, render_template
import pickle
import numpy as np
import shap
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import OneHotEncoder


# Initialize Flask app
app = Flask(__name__)

# Load models
with open('kmeans_model.pkl', 'rb') as f:
    kmeans_model = pickle.load(f)

with open('xgboost_model.pkl', 'rb') as f:
    xgboost_model = pickle.load(f)

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        name = request.form['name']
        age = int(request.form['age'])
        gender = request.form['gender']
        mmse_score = float(request.form['mmse_score'])
        functional_assessment = float(request.form['functional_assessment'])
        memory_complaints = int(request.form['memory_complaints'])
        behavioral_problems = int(request.form['behavioral_problems'])
        adl_score = float(request.form['adl_score'])

        # Prepare data for prediction
        input_data = np.array([[
            mmse_score, functional_assessment,
            memory_complaints, behavioral_problems, adl_score
        ]]).reshape(1, -1)

        # Predict Alzheimer's disease status
        disease_prediction = xgboost_model.predict(input_data)[0]

        #2. Extract the leaf node indices for each sample in the dataset XGBoost method


        if disease_prediction == 1:  # If Alzheimer's detected
            
            scaler = StandardScaler()
            X_scaled = scaler
            X_scaled = scaler.fit_transform(input_data)

            alzheimers_level = kmeans_model.predict(X_scaled)[0]

            level_map = {0: "Mild", 1: "Moderate", 2: "Severe"}  # Adjust based on KMeans clusters
            category = level_map.get(alzheimers_level, "Unknown")
            advice = f"The patient has Alzheimer's disease at a {category} level."
        else:
            category = "No Alzheimer's detected"
            advice = "No signs of Alzheimer's disease were detected."

        # Generate SHAP explanations
        explainer = shap.TreeExplainer(xgboost_model)
        shap_values = explainer.shap_values(input_data)

        feature_names = ["MMSE", "Functional Assessment","ADL Score", "Memory Complaints", "Behavioral Problems"]

        # Save SHAP Summary Plot
        static_dir = "static"
        os.makedirs(static_dir, exist_ok=True)
        shap_summary_image = os.path.join(static_dir, "shap_summary.png")
        plt.figure()
        shap.summary_plot(shap_values, input_data, feature_names = feature_names, show=False)
        plt.savefig(shap_summary_image)
        plt.close()

        # Save SHAP Force Plot
        shap_force_html = os.path.join(static_dir, "shap_force.html")
        shap.save_html(shap_force_html, shap.force_plot(
            explainer.expected_value, shap_values[0], input_data[0]
        ))

        # Render result page
        return render_template(
            'result.html',
            name=name,
            age=age,
            gender=gender,
            category=category,
            advice=advice,
            shap_summary_image=f"/{shap_summary_image}",
            shap_force_html=f"/{shap_force_html}"
        )

    except Exception as e:
        # Render an error message if something goes wrong
        return render_template('result.html', name="Error", category="Error", advice=str(e))

if __name__ == '__main__':
    app.run(debug=True)
