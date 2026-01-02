# Alzheimer's Disease Prediction


## Project Overview

A machine learning-based platform for early detection of Alzheimer's disease. It classifies users into risk levels (Low, Moderate, High, or Normal) based on their cognitive, functional, and lifestyle data, offering actionable recommendations to support early interventions.

## Features

- **Risk Prediction**: Categorizes individuals into varying Alzheimer's risk levels.
- **Personalized Recommendations**: Tailored advice for each risk category.
- **User-Friendly Interface**: Simple and intuitive web-based interaction.
-  **Model Transparency**: SHAP and LIME values explain feature contributions to predictions.

## How It Works

1. User fills in health and lifestyle questionnaire via a web interface.

2. System predicts Alzheimer's risk using the XGBoost model and KMeans model.

3. Displays risk level with recommendations for preventive or follow-up actions.

## Installation & Usage
**Website**
 1. Start the application:

> Use any python IDE to run the coding

    python app.py

 2. Open [this website](http://127.0.0.1:5000) in your browser.

**Website**
1. Use any Jupyter Notebook / VS Code to run the Show_mlflow.ipynb coding to open mlflow ui.

> Open terminal or command prompt(Admin) to run the coding. Make sure the path is correct before run.

    mlflow ui

 2. Open [this website](http://127.0.0.1:5000) in your browser.



## Technologies

-   **Dataset**: Kaggle (health, cognitive, and lifestyle data)
-   **Machine Learning**: XGBoost, KMeans clustering
-   **Model Trace**: MLflow Tracking
-   **Framework**: Flask (web interface)


## Authors

 - Ng Jun Cherng (Product Manager) - Oversees project execution, ensures alignment with objectives, and supervise the project to meet the dateline.
 - Chan Mei Yie (Machine Learning Engineer) - Developed the machine learning pipeline, models and assists in integrating the model into the system. 
 - Dee Ying A/P Kok Hoe (Machine Learning Reseacher) - Conducted research on suitable algorithms and evaluated model performance.
 - Low Jack Han (DevOps Engineer) - Developed the website, integrated the machine learning model, and delivered the final product.
 - Looi Yong Hang (MLOps Engineer) - Managed model versioning and tracked changes across versions for performance comparison.
 - Tan Wui Han (Data Engineer) - Preprocessed and managed datasets, ensuring clean and structured data.
 - Wong Wai Kuan (Data Scientist) - Analyzed data insights, conducted feature engineering, and contributed to model selection.
 - Choo Yun Huoy (Lecturer) - Project advisor and guide.


## Report Task Distribution 
- Executive Summary (Low Jack Han)
- Project Background (Wong Wai Kuan)
- Objective (Ng Jun Cherng)
- Project Scope & Significant (Tan Wui Han)
- Proposed Solution (Dee Ying A/P Kok Hoe)
- Machine Learning Experimentation (Chan Mei Yie)
- Results Discussion (Chan Mei Yie)
- Conclusion (Looi Yong Hang)


## License

MIT License


> To understand more details on codings and results part please refer to our final file (Alzheimer_v_6_Final.ipynb)

```
