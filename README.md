# Diabetes Risk Prediction API

A machine learning pipeline that predicts diabetes risk using the Pima Indians Diabetes dataset, deployed as a FastAPI backend with a Streamlit frontend for interactive predictions.

Built as part of my Machine Learning internship at AnalystLab Africa.

## Overview

This project takes a patient's health metrics and returns a diabetes risk prediction along with a probability score. It covers the full pipeline: data preprocessing, model training and tuning, API deployment, and a user-facing interface.

## Features

- **FastAPI backend** with Pydantic request validation
- **Trained Gradient Boosting model** tuned via RandomizedSearchCV (~0.93 ROC-AUC)
- **Health check endpoint** to confirm model and scaler load correctly
- **Streamlit frontend** for real-time predictions without needing Swagger UI or Postman
- **Logging middleware** for request tracking

## Tech Stack

- Python
- FastAPI
- Streamlit
- scikit-learn
- Pandas / NumPy
- Uvicorn

## Project Structure

```
FASTAPI_ML_API_CODE/
├── notebooks/
│   └── Diabetes.ipynb          # EDA, preprocessing, model training and tuning
├── main.py                     # FastAPI application
├── streamlit_app.py            # Streamlit frontend
├── diabetes_model.pkl          # Trained model artifact
├── scaler.pkl                  # Fitted scaler artifact
├── requirements.txt
└── README.md
```

## Setup

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/diabetes-prediction-api.git
   cd diabetes-prediction-api
   ```

2. Create and activate a virtual environment
   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS/Linux
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

## Running the App

You'll need two terminals running at the same time.

**Terminal 1 — start the API:**
```bash
uvicorn main:app --reload
```
API will be available at `http://127.0.0.1:8000`
Interactive docs at `http://127.0.0.1:8000/docs`

**Terminal 2 — start the frontend:**
```bash
streamlit run streamlit_app.py
```
App will open automatically at `http://localhost:8501`

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|--------------|
| GET | `/` | Welcome message |
| GET | `/health` | Confirms model and scaler are loaded |
| POST | `/predict` | Returns prediction, probability, and risk label |

### Example Request

```json
{
  "Pregnancies": 6,
  "Glucose": 148,
  "BloodPressure": 72,
  "SkinThickness": 35,
  "Insulin": 0,
  "BMI": 33.5,
  "DiabetesPedigreeFunction": 0.627,
  "Age": 70
}
```

### Example Response

```json
{
  "prediction": 1,
  "probability": 0.7298,
  "risk_label": "High risk"
}
```

## Model

The model was trained on the Pima Indians Diabetes dataset following this process:

- Group-based median imputation for biologically impossible zero values
- Comparison across Logistic Regression, Random Forest, Gradient Boosting, and Decision Tree classifiers
- Hyperparameter tuning with RandomizedSearchCV
- Final model: Random Forest, tuned, achieving ~0.93 ROC-AUC

Full training process is documented in `notebooks/Diabetes.ipynb`.

## Roadmap

- [x] Train and tune classification model
- [x] Build FastAPI backend with validation
- [x] Build Streamlit frontend
- [ ] Dockerize the application
- [ ] Deploy to cloud hosting

## Author

Michael — Machine Learning Intern at AnalystLab Africa

## License

This project is for educational purposes as part of an internship program.
