# Heart Disease XGBoost API

This project is an end-to-end machine learning pipeline for predicting heart disease using an XGBoost model, based on the [Heart Failure Prediction dataset from Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction). The API is built with FastAPI and serves as a production-ready model for real-time predictions.

## Features

- **Custom Feature Engineering**: Creates binned groups based on features like age, blood pressure, cholesterol, and oldpeak values.
- **One-Hot Encoding**: Automatically encodes categorical variables.
- **FastAPI**: Exposes a RESTful API to make real-time predictions.
- **Model**: Uses an XGBoost classifier to predict the likelihood of heart disease based on user inputs.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/wilsojaw/heart-disease-xgboost.git
   cd heart-disease-xgboost

   python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3.	Install the dependencies:
pip install -r requirements.txt

Training the Model
	1.	Preprocess the data and train the model using:

python src/train.py

	2.	This will:
	•	Preprocess the data (handle missing values, one-hot encode categorical features).
	•	Train the model and save it as model/model.pkl.

Running the FastAPI Server

Once the model is trained, start the FastAPI server to expose the model’s prediction API:
	1.	Run the server with:
 		uvicorn api.main:app --reload

		2.	The API will be available at http://127.0.0.1:8000.
	3.	Access the API docs at:

 The Swagger UI will let you interactively test the /predict endpoint.

API Endpoints

POST /predict

Predict whether a person has heart disease based on the provided parameters.

Request Body:

{
  "Age": 60,
  "Sex": "F",
  "ChestPainType": "NAP",
  "RestingBP": 120,
  "Cholesterol": 230,
  "FastingBS": 0,
  "RestingECG": "Normal",
  "MaxHR": 140,
  "ExerciseAngina": "N",
  "Oldpeak": 1.2,
  "ST_Slope": "Flat"
}

Response:

{
  "prediction": 1
}

	•	1 = heart disease likely.
	•	0 = no heart disease detected.

Testing the Model

You can test the /predict endpoint using the Swagger UI or by sending a POST request directly from the terminal or Postman. Here’s an example with curl:

curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
  "Age": 60,
  "Sex": "F",
  "ChestPainType": "NAP",
  "RestingBP": 120,
  "Cholesterol": 230,
  "FastingBS": 0,
  "RestingECG": "Normal",
  "MaxHR": 140,
  "ExerciseAngina": "N",
  "Oldpeak": 1.2,
  "ST_Slope": "Flat"
}'
