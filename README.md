# immo_ml

## Create data 
- to create the `preprocessor.pkl` run
    `python3 preprocessing.py`

- to create the `random_forest.pkl`run
    `python3 modeling.py`

Once the pickle files are created you can test predicting the price of a house using 
`python3 test_model.py`

# Real Estate Price Prediction

This project focuses on preprocessing real estate data and training a machine learning model to predict property prices. The application is built using Python, Pandas, Scikit-learn, and Streamlit.

## Project Structure

```plaintext
├── data
│   ├── final_dataset.json    # Input data file
│   ├── preprocessor.pkl      # Saved preprocessor
│   ├── random_forest.pkl     # Saved Random Forest model
│   ├── gb.pkl                # Saved Gradient Boosting model
├── preprocessing.py          # Script for preprocessing data
├── modeling.py               # Script for training models
├── app.py                    # Streamlit app script
├── README.md                 # Project documentation
├── requirements.txt          # Python package dependencies
└── venv                      # Virtual environment directory

Prerequisites
Python 3.7 or higher
Virtual environment tool (venv)

Set Up Virtual Environment:  python -m venv venv

Activate Virtual Environment:

On Windows:           .\venv\Scripts\activate
On macOS and Linux:    source venv/bin/activate
Install Dependencies:  pip install -r requirements.txt


Run the Project: 

Run Data Preprocessing:  python preprocessing.py
Train the Models :       python modeling.py
Run the Streamlit App:   streamlit run app.py


Files and Scripts
data/final_dataset.json
This file contains the raw dataset used for preprocessing and model training.

preprocessing.py
This script preprocesses the raw data, creates a preprocessor pipeline, and saves it as preprocessor.pkl.

modeling.py
This script loads the preprocessor, trains the Random Forest and Gradient Boosting models, evaluates them, and saves the models as random_forest.pkl and gb.pkl.

app.py
This script runs the Streamlit app which loads the preprocessor and model to make predictions on new data inputs.

requirements.txt
This file contains the list of Python packages required to run the project.

## Usage

After installing and starting the application, open your web browser and navigate to:

[http://localhost:8501/#prediction]

Here you can interact with the prediction feature of the application.


Author
Rachael Shen 