# Disaster Response Pipeline Project
## 1. Summary
Analyze disaster data from Appen (https://appen.com/) to build a model for an API that classifies disaster messages.

## 2. Installations
- pandas = 1.2.4
- numpy = 1.19.5
- matplotlib = 3.3.4
- seaborn = 0.11.1
- sklearn = 1.1.1
- nltk = 3.6.1
- joblib = 1.1.0
- nltk = 3.6.1
- re = 2.2.1
- sqlalchemy = 1.4.7

## 3. Files distribution
```
├── app
│   └── template
│       ├── master.html  # main page of web app
│       └── go.html  # classification result page of web app
│   └── run.py  # Flask file that runs app
├── data
│   └── disaster_categories.csv  # data to process 
│   └── disaster_messages.csv  # data to process
│   └── process_data.py
│   └── DisasterResponse.db   # database to save clean data to
├── models  
│   ├── train_classifier.py
│   └── classifier.pkl  # saved model 
└── README.md     
```

## 4. How to Interact with this project?
a. Run the following commands in the project's root directory to set up your database and model.
- To run ETL pipeline that cleans data and stores in database
```
python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
```
- To run ML pipeline that trains classifier and saves
```
python train_classifier.py ../data/DisasterResponse.db classifier.pkl
```
b. Run the following command in the app's directory to run your web app.
```
python run.py
```
c. Go to http://0.0.0.0:3001/

Note: Only `n_estimators` was used in `GridSearchCV`, however there are more hyperparameters that were commented due to the processing time.
