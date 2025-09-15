# mlops-assignment-1
Assignment 1: GitHub, ML Models, and MLflow Tracking

MLOps Assignment 1

Problem statement & dataset description
    This assignment was basically to get hands on with GitHub basics and MLflow tracking stuff. It’s the first ML assignment so nothing too crazy. I created a public repository called mlops-assignment-1 to store everything. I structured the project with folders for data, notebooks, source code, models, and results using mkdir commands so it’s clean and organized. For the ML part, I used the Iris dataset because it’s small, classic, easy to work with, and perfect for practicing ML models training and evaluation.

Model selection & comparison
    I trained three ML models - Logistic Regression, Random Forest, and SVM. I chose these because they’re standard, easy to compare, and cover different approaches. While training, I saved the trained models as pickle files in the models/ folder so I could reuse them later. After training, I evaluated each model using accuracy, precision, recall, and F1-score. Random Forest performed the best overall, though the other models were also decent.

MLflow logging & tracking
    I set up MLflow to log parameters, metrics, and artifacts for each model run. I tracked the metrics through the MLflow UI which made it easy to compare runs. I also logged confusion matrices and evaluation results to see how each model was performing. The tracking helped confirm that Random Forest was the best model and worth registering.

Model registration
    The best model, Random Forest, was registered in the MLflow Model Registry with a version number. This way it’s easy to track which model is production-ready. Screenshots of MLflow UI and model registration are included in the results/ folder for reference.

How to run the code
To reproduce everything:

 - Clone the repository

 - Make sure Python and required libraries (scikit-learn, MLflow, pandas, joblib) are     installed

 - Run Train_Models.py to train models and save pickle files

 - Run Track_Models.py to log metrics and models to MLflow

 - Start MLflow UI with mlflow ui and open http://localhost:5000 to see logged experiments

Repository link
https://github.com/Sharjeelkhhan/mlops-assignment-1