import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

iris = load_iris()
X, y = iris.data, iris.target
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

mlflow.set_experiment("iris_models")

models = [
    ("Logistic Regression", LogisticRegression(max_iter=10)),
    ("Random Forest", RandomForestClassifier(n_estimators=20, random_state=0)),
    ("SVM", SVC(kernel="rbf"))
]

results = []
for name, model in models:
    with mlflow.start_run(run_name=name):
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred, average="macro")
        rec = recall_score(y_test, pred, average="macro")
        f1 = f1_score(y_test, pred, average="macro")

        mlflow.log_metric("Accuracy", acc)
        mlflow.log_metric("Precision", prec)
        mlflow.log_metric("Recall", rec)
        mlflow.log_metric("F1", f1)
        mlflow.sklearn.log_model(model, name.lower().replace(" ", "_"))

        results.append([name, acc, prec, rec, f1])

df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1"])
print(df)

best_model = RandomForestClassifier(n_estimators=20, random_state=0)
best_model.fit(x_train, y_train)

with mlflow.start_run(run_name="BestModelRegistration"):
    mlflow.sklearn.log_model(best_model, "random_forest_model", registered_model_name="IrisRandomForest")

