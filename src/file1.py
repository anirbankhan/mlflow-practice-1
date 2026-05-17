import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt   
import seaborn as sns
# load wine datasets
wine = load_wine()
X = wine.data 
y = wine.target 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Defining the params for random model
max_depth = 10
n_estimators = 10
# set the experiemnet name in mlflow
mlflow.set_tracking_uri('http://localhost:5000')
# mlflow.set_tracking_uri('http://ec2-3-235-153-232.compute-1.amazonaws.com:5000/')
# mlflow.set_experiment('MLOPS-PRACTICE-1')

with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # logging metrics in mlflow
    mlflow.log_metric('accuracy',accuracy)
    # logging params in mlflow
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)
    print (accuracy)

    # creating a confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='crest', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig('confusion-matrix.png')
    # log artifact using mlflow
    mlflow.log_artifact('confusion-matrix.png')
    # log artifacts - same python file where we are writing the code
    mlflow.log_artifact(__file__)
    # set the tags
    mlflow.set_tags({'Author': 'Anirban', 'env': 'dev'})
    # log the model
    mlflow.sklearn.log_model(rf, 'Random Forest Model')