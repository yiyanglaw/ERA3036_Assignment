from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import joblib

def train_model(X_train, y_train, algorithm):
    if algorithm == "knn":
        model = KNeighborsClassifier()
    elif algorithm == "decision_tree":
        model = DecisionTreeClassifier()
    elif algorithm == "logistic_regression":
        model = LogisticRegression()
    elif algorithm == "random_forest":
        model = RandomForestClassifier()
    elif algorithm == "neural_network":
        model = MLPClassifier()
    else:
        raise ValueError("Invalid algorithm")

    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    return cm, accuracy, precision, recall, f1

def save_model(model, filename):
    joblib.dump(model, filename)

def load_model(filename):
    return joblib.load(filename)
