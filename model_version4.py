import cv2
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import joblib
import argparse
from io import BytesIO
import os
import PIL.Image
from joblib import Parallel, delayed
from sklearn.metrics import roc_curve


def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    classifiers = {
        'SVM': SVC(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'K-NN': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
        'AdaBoost': AdaBoostClassifier(),
        'LDA': LinearDiscriminantAnalysis(),
        'Logistic Regression': LogisticRegression(),
        'Neural Network': MLPClassifier(max_iter=1000)
    }

    best_models = {}
    best_accuracy = 0

    for name, model in classifiers.items():
        param_grid = {}  
        tuned_model = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        tuned_model.fit(X_train, y_train)

        y_pred = tuned_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_models['best_classifier'] = {
                'model': tuned_model,
                'accuracy': accuracy,
                'y_pred': y_pred
            }

        print(f"{name} Accuracy: {accuracy:.4f}")
        best_models[name] = {
            'model': tuned_model,
            'accuracy': accuracy,
            'y_pred': y_pred
        }

    return best_models



def preprocess_image(image):
    image_np = np.array(image)
    flattened_image = image_np.flatten()
    return flattened_image

def load_data(folder_path):
    X = []
    y = []

    def process_image(image_path):
        with open(image_path, 'rb') as file:
            return preprocess_image(PIL.Image.open(file))

    for shape_label, shape in enumerate(['circle', 'square', 'triangle']):
        shape_folder = os.path.join(folder_path, shape)
        image_paths = [os.path.join(shape_folder, image_file) for image_file in os.listdir(shape_folder) if os.path.isfile(os.path.join(shape_folder, image_file))]

        results = Parallel(n_jobs=-1)(delayed(process_image)(image_path) for image_path in image_paths)

        X.extend(results)
        y.extend([shape_label] * len(results))

    X = np.array(X)
    y = np.array(y)

    return X, y

def preprocess_and_split_data(folder_path):
    X, y = load_data(folder_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, y_train, X_test, y_test

def hyperparameter_tuning(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)  

    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # One-vs-Rest ROC AUC
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    roc_auc = roc_auc_score(y_test_bin, y_probs, average='macro')

    fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_probs.ravel())

    return {
        'conf_matrix': conf_matrix,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr,
        'y_pred': y_pred,
        'y_probs': y_probs  # Include predicted probabilities
    }

def save_model(model, model_path='best_model2.joblib'):
    joblib.dump(model, model_path)

def load_model(model_path='best_model2.joblib'):
    return joblib.load(model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate models on shape data.')
    parser.add_argument('--train_folder', type=str, required=True, help='Path to the training data folder')
    parser.add_argument('--validate_folder', type=str, required=True, help='Path to the validation data folder')

    args = parser.parse_args()

    X_train, y_train, X_val, y_val = preprocess_and_split_data(args.train_folder)
    X_test, y_test = load_data(args.validate_folder)

    best_models = train_and_evaluate_model(X_train, y_train, X_val, y_val)

    best_classifier_info = best_models['best_classifier']
    best_classifier = best_classifier_info['model']
    param_grid = {}  # Define hyperparameter grid here
    best_classifier_tuned = hyperparameter_tuning(best_classifier, param_grid, X_train, y_train)
    best_models['best_classifier_tuned'] = {
        'model': best_classifier_tuned,
        'accuracy': None,  
        'y_pred': None
    }

    save_model(best_models['best_classifier_tuned'], model_path='best_model2_tuned.joblib')

    best_models['best_classifier_tuned']['y_pred'] = best_models['best_classifier_tuned']['model'].predict(X_test)
    best_models['best_classifier_tuned']['accuracy'] = accuracy_score(y_test, best_models['best_classifier_tuned']['y_pred'])

    tuned_metrics = evaluate_model(best_models['best_classifier_tuned']['model'], X_test, y_test)
    print("\nTuned Model Metrics:")
    print(f"Confusion Matrix:\n{tuned_metrics['conf_matrix']}")
    print("Accuracy: {:.4f}".format(tuned_metrics['accuracy']))
    print("Precision: {:.4f}".format(tuned_metrics['precision']))
    print("Recall: {:.4f}".format(tuned_metrics['recall']))
    print("F1 Score: {:.4f}".format(tuned_metrics['f1']))
    print("ROC AUC Score: {:.4f}".format(tuned_metrics['roc_auc']))


    with open('tuned_model_metrics.txt', 'w') as file:
        file.write("\nTuned Model Metrics:\n")
        file.write(f"Confusion Matrix:\n{tuned_metrics['conf_matrix']}\n")
        file.write("Accuracy: {:.4f}\n".format(tuned_metrics['accuracy']))
        file.write("Precision: {:.4f}\n".format(tuned_metrics['precision']))
        file.write("Recall: {:.4f}\n".format(tuned_metrics['recall']))
        file.write("F1 Score: {:.4f}\n".format(tuned_metrics['f1']))
        file.write("ROC AUC Score: {:.4f}\n".format(tuned_metrics['roc_auc']))

    print("Model training and evaluation completed.")
