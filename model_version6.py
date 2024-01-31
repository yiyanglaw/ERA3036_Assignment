#computer vision library
import cv2

#package for scientific computing with Python
import numpy as np

#preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV,ParameterGrid

#8 types of classifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier

#performance metrics 
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import roc_curve


#one hot encoding
from sklearn.preprocessing import label_binarize

#generate random numbers
import random

#parsing command line arguement
import argparse


from io import BytesIO

#operating system operation
import os

#image processing
import PIL.Image

#Used for parallelizing the execution of a function
import joblib
from joblib import Parallel, delayed


# Import tqdm for the progress bar
from tqdm import tqdm  

#function to train and evaluate classifier
def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    classifiers = {
        'SVM': SVC(),
        'Decision Tree': DecisionTreeClassifier(),
        'K-NN': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
        'Logistic Regression': LogisticRegression(),
        'Neural Network': MLPClassifier(max_iter=100),
        'Random Forest Classifier': RandomForestClassifier(),
        'Gradient Boosting Tree': GradientBoostingClassifier(),
    }
    
    #Create a dictionary of different classifiers to be evaluated.

    best_models = {}
    best_accuracy = 0

    for name, model in classifiers.items():
        if name != 'Random Forest Classifier':
            continue
        # Perform hyperparameter tuning using GridSearchCV
        param_grid = list(ParameterGrid(get_hyperparameter_grid(name)))
        with tqdm(total=len(param_grid), desc=f"{name} Training") as pbar:
            for params in param_grid:
                tuned_model = model.set_params(**params)
                tuned_model.fit(X_train, y_train)
                y_pred = tuned_model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_models['best_classifier'] = {
                        'model': tuned_model,
                        'accuracy': accuracy,
                        'y_pred': y_pred,
                        'feature_importances': getattr(tuned_model, 'coef_', None)
                    }

                print(f"{name} Accuracy: {accuracy:.4f}")

                if name == 'Random Forest Classifier' and accuracy > 0.87:
                    print(f"Stopping training for Neural Network as accuracy reached {accuracy:.4f}")
                    return best_models

                best_models[name] = {
                    'model': tuned_model,
                    'accuracy': accuracy,
                    'y_pred': y_pred
                }
                pbar.update(1)

    return best_models

def get_hyperparameter_grid(model_name):
    if model_name == 'SVM':
        return {
            'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000, 100000],
            'kernel': ['linear', 'rbf', 'sigmoid', 'poly'],
            'degree': [2, 3, 4],  
            'gamma': [1e-3, 1e-2, 1e-1, 1, 10],
            'random_state': [29] , 
            'probability': [True] 

        }

    elif model_name == 'Decision Tree':
        return {'max_depth': [None, 5, 10, 15], 'min_samples_split': [2, 5, 10, 20]}
    elif model_name == 'Random Forest':
        return {'n_estimators': [50, 100, 200, 300, 500],  
                'max_depth': [None, 10, 20, 30, 50],  
                'min_samples_split': [2, 5, 10, 20, 30],  
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt', 'log2', None],
                'bootstrap': [True, False]}

    elif model_name == 'K-NN':
        return {'n_neighbors': [3, 5, 7, 10], 'weights': ['uniform', 'distance']}
   
    elif model_name == 'Logistic Regression':
        return {'C': [0.1, 1, 10, 100], 'penalty': ['l2']}

    elif model_name == 'Neural Network':
        return {
            'hidden_layer_sizes': [(50,), (100,)],
            'alpha': [0.0001, 0.001],
            'activation': ['relu', 'tanh'],
            'solver': ['adam'],
            'learning_rate': ['constant', 'invscaling'],
            'learning_rate_init': [0.001, 0.01],
            'max_iter': [300],  
            'batch_size': [16],  
            'early_stopping': [True],
            'validation_fraction': [0.1],
            'n_iter_no_change': [5],
            'random_state': [20],  
            'shuffle': [True]  
        }
    
    elif model_name == 'Random Forest Classifier':
        return {'n_estimators': [50, 100, 200, 300, 500],
                'max_depth': [None, 10, 20, 30, 50],
                'min_samples_split': [2, 5, 10, 20, 30],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False]}

    elif model_name == 'Gradient Boosting Tree':
        return {'n_estimators': [50, 100, 200],
                'learning_rate': [ 0.1, 0.2, 0.5],
                'max_depth': [3, 5, 7, 10],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'subsample': [0.8, 0.9, 1.0]}

    elif model_name == 'Naive Bayes Classifier':
        return {}  

        
    # Return empty dictionary if model_name is not recognized
    return {}

#function to preprocesss image
def preprocess_image(image):
    image_np = np.array(image)
    flattened_image = image_np.flatten()
    return flattened_image

#function to processs image
def process_image(image_path):
    with open(image_path, 'rb') as file:
        return preprocess_image(PIL.Image.open(file))

#function to load image from folder
import os

def load_data(folder_path):
    shape_folders = [os.path.join(folder_path, shape) for shape in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, shape))]

    image_paths = []
    labels = []

    for shape_folder in shape_folders:
        print(f"Processing shape folder: {shape_folder}")
        
        # Check if the folder contains any images
        if not os.listdir(shape_folder):
            print(f"Warning: {shape_folder} is empty.")
            continue

        current_label = os.path.basename(shape_folder)  # Extract label from folder name
        image_paths += [os.path.join(shape_folder, image_file) for image_file in os.listdir(shape_folder) if image_file.endswith(('.png', '.jpg', '.jpeg'))]
        labels += [current_label] * len(os.listdir(shape_folder))

    return image_paths, labels


#perform train_test split
def preprocess_and_split_data(folder_path):
    X, y = load_data(folder_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, y_train, X_test, y_test


#hyperparameter_tuning
def hyperparameter_tuning(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

#performance evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)  # Get predicted probabilities

    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # One-vs-Rest ROC AUC
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    roc_auc = roc_auc_score(y_test_bin, y_probs, average='macro')

    # Calculate false positive rate (fpr) and true positive rate (tpr)
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
        'y_probs': y_probs  
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

    # Load, preprocess, and split data
    X_train, y_train, X_val, y_val = preprocess_and_split_data(args.train_folder)
    X_test, y_test = load_data(args.validate_folder)

    # Train and evaluate the model
    best_models = train_and_evaluate_model(X_train, y_train, X_val, y_val)

    # Hyperparameter tuning for the best classifier
    best_classifier_info = best_models['best_classifier']
    best_classifier = best_classifier_info['model']
    param_grid = {} 
    best_classifier_tuned = hyperparameter_tuning(best_classifier, param_grid, X_train, y_train)
    best_models['best_classifier_tuned'] = {
        'model': best_classifier_tuned,
        'accuracy': None,  
        'y_pred': None
    }

    # Save the best models
    save_model(best_models['best_classifier_tuned'], model_path='best_rf.joblib')

    # Evaluate the tuned model
    best_models['best_classifier_tuned']['y_pred'] = best_models['best_classifier_tuned']['model'].predict(X_test)
    best_models['best_classifier_tuned']['accuracy'] = accuracy_score(y_test, best_models['best_classifier_tuned']['y_pred'])

    # Print and save evaluation metrics 
    tuned_metrics = evaluate_model(best_models['best_classifier_tuned']['model'], X_test, y_test)
    print("\nTuned Model Metrics:")
    print(f"Confusion Matrix:\n{tuned_metrics['conf_matrix']}")
    print("Accuracy: {:.4f}".format(tuned_metrics['accuracy']))
    print("Precision: {:.4f}".format(tuned_metrics['precision']))
    print("Recall: {:.4f}".format(tuned_metrics['recall']))
    print("F1 Score: {:.4f}".format(tuned_metrics['f1']))
    print("ROC AUC Score: {:.4f}".format(tuned_metrics['roc_auc']))


    #Save metrics to a text file
    with open('tuned_rf.txt', 'w') as file:
        file.write("\nTuned Model Metrics:\n")
        file.write(f"Confusion Matrix:\n{tuned_metrics['conf_matrix']}\n")
        file.write("Accuracy: {:.4f}\n".format(tuned_metrics['accuracy']))
        file.write("Precision: {:.4f}\n".format(tuned_metrics['precision']))
        file.write("Recall: {:.4f}\n".format(tuned_metrics['recall']))
        file.write("F1 Score: {:.4f}\n".format(tuned_metrics['f1']))
        file.write("ROC AUC Score: {:.4f}\n".format(tuned_metrics['roc_auc']))

    print("Model training and evaluation completed.")






