import cuml
import cuml.svm
import cuml.ensemble
import cuml.neighbors
import cuml.linear_model
from cuml.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import joblib
import argparse
import os
import PIL.Image
from io import BytesIO
import numpy as np
import cv2

def preprocess_image(uploaded_file):
    image = PIL.Image.open(uploaded_file)
    image_np = np.array(image)
    img = cv2.resize(image_np, (64, 64))
    flattened_image = img.flatten()
    return flattened_image

def load_data(folder_path):
    X = []
    y = []
    
    for shape_label, shape in enumerate(['circle', 'square', 'triangle']):
        shape_folder = os.path.join(folder_path, shape)
        
        for image_file in os.listdir(shape_folder):
            image_path = os.path.join(shape_folder, image_file)
            if os.path.isfile(image_path):
                with open(image_path, 'rb') as file:
                    flattened_image = preprocess_image(file)
                    X.append(flattened_image)
                    y.append(shape_label)
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y

def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    classifiers = {
        'SVM': cuml.svm.SVC(),
        'Decision Tree': cuml.ensemble.RandomForestClassifier(),
        'Random Forest': cuml.ensemble.RandomForestClassifier(),
        'K-NN': cuml.neighbors.KNeighborsClassifier(),
        'Naive Bayes': cuml.naive_bayes.GaussianNB(),
        'AdaBoost': cuml.ensemble.AdaBoostClassifier(),
        'LDA': cuml.linear_model.LogisticRegression(),
        'Logistic Regression': cuml.linear_model.LogisticRegression(),
        'Neural Network': cuml.linear_model.MLPClassifier(max_iter=1000)
    }

    best_models = {}
    best_accuracy = 0

    for name, model in classifiers.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_models['best_classifier'] = {
                'model': model,
                'accuracy': accuracy,
                'y_pred': y_pred
            }

        print(f"{name} Accuracy: {accuracy:.4f}")
        best_models[name] = {
            'model': model,
            'accuracy': accuracy,
            'y_pred': y_pred
        }

    return best_models

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return {'conf_matrix': conf_matrix, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

def save_model(model, model_path='best_model.joblib'):
    joblib.dump(model, model_path)

def main(train_folder, validate_folder):
    X_train, y_train = load_data(train_folder)
    X_validate, y_validate = load_data(validate_folder)

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    best_models = train_and_evaluate_model(X_train, y_train, X_test, y_test)

    save_model(best_models['best_classifier'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate models on shape data.')
    parser.add_argument('--train_folder', type=str, required=True, help='Path to the training data folder')
    parser.add_argument('--validate_folder', type=str, required=True, help='Path to the validation data folder')

    args = parser.parse_args()

    main(args.train_folder, args.validate_folder)
