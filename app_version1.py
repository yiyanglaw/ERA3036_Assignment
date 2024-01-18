# app.py
import streamlit as st
from model import preprocess_image, load_data, train_and_evaluate_model, evaluate_model, save_model
from sklearn.model_selection import train_test_split

def main():
    st.title("Shape Classifier App")

    train_folder = "C:\\Users\\Law Yi Yang\\Downloads\\shape\\three_shapes_filled\\three_shapes_filled\\train"
    validate_folder = "C:\\Users\\Law Yi Yang\\Downloads\\shape\\three_shapes_filled\\three_shapes_filled\\validate"

    X_train, y_train = load_data(train_folder)
    X_test, y_test = load_data(validate_folder)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    best_models = train_and_evaluate_model(X_train, y_train, X_val, y_val)

    save_model(best_models['best_classifier'])

    st.header("Classifier Performance Metrics")

    classifiers = ['SVM', 'Decision Tree', 'Random Forest', 'K-NN', 'Naive Bayes', 'AdaBoost', 'LDA', 'Logistic Regression', 'Neural Network']

    for classifier_name in classifiers:
        if classifier_name == 'best_classifier':
            continue

        classifier_info = best_models[classifier_name]
        classifier = classifier_info['model']
        metrics = evaluate_model(classifier, X_test, y_test)

        st.subheader(f"{classifier_name} Metrics")
        st.write(f"Confusion Matrix:\n{metrics['conf_matrix']}")
        st.write("Accuracy: {:.4f}".format(metrics['accuracy']))
        st.write("Precision: {:.4f}".format(metrics['precision']))
        st.write("Recall: {:.4f}".format(metrics['recall']))
        st.write("F1 Score: {:.4f}".format(metrics['f1']))
        st.write("\n---\n")

    st.header("Shape Classification")

    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

        processed_image = preprocess_image(uploaded_file)

        best_classifier_info = best_models['best_classifier']
        best_classifier = best_classifier_info['model']
        best_prediction = best_classifier.predict([processed_image])[0]

        st.write("\nBest Prediction:")
        st.write(f"Best Classifier: {best_prediction}")

if __name__ == "__main__":
    main()
