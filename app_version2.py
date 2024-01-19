import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_curve, auc, roc_curve
from PIL import Image
from yellowbrick.classifier import ClassificationReport
import matplotlib.pyplot as plt


from model2 import preprocess_image, load_model, evaluate_model, train_and_evaluate_model, load_data




def plot_precision_recall_curves(y_test, y_probs, num_classes):
    fig_pr_curves = go.Figure()

    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve(y_test == i, y_probs[:, i])
        auc_score = auc(recall, precision)
        fig_pr_curves.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=f'Class {i} (AUC={auc_score:.2f})'))

    fig_pr_curves.update_layout(title_text='Precision-Recall Curves', xaxis_title='Recall', yaxis_title='Precision')
    st.plotly_chart(fig_pr_curves)

def plot_roc_curves(y_test, y_probs, num_classes):
    fig_roc_curves = go.Figure()

    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_test == i, y_probs[:, i])
        auc_score = auc(fpr, tpr)
        fig_roc_curves.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'Class {i} (AUC={auc_score:.2f})'))

    fig_roc_curves.update_layout(title_text='ROC Curves for Each Class', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
    st.plotly_chart(fig_roc_curves)

def plot_eda(X_train, y_train):
    class_distribution = np.bincount(y_train)
    labels = ['Circle', 'Square', 'Triangle']
    fig_pie = go.Figure(data=[go.Pie(labels=labels, values=class_distribution)])
    fig_pie.update_layout(title_text='Class Distribution Pie Chart')
    st.plotly_chart(fig_pie)

def plot_confusion_matrix_heatmap(conf_matrix):
    fig_heatmap = px.imshow(
        conf_matrix,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=['Circle', 'Square', 'Triangle'],
        y=['Circle', 'Square', 'Triangle'],
        color_continuous_scale='Viridis',
    )
    fig_heatmap.update_layout(title_text='Confusion Matrix Heatmap')
    st.plotly_chart(fig_heatmap)

def plot_training_progress():
    epochs = [1, 2, 3, 4, 5]
    accuracy = [0.6, 0.7, 0.8, 0.85, 0.9]
    loss = [0.8, 0.7, 0.6, 0.5, 0.4]

    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=epochs, y=accuracy, mode='lines+markers', name='Accuracy'))
    fig_line.add_trace(go.Scatter(x=epochs, y=loss, mode='lines+markers', name='Loss'))

    fig_line.update_layout(title_text='Model Training Progress', xaxis_title='Epoch', yaxis_title='Metric')
    st.plotly_chart(fig_line)

def classify_shapes(uploaded_files, best_model):
    predictions = []
    for uploaded_file in uploaded_files:
        processed_image = preprocess_image(Image.open(uploaded_file))
        best_classifier = best_model['model']
        best_prediction = best_classifier.predict([processed_image])[0]
        predictions.append(best_prediction)
    return predictions



def main():
    st.markdown(
        """
        <style>
            body {
                background-color: #f0f2f6;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Set a background image
    st.markdown(
        """
        <style>
            
            .stApp {
                background-image: url('https://browsecat.art/sites/default/files/a-road-5k-wallpapers-41105-2616-8136744.png');
                background-size: cover;
            }
        </style>
        """,
        unsafe_allow_html=True
    )


    
    
    st.title("Shape Classifier App ")


    best_model_tuned = load_model('best_model2_tuned.joblib')

    st.header("Shape Classification")

    # Create a list to store the names of uploaded files
    uploaded_file_names = []

    uploaded_files = st.file_uploader("Choose multiple images...", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        # Display images with reduced size and a maximum of 5 images horizontally
        col1, col2, col3, col4, col5 = st.columns(5)
        for i, uploaded_file in enumerate(uploaded_files):
            uploaded_file_names.append(uploaded_file.name)
            # Use the appropriate column to display images
            if i % 5 == 0:
                col1.image(uploaded_file, caption=f"Uploaded Image {i + 1}", use_column_width=True, width=100, clamp=True)
            elif i % 5 == 1:
                col2.image(uploaded_file, caption=f"Uploaded Image {i + 1}", use_column_width=True, width=100, clamp=True)
            elif i % 5 == 2:
                col3.image(uploaded_file, caption=f"Uploaded Image {i + 1}", use_column_width=True, width=100, clamp=True)
            elif i % 5 == 3:
                col4.image(uploaded_file, caption=f"Uploaded Image {i + 1}", use_column_width=True, width=100, clamp=True)
            elif i % 5 == 4:
                col5.image(uploaded_file, caption=f"Uploaded Image {i + 1}", use_column_width=True, width=100, clamp=True)

        # Button to classify the uploaded images
        if st.button("Classify"):
            predictions = classify_shapes(uploaded_files, best_model_tuned)
            for i, (prediction, file_name) in enumerate(zip(predictions, uploaded_file_names)):
                st.write(f"Predicted Shape for {file_name}: {prediction}")



    # Dashboard Section
    st.title("Shape Classifier App - Dashboard")

    validate_folder = "C:\\Users\\Law Yi Yang\\Downloads\\shape\\three_shapes_filled\\three_shapes_filled\\validate"
    X_test, y_test = load_data(validate_folder)

    tuned_metrics = evaluate_model(best_model_tuned['model'], X_test, y_test)

    # Display evaluation metrics as an image
    st.header("Evaluation Metrics")
    fig_metrics = plt.figure(figsize=(8, 5))
    plt.bar(["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC Score"],
            [tuned_metrics['accuracy'], tuned_metrics['precision'], tuned_metrics['recall'],
             tuned_metrics['f1'], tuned_metrics['roc_auc']])
    plt.title("Model Evaluation Metrics")
    plt.xlabel("Metrics")
    plt.ylabel("Values")
    st.pyplot(fig_metrics)

    # Classification Report
    st.header("Classification Report")

    visualizer = ClassificationReport(best_model_tuned['model'], classes=['Circle', 'Square', 'Triangle'], support=True, cmap='Blues')
    visualizer.score(X_test, y_test)

    visualizer.finalize()

    visualizer.fig.savefig("classification_report.png")
    st.image("classification_report.png")


    st.header("Confusion Matrix")
    plot_confusion_matrix_heatmap(tuned_metrics['conf_matrix'])

    st.header("Exploratory Data Analysis (EDA)")
    plot_eda(X_test, y_test)

    st.header("Interactive Image Display")

    st.header("Prediction Probability Distribution")

    st.header("Precision-Recall Curves for Each Class")
    num_classes = len(np.unique(y_test))
    plot_precision_recall_curves(y_test, tuned_metrics['y_probs'], num_classes)

    st.header("ROC Curves for Each Class")
    plot_roc_curves(y_test, tuned_metrics['y_probs'], num_classes)

    st.header("Model Training Progress")
    plot_training_progress()
    
    st.markdown("""
        <div style="text-align:center; padding: 10px; margin-top: 50px; background-color: #f4f4f4;">
            <p style="font-size: 24px; color: #777;">Machine Learning Concepts and Technologies</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()