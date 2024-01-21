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



def classify_shapes(uploaded_files, selected_model):
    predictions = []
    best_model = load_model(selected_model)
    for uploaded_file in uploaded_files:
        processed_image = preprocess_image(Image.open(uploaded_file))
        best_classifier = best_model['model']
        best_prediction = best_classifier.predict([processed_image])[0]
        predictions.append(best_prediction)
    return predictions

def evaluate_and_display(model_name, model_dict, X_test, y_test):
    st.header(f"Evaluation Metrics for {model_name}")

    model = model_dict['model']
    
    metrics = evaluate_model(model, X_test, y_test)

    fig_metrics = plt.figure(figsize=(8, 5))
    plt.bar(["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC Score"],
            [metrics['accuracy'], metrics['precision'], metrics['recall'],
             metrics['f1'], metrics['roc_auc']])
    plt.title(f"{model_name} Evaluation Metrics")
    plt.xlabel("Metrics")
    plt.ylabel("Values")
    st.pyplot(fig_metrics)

    st.header(f"Classification Report for {model_name}")
    visualizer = ClassificationReport(model, classes=['Circle', 'Square', 'Triangle'], support=True, cmap='Blues')
    visualizer.score(X_test, y_test)

    visualizer.finalize()

    report_filename = f"classification_report_{model_name.lower()}.png"
    visualizer.fig.savefig(report_filename)
    st.image(report_filename)

    # Plot Confusion Matrix
    st.header(f"Confusion Matrix for {model_name}")
    plot_confusion_matrix_heatmap(metrics['conf_matrix'])

    st.header(f"Precision-Recall Curves for Each Class - {model_name}")
    num_classes = len(np.unique(y_test))
    plot_precision_recall_curves(y_test, metrics['y_probs'], num_classes)

    st.header(f"ROC Curves for Each Class - {model_name}")
    plot_roc_curves(y_test, metrics['y_probs'], num_classes)
    
    
    if hasattr(model, 'feature_importances_'):
        st.header(f"Feature Importance for {model_name}")
        feature_importance = model.feature_importances_
        
        features = [f"Feature {i+1}" for i in range(len(feature_importance))]

        fig_feature_importance = px.bar(x=features, y=feature_importance,
                                        labels={'x': 'Feature', 'y': 'Importance'},
                                        title=f'Feature Importance - {model_name}',
                                        color=feature_importance,
                                        color_continuous_scale='Viridis')
        fig_feature_importance.update_traces(texttemplate='%{text:.2f}', textposition='outside')

        st.plotly_chart(fig_feature_importance)
    else:
        st.warning("Feature importance is not available for this model.")



def plot_predicted_distribution(predictions):
    labels = ['Circle', 'Square', 'Triangle']
    class_labels = [labels[prediction] for prediction in predictions]

    fig_distribution = px.scatter(x=list(range(len(predictions))), y=predictions,
                                  labels={'x': 'Image Index', 'y': 'Predicted Shape'},
                                  title='Distribution of Predicted Shapes')

    fig_distribution.update_layout(yaxis=dict(title='Predicted Shape', tickmode='array', tickvals=[0, 1, 2], ticktext=labels))

    st.plotly_chart(fig_distribution)




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


    
    
    st.markdown("""
        <a id="top"></a>
        <script>
            window.onload = function() {
                document.getElementById('top').scrollIntoView({ behavior: 'smooth' });
            };
        </script>
    """, unsafe_allow_html=True)
    
    
 
    st.markdown("""
        [About Us](https://www.youtube.com/) | 
        [Another Link](https://example.com/)
    """)

    st.markdown(
        """
        <style>
            .about-link {
                float: right;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

        

    st.markdown(
        """
        <style>
            .classify-button {
                float: left;
            }
        </style>
        """,
        unsafe_allow_html=True
        )

    
    st.title("Shape Classifier App ")

    st.header("Shape Classification")

    selected_model = st.selectbox("Select Model:", [ 'svm.joblib', 'knn.joblib','nn.joblib', 'log.joblib', 'decision_t.joblib'])

    uploaded_file_names = []

    uploaded_files = st.file_uploader("Choose multiple images...", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        col1, col2, col3, col4, col5 = st.columns(5)
        for i, uploaded_file in enumerate(uploaded_files):
            uploaded_file_names.append(uploaded_file.name)
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
        if st.button("Classify",key="classify_button"):
            predictions = classify_shapes(uploaded_files, selected_model)
            for i, (prediction, file_name) in enumerate(zip(predictions, uploaded_file_names)):
                st.write(f"Predicted Shape for {file_name}: {prediction}")
                
                
            plot_predicted_distribution(predictions)                
    

        


    # Dashboard Section
    st.title("Shape Classifier App - Dashboard")

    validate_folder = "C:\\Users\\Law Yi Yang\\Downloads\\shape\\three_shapes_filled\\three_shapes_filled\\validate"
    X_test, y_test = load_data(validate_folder)


    st.header(f"Classification Report for {selected_model}")

    # Load the selected model
    model = load_model(selected_model)

    evaluate_and_display(selected_model, model, X_test, y_test)



    st.header("Exploratory Data Analysis (EDA)")
    plot_eda(X_test, y_test)



    
    
    model_names = ['svm.joblib', 'knn.joblib', 'nn.joblib', 'log.joblib', 'decision_t.joblib']
    model_accuracies = []

    for model_name in model_names:
        model = load_model(model_name)
        metrics = evaluate_model(model['model'], X_test, y_test)
        accuracy = metrics['accuracy']
        model_accuracies.append(accuracy)

    # Create a bar chart to compare model accuracies
    fig_model_comparison = px.bar(x=model_names, y=model_accuracies, text=model_accuracies,
                                  labels={'x': 'Model', 'y': 'Accuracy'},
                                  title='Model Comparison - Accuracy',
                                  color=model_accuracies,
                                  color_continuous_scale='Viridis')
    fig_model_comparison.update_traces(texttemplate='%{text:.2f}', textposition='outside')

    st.plotly_chart(fig_model_comparison)
    
  

    st.markdown("""
        <div style="text-align:center; padding: 10px; margin-top: 50px; background-color: #f4f4f4;">
            <a href="#top" style="text-decoration: none; color: #777; font-size: 24px;">Machine Learning Concepts and Technologies</a>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

