# Web application framework 
import streamlit as st

# Library for interactive plots
import plotly.express as px
import plotly.graph_objects as go

#package for scientific computing with Python
import numpy as np

#performance metrics evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_curve, auc, roc_curve
from yellowbrick.classifier import ClassificationReport

#image processing
from PIL import Image

# Graph Plotting library
import matplotlib.pyplot as plt

#generate learning curve
from sklearn.model_selection import learning_curve

#import functions from model.py
from model_version6 import preprocess_image, load_model, evaluate_model, train_and_evaluate_model, load_data

#unsupervised learning evaluation
from sklearn.metrics import silhouette_score

#plot decision tree
from sklearn.tree import plot_tree

#unzip folder
import zipfile

import os


# Function to plot Precision-Recall Curves
def plot_precision_recall_curves(y_test, y_probs, num_classes):
    fig_pr_curves = go.Figure()

    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve(y_test == i, y_probs[:, i])
        auc_score = auc(recall, precision)
        fig_pr_curves.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=f'Class {i} (AUC={auc_score:.2f})'))

    fig_pr_curves.update_layout(title_text='Precision-Recall Curves', xaxis_title='Recall', yaxis_title='Precision')
    st.plotly_chart(fig_pr_curves)

# Function to plot ROC Curves
def plot_roc_curves(y_test, y_probs, num_classes):
    fig_roc_curves = go.Figure()

    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_test == i, y_probs[:, i])
        auc_score = auc(fpr, tpr)
        fig_roc_curves.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'Class {i} (AUC={auc_score:.2f})'))

    fig_roc_curves.update_layout(title_text='ROC Curves for Each Class', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
    st.plotly_chart(fig_roc_curves)

# Function to plot EDA
def plot_eda(X_train, y_train):
    class_distribution = np.bincount(y_train)
    labels = ['Circle', 'Square', 'Triangle']
    fig_pie = go.Figure(data=[go.Pie(labels=labels, values=class_distribution)])
    fig_pie.update_layout(title_text='Class Distribution Pie Chart')
    st.plotly_chart(fig_pie)

# Function to plot Confusion Matrix
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

#function to classify shapes
def classify_shapes(uploaded_files, selected_model):
    predictions = []
    best_model = load_model(selected_model)
    
    for uploaded_file in uploaded_files:
        # Process the image
        processed_image = preprocess_image(Image.open(uploaded_file))

        # Check if the image has the correct dimensions
        expected_dimensions = 3072
        if processed_image.size != expected_dimensions:
            st.warning(f"Error: Image dimensions are not {expected_dimensions}. Please upload images with the correct dimensions.")
            continue

        # Make predictions using the loaded model
        best_classifier = best_model['model']
        best_prediction = best_classifier.predict([processed_image])[0]
        predictions.append(best_prediction)

    return predictions


#Function for performance/data visualization
def evaluate_and_display(model_name, model_dict, X_test, y_test, X_train, y_train):
    

    # Extract the model from the dictionary
    model = model_dict['model']
    
    
    metrics = evaluate_model(model, X_test, y_test)

    # Display evaluation metrics as an image
    fig_metrics = plt.figure(figsize=(8, 5))
    plt.bar(["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC Score"],
            [metrics['accuracy'], metrics['precision'], metrics['recall'],
             metrics['f1'], metrics['roc_auc']])
    plt.title(f"{model_name} Evaluation Metrics")
    plt.xlabel("Metrics")
    plt.ylabel("Values")
    st.pyplot(fig_metrics)

    # Classification Report
    st.header(f"Classification Report for {model_name}")
    visualizer = ClassificationReport(model, classes=['Circle', 'Square', 'Triangle'], support=True, cmap='Blues')
    visualizer.score(X_test, y_test)

    # Finalize to make it ready for saving
    visualizer.finalize()

    # Save the figure and then display it
    report_filename = f"classification_report_{model_name.lower()}.png"
    visualizer.fig.savefig(report_filename)
    st.image(report_filename)
    
    st.header(f"Evaluation Metrics for {model_name}")

    # Get hyperparameters
    hyperparameters = model.get_params()

    # Display hyperparameters
    st.subheader("Hyperparameters:")
    st.json(hyperparameters)

    # Plot Confusion Matrix
    st.header(f"Confusion Matrix for {model_name}")
    plot_confusion_matrix_heatmap(metrics['conf_matrix'])

    # Precision-Recall Curves for each class Section
    st.header(f"Precision-Recall Curves for Each Class - {model_name}")
    num_classes = len(np.unique(y_test))
    plot_precision_recall_curves(y_test, metrics['y_probs'], num_classes)

    # ROC Curves for each class Section
    st.header(f"ROC Curves for Each Class - {model_name}")
    plot_roc_curves(y_test, metrics['y_probs'], num_classes)
    
    # Learning Curves Section  (unused to reduce the computational time)
    #st.header(f"Learning Curves for {model_name}")
    #plot_learning_curves(model, X_train, y_train)

    
    # Feature Importance Plot
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
    
    # Plot Decision Tree
    if model_name == 'decision_t.joblib':
        st.header(f"Decision Tree Visualization for {model_name}")
        plt.figure(figsize=(15, 10))
        plot_tree(model, filled=True, feature_names=[f"Pixel {i}" for i in range(X_train.shape[1])], class_names=['Circle', 'Square', 'Triangle'],max_depth=3)
        st.pyplot(plt.gcf())

        
        
        
# Function to plot Learning Curves for each model
def plot_learning_curves(model, X_train, y_train):
    train_sizes, train_scores, validation_scores = learning_curve(
        model, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1
    )

    # Calculate mean and standard deviation for training and validation scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    validation_scores_mean = np.mean(validation_scores, axis=1)
    validation_scores_std = np.std(validation_scores, axis=1)

    # Plot learning curves
    fig_learning_curves = go.Figure()
    fig_learning_curves.add_trace(
        go.Scatter(
            x=train_sizes,
            y=train_scores_mean,
            name="Training Score",
            mode="lines",
            line=dict(color="blue"),
        )
    )
    fig_learning_curves.add_trace(
        go.Scatter(
            x=train_sizes,
            y=validation_scores_mean,
            name="Validation Score",
            mode="lines",
            line=dict(color="orange"),
        )
    )

    # Add shaded areas for variance
    fig_learning_curves.add_trace(
        go.Scatter(
            x=np.concatenate([train_sizes, train_sizes[::-1]]),
            y=np.concatenate(
                [train_scores_mean - train_scores_std, (train_scores_mean + train_scores_std)[::-1]]
            ),
            fill="toself",
            fillcolor="rgba(0,100,80,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig_learning_curves.add_trace(
        go.Scatter(
            x=np.concatenate([train_sizes, train_sizes[::-1]]),
            y=np.concatenate(
                [
                    validation_scores_mean - validation_scores_std,
                    (validation_scores_mean + validation_scores_std)[::-1],
                ]
            ),
            fill="toself",
            fillcolor="rgba(255,165,0,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig_learning_curves.update_layout(
        title_text="Learning Curves",
        xaxis_title="Training Examples",
        yaxis_title="Accuracy",
        legend=dict(x=0, y=1, traceorder="normal"),
    )

    st.plotly_chart(fig_learning_curves)
    

# Function to plot Predicted Shape Distribution
def plot_predicted_distribution(predictions):
    predictions_array = np.array(predictions, dtype=int)  # Convert the list to a NumPy array with integer dtype
    class_labels = ['Circle', 'Square', 'Triangle']

    # Count occurrences of each class
    class_counts = np.bincount(predictions_array, minlength=len(class_labels))

    fig_distribution = px.bar(x=class_labels, y=class_counts,
                              labels={'x': 'Predicted Shape', 'y': 'Count'},
                              title='Distribution of Predicted Shapes')

    st.plotly_chart(fig_distribution)

    
        
# Function to plot Precision, Recall, and F1 Score
def plot_model_metrics(model_names, metrics_list, metric_name):
    fig_metrics_comparison = px.bar(x=model_names, y=metrics_list, text=metrics_list,
                                    labels={'x': 'Model', 'y': metric_name},
                                    title=f'Model Comparison - {metric_name}',
                                    color=metrics_list,
                                    color_continuous_scale='Viridis')
    fig_metrics_comparison.update_traces(texttemplate='%{text:.2f}', textposition='outside')

    st.plotly_chart(fig_metrics_comparison)

    
    
# Function to evaluate unsupervised learning model 
def evaluate_unsupervised_model_with_clusters(model, X):
    silhouette_scores = []
    num_clusters_range = range(2, 11)  

    for num_clusters in num_clusters_range:
        model.set_params(n_clusters=num_clusters)
        cluster_labels = model.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette_avg)

    return num_clusters_range, silhouette_scores

# Function to unzip the 'three_shapes_filled.zip' file
def unzip_folder(zipped_folder_path, destination_folder):
    try:
        with zipfile.ZipFile(zipped_folder_path, 'r') as zip_ref:
            zip_ref.extractall(destination_folder)
        print(f"Successfully unzipped '{zipped_folder_path}' to '{destination_folder}'.")

        # Check if 'three_shapes_filled' folder exists inside the destination folder
        extracted_folder = os.path.join(destination_folder, 'three_shapes_filled')
        if not os.path.exists(extracted_folder):
            raise FileNotFoundError(f"The 'three_shapes_filled' folder is not found inside '{destination_folder}'.")

        # Check if 'train' and 'validate' folders exist inside the 'three_shapes_filled' folder
        train_folder = os.path.join(extracted_folder, 'train')
        validate_folder = os.path.join(extracted_folder, 'validate')
        if not (os.path.exists(train_folder) and os.path.exists(validate_folder)):
            raise FileNotFoundError(f"The 'train' and 'validate' folders are not found inside 'three_shapes_filled' folder.")
        
        return train_folder, validate_folder

    except Exception as e:
        print(f"Error while unzipping: {e}")

# Function to unzip the 'svm.zip' file
def unzip_svm_model():
    zipped_folder_path = 'svm.zip'
    destination_folder = '.'  # current directory

    try:
        with zipfile.ZipFile(zipped_folder_path, 'r') as zip_ref:
            zip_ref.extract('svm.joblib', destination_folder)
        st.success(f"Successfully unzipped 'svm.joblib' from '{zipped_folder_path}' to '{destination_folder}'.")
    except Exception as e:
        st.error(f"Error while unzipping: {e}")

#Main functions to call all other functions
def main():
    # Set a background color
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
    
    # Anchor link to the top of the page using st.markdown
    st.markdown("""
        <a id="top"></a>
        <script>
            window.onload = function() {
                document.getElementById('top').scrollIntoView({ behavior: 'smooth' });
            };
        </script>
    """, unsafe_allow_html=True)

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
    
    #link to other website
    st.markdown("""
        [Download Resources](https://www.youtube.com/) | 
        [User Manual](https://example.com/)
    """)

    # Custom CSS to position the links at the top right corner
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

    # Custom CSS to position the second button at the top left corner
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

    # Shape Classification Section
    st.header("Shape Classification")

    unzip_svm_model()

    # select the supervised model
    selected_supervised_model = st.selectbox("Select Supervised Model:", ['svm.joblib','svm_without_data_aug.joblib', 'knn.joblib', 'nn.joblib', 'log.joblib', 'decision_t.joblib','gnb.joblib','gbt.joblib','rf.joblib'])
    
    # Create a list to store the names of uploaded files
    uploaded_file_names = []

    uploaded_files = st.file_uploader("Choose multiple images...", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        # Display images with a maximum of 5 images horizontally
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

        if st.button("Classify", key="classify_button"):
            predictions = classify_shapes(uploaded_files, selected_supervised_model)
            for i, (prediction, file_name) in enumerate(zip(predictions, uploaded_file_names)):
                st.write(f"Predicted Shape for {file_name}: {prediction}")

            plot_predicted_distribution(predictions)

    # Dashboard Section
    st.title("Shape Classifier App - Dashboard")

    zipped_folder_path = 'three_shapes_filled.zip'
    destination_folder = '.'  #  current directory

    # Call the unzip_folder function
    train_folder, validate_folder = unzip_folder('three_shapes_filled.zip', '.')


    # Load and preprocess data for testing
    X_test, y_test = load_data(validate_folder)
    
    # Load and preprocess data for training
    X_train, y_train = load_data(train_folder)


    # Classification Report for Supervised Models
    st.header(f"Classification Report for {selected_supervised_model}")

    # Load the selected supervised model
    supervised_model = load_model(selected_supervised_model)

    # Evaluate and display classification report
    evaluate_and_display(selected_supervised_model, supervised_model, X_test, y_test, X_train, y_train)

    # Exploratory Data Analysis (EDA) for Supervised Models
    st.header("Exploratory Data Analysis (EDA)")
    plot_eda(X_test, y_test)

    # Model Comparison Section for Supervised Models
    st.title("Model Comparison")

    # Create a list of supervised model
    supervised_model_names = ['svm.joblib','svm_without_data_aug.joblib', 'knn.joblib', 'nn.joblib', 'log.joblib', 'decision_t.joblib','gnb.joblib','gbt.joblib','rf.joblib']
    supervised_model_accuracies = []
    supervised_model_precisions = []
    supervised_model_recalls = []
    supervised_model_f1_scores = []

    for model_name in supervised_model_names:
        model = load_model(model_name)
        metrics = evaluate_model(model['model'], X_test, y_test)
        accuracy = metrics['accuracy']
        precision = metrics['precision']
        recall = metrics['recall']
        f1_score_value = metrics['f1']

        supervised_model_accuracies.append(accuracy)
        supervised_model_precisions.append(precision)
        supervised_model_recalls.append(recall)
        supervised_model_f1_scores.append(f1_score_value)

    # Model Accuracy Comparison for Supervised Models
    fig_supervised_accuracy_comparison = px.bar(x=supervised_model_names, y=supervised_model_accuracies, text=supervised_model_accuracies,
                                                labels={'x': 'Model', 'y': 'Accuracy'},
                                                title='Supervised Model Comparison - Accuracy',
                                                color=supervised_model_accuracies,
                                                color_continuous_scale='Viridis')
    fig_supervised_accuracy_comparison.update_traces(texttemplate='%{text:.2f}', textposition='outside')

    st.plotly_chart(fig_supervised_accuracy_comparison)

    # Model Precision Comparison for Supervised Models
    plot_model_metrics(supervised_model_names, supervised_model_precisions, 'Precision')

    # Model Recall Comparison for Supervised Models
    plot_model_metrics(supervised_model_names, supervised_model_recalls, 'Recall')

    # Model F1 Score Comparison for Supervised Models
    plot_model_metrics(supervised_model_names, supervised_model_f1_scores, 'F1 Score')

    

    # Unsupervised Learning Section
    st.header("Unsupervised Learning")

    # Allow the user to select the unsupervised model
    selected_unsupervised_model = st.selectbox("Select Unsupervised Model:", ['kmeans.joblib'], key="unsupervised_model_selection")

    # Load the selected unsupervised model
    unsupervised_model = load_model(selected_unsupervised_model)

    # Evaluate and display metrics for unsupervised model 
    num_clusters_range, silhouette_scores = evaluate_unsupervised_model_with_clusters(unsupervised_model, X_train)

    # Plot Silhouette Score vs Number of Clusters
    fig_silhouette_vs_clusters = px.line(x=num_clusters_range, y=silhouette_scores, markers=True)
    fig_silhouette_vs_clusters.update_layout(title_text='Silhouette Score vs Number of Clusters', xaxis_title='Number of Clusters', yaxis_title='Silhouette Score')
    st.plotly_chart(fig_silhouette_vs_clusters)

    # Display the silhouette scores for each number of clusters
    st.write("Silhouette Scores for Each Number of Clusters:")
    st.write(list(zip(num_clusters_range, silhouette_scores)))
    
    # cluster labels
    cluster_labels = unsupervised_model.predict(X_train)

    #anchor for back to top
    st.markdown("""
        <div style="text-align:center; padding: 10px; margin-top: 50px; background-color: #f4f4f4;">
            <a href="#top" style="text-decoration: none; color: #777; font-size: 24px;">Machine Learning Concepts and Technologies</a>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    unzip_svm_model()
    main()


