import sqlite3
import re

def extract_metrics(file_content):
    pattern = r'Confusion Matrix:.*?(\d+)\D+(\d+)\D+(\d+)\D+(\d+).*?Accuracy:\s*([\d.]+).*?Precision:\s*([\d.]+).*?Recall:\s*([\d.]+).*?F1 Score:\s*([\d.]+).*?ROC AUC Score:\s*([\d.]+)'
    match = re.search(pattern, file_content, re.DOTALL)
    
    if match:
        # Extract values 
        confusion_matrix = [int(match.group(i)) for i in range(1, 5)]
        accuracy, precision, recall, f1_score, roc_auc_score = map(float, match.group(5, 6, 7, 8, 9))
        return confusion_matrix, accuracy, precision, recall, f1_score, roc_auc_score
    else:
        return None

def results_handling(db_path, file_paths):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_results (
            model_name TEXT,
            confusion_matrix TEXT,
            accuracy REAL,
            precision REAL,
            recall REAL,
            f1_score REAL,
            roc_auc_score REAL
        )
    ''')  

    for file_path in file_paths:
        model_name = file_path.split('.')[0]
        
        with open(file_path, 'r') as file:
            file_content = file.read()

        metrics = extract_metrics(file_content)

        if metrics:
            cursor.execute('''
                INSERT INTO model_results (
                    model_name,
                    confusion_matrix,
                    accuracy,
                    precision,
                    recall,
                    f1_score,
                    roc_auc_score       
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (model_name, str(metrics[0]), *metrics[1:]))

    # Commit changes and close the connection
    conn.commit()
    conn.close()
    
    
if __name__ == "__main__":
    file_paths = ["tuned_svm.txt","tuned_knn.txt", "tuned_gbt.txt", "tuned_nn.txt","tuned_decision_t.txt","tuned_logistic.txt","tuned_nb.txt","tuned_rf.txt"]  
    db_path = "model_results.db"
    results_handling(db_path, file_paths)
