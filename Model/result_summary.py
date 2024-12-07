import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import defaultdict

# Define paths
results_dir = "./scores/MLP"
summary_file = "./scores/MLP/summary_results.csv"

# Ensure the directory exists
os.makedirs(results_dir, exist_ok=True)

# Initialize a dictionary to store metrics grouped by n-gram
ngram_metrics = defaultdict(lambda: {
    "accuracy": [],
    "precision": [],
    "recall": [],
    "f1_score": []
})

# Process each prediction file
for file in os.listdir(results_dir):
    if file.endswith(".csv"):
        file_path = os.path.join(results_dir, file)
        
        # Extract n-gram information from the filename
        ngram = int(file.split("-gram")[0].split("--")[-1])
        
        # Read the predictions and true labels
        data = pd.read_csv(file_path)
        y_pred = data["pred"]
        y_true = data["true"]
        
        # Compute metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="weighted")
        recall = recall_score(y_true, y_pred, average="weighted")
        f1 = f1_score(y_true, y_pred, average="weighted")
        
        # Append metrics to the n-gram group
        ngram_metrics[ngram]["accuracy"].append(accuracy)
        ngram_metrics[ngram]["precision"].append(precision)
        ngram_metrics[ngram]["recall"].append(recall)
        ngram_metrics[ngram]["f1_score"].append(f1)

# Calculate averages for each n-gram group
summary_results = []
for ngram, metrics in ngram_metrics.items():
    average_results = {
        "Model": ngram,
        "Accuracy": sum(metrics["accuracy"]) / len(metrics["accuracy"]),
        "Precision": sum(metrics["precision"]) / len(metrics["precision"]),
        "Recall": sum(metrics["recall"]) / len(metrics["recall"]),
        "F1_score": sum(metrics["f1_score"]) / len(metrics["f1_score"]),
    }
    summary_results.append(average_results)

# Save the summary results to the CSV file
summary_df = pd.DataFrame(summary_results)
summary_df.to_csv(summary_file, index=False)

print(f"Metrics summary by n-gram saved to {summary_file}")
