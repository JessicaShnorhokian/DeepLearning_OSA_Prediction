import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model', type=str, help='Model type, current support lstm, cnn, rnn', default='cnn')
    parser.add_argument('--target_col', type=str, help='The column of the target label, choose Severity for multi-class, AHI_5 for binary cut-off at 5', default='AHI_5')
    parser.add_argument('--imb', type=str, help='Imbalance handling strategy, choose none, SMOTE, BorderlineSMOTE, SMOTETomek or ADASYN', default='none')
    return parser

parser = parse_arguments()
args = parser.parse_args()


def plot_scores(X, models_data, colors, title, file_path):
    num_metrics = len(X)
    num_models = len(models_data)
    bar_width = 0.10 
    bar_distance = 0.01 

    X_axis = np.arange(num_metrics) 

    plt.figure(figsize=(12, 6), dpi=100)  # Increase figure size and DPI for better quality

    for i, (model, scores) in enumerate(models_data.items()):
        model_offset = i * (bar_width + bar_distance) 
        plt.bar(X_axis + model_offset, scores, bar_width, label=model, color=colors[i])

    plt.xticks(X_axis + (num_models * (bar_width + bar_distance)) / 2 - bar_width / 2, X, fontsize=12) 
    plt.ylabel("Scores", fontsize=14)
    plt.yticks(fontsize=12)
    all_scores = [score for scores in models_data.values() for score in scores]
    max_score = max(all_scores)
    plt.yticks(np.arange(0, max_score + 0.05, 0.05)) 
    plt.title(title, fontsize=16)
    plt.legend(fontsize=12, loc='upper right')  # Position legend at the top right
    plt.tight_layout()

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0.1)  # Save with minimal whitespace
    plt.show()


def process_evaluation_results(sampling_method, target_col):
    methods = ["dbn", "gru", "rnn", "gcn"]
    metrics = ["accuracy", "recall", "f1_weighted", "bal_acc", "precision", "g_mean"]
    
    evaluation_data = {}
    
    for method in methods:
        file_path = f"results_imb_dl/{target_col}/{sampling_method}/{method}/evaluation_results.csv"
        try:
            data = pd.read_csv(file_path)
            data = data.apply(pd.to_numeric, errors='coerce')
            data_mean = data.mean(skipna=True).to_dict()
            evaluation_data[method] = [data_mean.get(metric, np.nan) for metric in metrics]
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            evaluation_data[method] = [np.nan] * len(metrics)
    
    return evaluation_data

X = ['accuracy', 'recall_weighted', 'f1_weighted', 'bal_acc', 'precision_weighted', 'g_mean']

colors = ['#FF6F91', '#FF9671', '#FFD15F', '#E8F898']


if(args.imb=='none'):
    if(args.target_col == 'Severity'):
        data = {
            'dbn': [0.4044, 0.4044, 0.3552, 0.3292, 0.377, 0.2032],
            'gru': [0.4588, 0.4588, 0.4172, 0.3832, 0.4242, 0.2952],
            'rnn': [0.451, 0.451, 0.3972, 0.3722, 0.4118, 0.2588],
            'gcn': [0.4286, 0.4286, 0.3292, 0.3066, 0.4132, 0.1162]
        }
    if(args.target_col == 'AHI_5'):
        data = {
            'dbn': [0.815, 0.815, 0.8006, 0.5466, 0.8008, 0.3764],
            'gru': [0.8246, 0.8246, 0.8208, 0.6164, 0.8364, 0.5218],
            'rnn': [0.7718, 0.7718, 0.776, 0.5758, 0.8224, 0.4586],
            'gcn': [0.869, 0.869, 0.8082, 0.5002, 0.886, 0.0084]
        }

    if(args.target_col == 'AHI_15'):
        data = {
            'dbn': [0.6462, 0.6462, 0.6374, 0.615, 0.6544, 0.5858],
            'gru': [0.7114, 0.7114, 0.7032, 0.6682, 0.706, 0.6468],
            'rnn': [0.7026, 0.7026, 0.6764, 0.6342, 0.7016, 0.5744],
            'gcn': [0.6702, 0.6702, 0.6306, 0.5866, 0.6556, 0.496]
        }
    
    if(args.target_col == 'AHI_30'):
        data = {
            'dbn': [0.636, 0.636, 0.634, 0.6186, 0.6354, 0.6106],
            'gru': [0.6942, 0.6942, 0.6898, 0.6724, 0.6914, 0.6616],
            'rnn': [0.6934, 0.6934, 0.6866, 0.6666, 0.6896, 0.6506],
            'gcn': [0.6486, 0.6486, 0.6266, 0.6074, 0.6436, 0.559]
        }
else:
    data = process_evaluation_results(args.imb, args.target_col)



output_dir = "data_visualization/metric_comparison"

plot_scores(X, data, colors, f"{args.target_col} Classification - {args.imb}", f"{output_dir}/{args.target_col}/{args.target_col} Classification - {args.imb}.png")

print("Data visualization completed!")