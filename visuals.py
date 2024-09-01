import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve
import os


def parse_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model', type=str, help='Model type, current support lstm, cnn, rnn', default='cnn')
    parser.add_argument('--target_col', type=str, help='The column of the target label, choose Severity for multi-class, AHI_5 for binary cut-off at 5', default='AHI_5')
    parser.add_argument('--figure_type', type=str, help='The type of figure to plot, choose roc_curve, precision_recall_curve, confusion_matrix, scores', default='confusion_matrix')
    parser.add_argument('--imb', type=str, help='Imbalance handling strategy, choose none, SMOTE, BorderlineSMOTE, SMOTETomek or ADASYN', default='none')

    return parser

parser = parse_arguments()
args = parser.parse_args()
output_dir = "data_visualization/metric_comparison"
colors = ['#FF6F91', '#FF9671', '#FFD15F', '#E8F898']

if args.figure_type == 'roc_curve' or args.figure_type == 'precision_recall_curve' or args.figure_type == 'confusion_matrix':
    if args.imb=='none':
        path = 'results/{}/{}/predictions_fold_5.csv'.format(args.target_col ,args.model)
    else:
        path = 'results_imb_dl/{}/{}/{}/predictions_fold_5.csv'.format(args.target_col ,args.imb,args.model)

    df = pd.read_csv(path)
    true_label = df['true_labels']
    pred_label = df['predicted_labels']


X = ['accuracy', 'recall_weighted', 'f1_weighted', 'bal_acc', 'precision_weighted', 'g_mean']


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

def plot_scores(X, models_data, colors, title, file_path):
    num_metrics = len(X)
    num_models = len(models_data)
    bar_width = 0.15
    bar_spacing = 0.02  # Added spacing between individual bars
    group_width = (bar_width + bar_spacing) * num_models - bar_spacing
    group_spacing = 0.3  # Spacing between score groups

    X_axis = np.arange(num_metrics) * (group_width + group_spacing)

    plt.figure(figsize=(12, 6), dpi=100)
    ax = plt.gca()

    for i, (model, scores) in enumerate(models_data.items()):
        offset = i * (bar_width + bar_spacing)
        plt.bar(X_axis + offset, scores, bar_width, label=model, color=colors[i])

    plt.xticks(X_axis + group_width / 2, X, fontsize=12)
    plt.ylabel("Scores", fontsize=14)
    plt.yticks(fontsize=12)

    all_scores = [score for scores in models_data.values() for score in scores]
    max_score = max(all_scores)
    plt.yticks(np.arange(0, max_score + 0.05, 0.05))

    plt.title(title, fontsize=16)

    # Remove top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Keep bottom and left borders visible
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    # Adjust the plot layout to make room for the legend
    plt.tight_layout()

    # Add legend at the bottom
    plt.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.09), ncol=num_models, frameon=False)

    # Adjust the figure size to accommodate the legend
    fig = plt.gcf()
    fig.set_size_inches(12, 6.5)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    
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


if(args.figure_type == 'scores'):
    if(args.imb == 'none'):
        title = f"{args.target_col} Classification - imbalanced"
    else:
        title = f"{args.target_col} Classification - {args.imb}"
    plot_scores(X, data, colors, title, f"{output_dir}/{args.target_col}/{args.target_col} Classification - {args.imb}.png")

if(args.figure_type == 'roc_curve'):
    fpr, tpr, thresholds = roc_curve(true_label, pred_label)
    auc = roc_auc_score(true_label, pred_label)
    print(auc)
    plt.plot(fpr, tpr, label="CNN, auc="+str(auc))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - SMOTE - Demographic')
    plt.legend()
    plt.show()

if(args.figure_type == 'precision_recall_curve'):
    precision, recall, thresholds = precision_recall_curve(true_label, pred_label)
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()

if(args.figure_type == 'confusion_matrix'):
    cm = confusion_matrix(true_label, pred_label)
    print(cm)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{args.target_col} - {args.imb} - {args.model}')
    plt.tight_layout()
    output_dir = f'data_visualization/confusion_matrices/{args.target_col}/{args.imb}'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/{args.target_col} - {args.imb} - {args.model}.png')
    plt.close()
