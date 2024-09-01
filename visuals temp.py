import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve,average_precision_score
import os
from sklearn.preprocessing import label_binarize
from sklearn.metrics import auc


output_dir = "data_visualization/metric_comparison"
colors = ['#D65DB1', '#FF9671', '#FFD15F', '#8CF5E3']

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



target_col = 'Severity'
if(target_col == 'Severity'):
        data = {
            'dbn': [0.4044, 0.4044, 0.3552, 0.3292, 0.377, 0.2032],
            'gru': [0.4588, 0.4588, 0.4172, 0.3832, 0.4242, 0.2952],
            'rnn': [0.451, 0.451, 0.3972, 0.3722, 0.4118, 0.2588],
            'gcn': [0.4286, 0.4286, 0.3292, 0.3066, 0.4132, 0.1162]
        }
if(target_col == 'AHI_5'):
        data = {
            'dbn': [0.815, 0.815, 0.8006, 0.5466, 0.8008, 0.3764],
            'gru': [0.8246, 0.8246, 0.8208, 0.6164, 0.8364, 0.5218],
            'rnn': [0.7718, 0.7718, 0.776, 0.5758, 0.8224, 0.4586],
            'gcn': [0.869, 0.869, 0.8082, 0.5002, 0.886, 0.0084]
        }

if(target_col == 'AHI_15'):
        data = {
            'dbn': [0.6462, 0.6462, 0.6374, 0.615, 0.6544, 0.5858],
            'gru': [0.7114, 0.7114, 0.7032, 0.6682, 0.706, 0.6468],
            'rnn': [0.7026, 0.7026, 0.6764, 0.6342, 0.7016, 0.5744],
            'gcn': [0.6702, 0.6702, 0.6306, 0.5866, 0.6556, 0.496]
        }
    
if(target_col == 'AHI_30'):
        data = {
            'dbn': [0.636, 0.636, 0.634, 0.6186, 0.6354, 0.6106],
            'gru': [0.6942, 0.6942, 0.6898, 0.6724, 0.6914, 0.6616],
            'rnn': [0.6934, 0.6934, 0.6866, 0.6666, 0.6896, 0.6506],
            'gcn': [0.6486, 0.6486, 0.6266, 0.6074, 0.6436, 0.559]
        }

def plot_precision_recall_curves(num_models, models, target_col, imbalance_techniques, colors):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for i in range(num_models):
        model = models[i]
        imbalance_technique = imbalance_techniques[i]
        path = f'results_imb_dl/{target_col}/{imbalance_technique}/{model}/predictions_fold_5.csv'
        df = pd.read_csv(path)
        true_label = df['true_labels']
        pred_label = df['predicted_labels']
        
        # Check if it's binary or multiclass
        unique_classes = np.unique(true_label)
        n_classes = len(unique_classes)
        
        if n_classes == 2:
            # Binary classification
            precision, recall, _ = precision_recall_curve(true_label, pred_label)
            average_precision = average_precision_score(true_label, pred_label)
            ax.plot(recall, precision, color=colors[i], label=f'{model} - {imbalance_technique} (AP = {average_precision:.3f})')
        else:
            # Multiclass classification
            true_label_bin = label_binarize(true_label, classes=unique_classes)
            pred_label_bin = label_binarize(pred_label, classes=unique_classes)
            
            # Compute micro-average precision-recall curve and average precision
            precision, recall, _ = precision_recall_curve(true_label_bin.ravel(), pred_label_bin.ravel())
            average_precision = average_precision_score(true_label_bin, pred_label_bin, average="micro")
            ax.plot(recall, precision, color=colors[i], label=f'{model} - {imbalance_technique} (micro-avg AP = {average_precision:.3f})')
    
    # Add baseline
    ax.plot([0, 1], [1/n_classes, 1/n_classes], linestyle='--', color='black', label='Baseline')
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve - {target_col}')
    ax.legend(loc="center right")
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    plt.show()

def plot_roc_curves(num_models, models, target_col, imbalance_techniques, colors):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for i in range(num_models):
        model = models[i]
        imbalance_technique = imbalance_techniques[i]
        path = f'results_imb_dl/{target_col}/{imbalance_technique}/{model}/predictions_fold_5.csv'
        df = pd.read_csv(path)
        true_label = df['true_labels']
        pred_label = df['predicted_labels']
        
        unique_classes = np.unique(true_label)
        n_classes = len(unique_classes)
        
        if n_classes == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(true_label, pred_label)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[i], lw=2, label=f'{model} - {imbalance_technique} (AUC = {roc_auc:.2f})')
        else:
            # Multiclass classification
            true_label_bin = label_binarize(true_label, classes=unique_classes)
            pred_label_bin = label_binarize(pred_label, classes=unique_classes)
            
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            for j in range(n_classes):
                fpr[j], tpr[j], _ = roc_curve(true_label_bin[:, j], pred_label_bin[:, j])
                roc_auc[j] = auc(fpr[j], tpr[j])
            
            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(true_label_bin.ravel(), pred_label_bin.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            
            ax.plot(fpr["micro"], tpr["micro"], color=colors[i], lw=2, label=f'{model} - {imbalance_technique} (micro-avg AUC = {roc_auc["micro"]:.2f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Baseline')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve - {target_col}')
    ax.legend(loc="lower right")
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()
# Your existing code for calling the function
#Severity
num_models_severity = 4
models_severity = ['gru', 'rnn', 'dbn', 'gcn']
target_col_severity = 'Severity'
imbalance_techniques_severity = ['SMOTE', 'SMOTETomek', 'SMOTE', 'SMOTETomek']

# plot_roc_curves(num_models_severity, models_severity, target_col_severity, imbalance_techniques_severity, colors)
# plot_precision_recall_curves(num_models_severity, models_severity, target_col_severity, imbalance_techniques_severity, colors)

#AHI_5
num_models_AHI_5 = 4
models_AHI_5 = ['gru', 'rnn', 'dbn', 'gcn']
target_col_AHI_5 = 'AHI_5'
imbalance_techniques_AHI_5 = ['SMOTE', 'SMOTE', 'BorderlineSMOTE', 'BorderlineSMOTE']

# plot_roc_curves(num_models_AHI_5, models_AHI_5, target_col_AHI_5, imbalance_techniques_AHI_5, colors)
# plot_precision_recall_curves(num_models_AHI_5, models_AHI_5, target_col_AHI_5, imbalance_techniques_AHI_5, colors)

#AHI_15
num_models_AHI_15 = 4
models_AHI_15 = ['gru', 'rnn', 'dbn', 'gcn']
target_col_AHI_15 = 'AHI_15'
imbalance_techniques_AHI_15 = ['SMOTETomek', 'SMOTETomek', 'SMOTETomek', 'SMOTETomek']

# plot_roc_curves(num_models_AHI_15, models_AHI_15, target_col_AHI_15, imbalance_techniques_AHI_15, colors)
# plot_precision_recall_curves(num_models_AHI_15, models_AHI_15, target_col_AHI_15, imbalance_techniques_AHI_15, colors)

#AHI_30
num_models_AHI_30 = 4
models_AHI_30 = ['gru', 'rnn', 'dbn', 'gcn']
target_col_AHI_30 = 'AHI_30'
imbalance_techniques_AHI_30 = ['SMOTE', 'SMOTETomek', 'SMOTE', 'SMOTE']

# plot_roc_curves(num_models_AHI_30, models_AHI_30, target_col_AHI_30, imbalance_techniques_AHI_30, colors)
# plot_precision_recall_curves(num_models_AHI_30, models_AHI_30, target_col_AHI_30, imbalance_techniques_AHI_30, colors)


conf_matrix_severity_dbn = "results_imb_dl/Severity/SMOTE/DBN/predictions_fold_5.csv"
df = pd.read_csv(conf_matrix_severity_dbn)
true_label = df['true_labels']
pred_label = df['predicted_labels']
cm = confusion_matrix(true_label, pred_label)    
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('SMOTE - DBN - Severity')


plt.show()
def plot_grouped_bar_chart(scores_dict, title, category_labels):
    colors = ['#FF6F91', '#FF9671', '#FFD15F', '#E8F898']
    n_categories = len(category_labels)
    n_models = len(scores_dict)

    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.8 / n_models
    bar_spacing = 0.02
    group_spacing = 0.4

    X_axis = np.arange(n_categories) * (1 + group_spacing)
    max_score = max(max(scores) for scores in scores_dict.values())
    y_padding = 0.05 * max_score  # Add 5% padding to the y-axis

    for i, (model, scores) in enumerate(scores_dict.items()):
        offset = i * (bar_width + bar_spacing)
        ax.bar(X_axis + offset, scores, bar_width, label=model, color=colors[i])

    ax.set_xticks(X_axis + (n_models * bar_width + (n_models - 1) * bar_spacing) / 2)
    ax.set_xticklabels(category_labels, fontsize=12)
    ax.set_ylabel("Scores", fontsize=14)
    ax.set_ylim(0, max_score + y_padding)

    # Set y-ticks based on the max score
    y_ticks = np.linspace(0, max_score, num=6)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{y:.2f}" for y in y_ticks], fontsize=12)
    ax.tick_params(axis='y', labelsize=12)

    ax.set_title(title, fontsize=16)

    # Remove top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Keep bottom and left borders visible
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    # Add legend at the bottom
    plt.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False, ncol=n_models)

    plt.tight_layout()
    plt.show()

def plot_grouped_bar_chart_balancing(scores_dict, title, category_labels):
    colors = ['#FF6F91', '#FF9671', '#FFD15F', '#E8F898']
    n_categories = len(category_labels)
    n_models = len(scores_dict)

    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.8 / n_models
    bar_spacing = 0.02
    group_spacing = 0.4

    X_axis = np.arange(n_categories) * (1 + group_spacing)
    max_score = max(max(scores) for scores in scores_dict.values())
    y_padding = 0.05 * max_score  # Add 5% padding to the y-axis

    for i, (model, scores) in enumerate(scores_dict.items()):
        offset = i * (bar_width + bar_spacing)
        ax.bar(X_axis + offset, scores, bar_width, label=model, color=colors[i])

    ax.set_xticks(X_axis + (n_models * bar_width + (n_models - 1) * bar_spacing) / 2)
    ax.set_xticklabels(category_labels, fontsize=12)
    ax.set_ylabel("Distribution(%)", fontsize=14)
    ax.set_ylim(0, max_score + y_padding)

    # Set y-ticks based on the max score
    y_ticks = np.linspace(0, max_score, num=6)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{y:.2f}" for y in y_ticks], fontsize=12)
    ax.tick_params(axis='y', labelsize=12)

    ax.set_title(title, fontsize=16)

    # Remove top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Keep bottom and left borders visible
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    #add score for the bar
    for i, (model, scores) in enumerate(scores_dict.items()):
        for j, score in enumerate(scores):
            ax.text(X_axis[j] + i * (bar_width + bar_spacing) , score + 0.5, f"{score:}", ha='center', va='bottom', fontsize=9)


    # Add legend at the bottom
    plt.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False, ncol=n_models)

    plt.tight_layout()
    plt.show()

#Severity
SMOTE_Severity_Scores = process_evaluation_results('SMOTE', 'Severity')
SMOTETomek_Severity_Scores = process_evaluation_results('SMOTETomek', 'Severity')
BorderlineSMOTE_Severity_Scores = process_evaluation_results('BorderlineSMOTE', 'Severity')
ADASYN_Severity_Scores = process_evaluation_results('ADASYN', 'Severity')

scores_severity = {
    'GRU-SMOTE': SMOTE_Severity_Scores['gru'],
    'RNN-SMOTE Tomek': SMOTETomek_Severity_Scores['rnn'],
    'GCN-SMOTE Tomek': SMOTETomek_Severity_Scores['gcn'],
    'DBN-SMOTE': SMOTE_Severity_Scores['dbn']
}


metrics = ['accuracy', 'recall_weighted', 'f1_weighted', 'bal_acc', 'precision_weighted', 'g_mean']

#plot_grouped_bar_chart(scores_severity, '', metrics)

#AHI_5
SMOTE_AHI_5_Scores = process_evaluation_results('SMOTE', 'AHI_5')
SMOTETomek_AHI_5_Scores = process_evaluation_results('SMOTETomek', 'AHI_5')
BorderlineSMOTE_AHI_5_Scores = process_evaluation_results('BorderlineSMOTE', 'AHI_5')
ADASYN_AHI_5_Scores = process_evaluation_results('ADASYN', 'AHI_5')

scores_AHI_5 = {
    'GRU-SMOTE': SMOTE_AHI_5_Scores['gru'],
    'RNN-SMOTE': SMOTE_AHI_5_Scores['rnn'],
    'GCN-Borderline SMOTE': BorderlineSMOTE_AHI_5_Scores['gcn'],
    'DBN-Borderline SMOTE': BorderlineSMOTE_AHI_5_Scores['dbn']
}

#plot_grouped_bar_chart(scores_AHI_5, '', metrics)

#AHI_15
SMOTE_AHI_15_Scores = process_evaluation_results('SMOTE', 'AHI_15')
SMOTETomek_AHI_15_Scores = process_evaluation_results('SMOTETomek', 'AHI_15')
BorderlineSMOTE_AHI_15_Scores = process_evaluation_results('BorderlineSMOTE', 'AHI_15')
ADASYN_AHI_15_Scores = process_evaluation_results('ADASYN', 'AHI_15')

scores_AHI_15 = {
    'GRU-SMOTE Tomek': SMOTETomek_AHI_15_Scores['gru'],
    'RNN-SMOTE Tomek': SMOTETomek_AHI_15_Scores['rnn'],
    'GCN-SMOTE Tomek': SMOTETomek_AHI_15_Scores['gcn'],
    'DBN-SMOTE Tomek': SMOTETomek_AHI_15_Scores['dbn']
}

#plot_grouped_bar_chart(scores_AHI_15, '', metrics)

#AHI_30
SMOTE_AHI_30_Scores = process_evaluation_results('SMOTE', 'AHI_30')
SMOTETomek_AHI_30_Scores = process_evaluation_results('SMOTETomek', 'AHI_30')
BorderlineSMOTE_AHI_30_Scores = process_evaluation_results('BorderlineSMOTE', 'AHI_30')
ADASYN_AHI_30_Scores = process_evaluation_results('ADASYN', 'AHI_30')

scores_AHI_30 = {
    'GRU-SMOTE': SMOTE_AHI_30_Scores['gru'],
    'RNN-SMOTE Tomek': SMOTETomek_AHI_30_Scores['rnn'],
    'GCN-SMOTE': SMOTE_AHI_30_Scores['gcn'],
    'DBN-SMOTE': SMOTE_AHI_30_Scores['dbn']
}

#plot_grouped_bar_chart(scores_AHI_30, '', metrics)

#balancing techniques
severity = ['0', '1', '2', '3']
distribution ={
        'SMOTE': [25,25,25,25],
        'Borderline SMOTE': [25,25,25,25],
        'SMOTE Tomek': [25.7, 24.9, 24.9,  24.4],
        'ADASYN': [25.3, 24.9, 26.3,  23.5]
    }
plot_grouped_bar_chart_balancing(distribution, '', severity)
