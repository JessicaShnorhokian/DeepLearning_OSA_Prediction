import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import preprocessing
import shap
import pickle
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
import os
import matplotlib.pyplot as plt
import argparse



# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define paths

# Define Args 
parser = argparse.ArgumentParser(description='')
parser.add_argument('--model', type=str, help='Model type, current support lstm, cnn, rnn')
parser.add_argument('--target_col', type=str, help='The column of the target label, choose Severity for multi-class, AHI_5 for binary cut-off at 5')

args = parser.parse_args()


model =args.model
target_col = args.target_col

path = 'data/OSA_complete_patients.csv'
model_pkl_path = f'results_imb_dl/{target_col}/SMOTE/{model}/best_model_fold_5.pkl'
model_path = f'data_visualization/intepretability/Shap/{target_col}/{model}/'

# Create directories if they don't exist
Path(model_path).mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(path, index_col=['PatientID'])
df.drop(df.columns[[0]], axis=1, inplace=True)

# Add columns AHI5, AHI15, and AHI30 
df['AHI_5'] = df['Severity'].apply(lambda x: 1 if x >= 1 else 0)
df['AHI_15'] = df['Severity'].apply(lambda x: 1 if x >= 2 else 0)
df['AHI_30'] = df['Severity'].apply(lambda x: 1 if x >= 3 else 0)

# Print Severity value counts
print("Severity value counts:", df['Severity'].value_counts())

# Scaling
seq_length = 49
x = preprocessing.MinMaxScaler().fit_transform(df.values[:, :seq_length])
y = df['Severity'].values

# Load the pre-trained model
with open(model_pkl_path, 'rb') as file:
    best_model = pickle.load(file)

# Ensure the model is in evaluation mode
best_model.eval()

# Cross-validation
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
fold_idx = 1
results = []

for train_ids, test_ids in sss.split(x, y):
    print('Fold: ', fold_idx)
    X_train, X_test = x[train_ids], x[test_ids]
    y_train, y_test = y[train_ids], y[test_ids]

    print(f"Fold {fold_idx} - Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    print(f"Fold {fold_idx} - Train set class distribution: {np.unique(y_train, return_counts=True)}")
    print(f"Fold {fold_idx} - Test set class distribution: {np.unique(y_test, return_counts=True)}")

    # Convert data to PyTorch Datasets and DataLoader
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Define a prediction function
    def predict_fn(data):
        if len(data.shape) == 2:
            data = data.reshape(data.shape[0], data.shape[1], 1)
        elif len(data.shape) == 1:
            data = data.reshape(1, -1, 1)
        data = torch.tensor(data, dtype=torch.float32).to(device)
        with torch.no_grad():
            output = best_model(data)
        return output.cpu().numpy()

    # SHAP interpretability
    masker = shap.maskers.Independent(X_train)
    explainer = shap.Explainer(predict_fn, masker)
    shap_values = explainer(X_test)

    # Save SHAP values
    shap_values_dir = os.path.join(model_path, 'shap_values')
    Path(shap_values_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(shap_values_dir, f'shap_values_fold_{fold_idx}.pkl'), 'wb') as f:
        pickle.dump(shap_values, f)

    X_test_df = pd.DataFrame(X_test, columns=df.columns[:seq_length])

    # Example SHAP plots for the current fold
    try:
        # For multi-class, we need to choose one class to visualize
        # Here, we'll visualize the positive class (index 1)

        #get all the unique values in the target column
        print(np.unique(y_test))
        for i in np.unique(y_test):
            print(f"Class {i} count: {np.sum(y_test == i)}")
            class_to_explain = i

        
            # Compute the mean absolute SHAP values for the chosen class
            mean_abs_shap = np.abs(shap_values[:, :, class_to_explain]).mean(0)
            print(f"Mean |SHAP value| : {mean_abs_shap}")
            
            # Sort features by importance
            feature_importance = pd.DataFrame(list(zip(X_test_df.columns, mean_abs_shap)), columns=['feature', 'importance'])
            feature_importance = feature_importance.sort_values('importance', ascending=False)
            feature_names = df.columns[:seq_length].tolist()
            plt.figure(figsize=(12, 10))

            shap.summary_plot(

                shap_values[:, :, class_to_explain], 
                X_test_df,
                plot_type="bar",
                feature_names=feature_names,
                max_display=10,
                show=False,
                color = '#FF6F91'

            )

            plt.xlabel('Mean |SHAP value|')

            plt.title(f'SHAP Summary Plot (Class {class_to_explain}) - {target_col}')
            plt.tight_layout()
            plt.savefig(os.path.join(model_path, f'shap_summary_plot_bar_class_{class_to_explain}.png'))
            plt.close()

    except Exception as e:
        print(f"Error in creating SHAP plots for fold {fold_idx}: {str(e)}")
        print(f"SHAP values shape: {shap_values.shape}")
        print(f"X_test_df shape: {X_test_df.shape}")
        print(f"X_test_df columns: {X_test_df.columns}")

    fold_idx += 1

# Example SHAP plots (for the last fold, can be customized)
# shap.summary_plot(shap_values, X_test)
# shap.summary_plot(shap_values, X_test, plot_type="bar")
# shap.dependence_plot("FeatureName", shap_values, X_test)  # Uncomment and replace "FeatureName" with your actual feature name
