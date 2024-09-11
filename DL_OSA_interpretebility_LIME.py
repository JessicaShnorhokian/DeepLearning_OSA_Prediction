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
from lime.lime_tabular import LimeTabularExplainer
import argparse

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define Args 
parser = argparse.ArgumentParser(description='')
parser.add_argument('--model', type=str, help='Model type, current support lstm, cnn, rnn')
parser.add_argument('--target_col', type=str, help='The column of the target label, choose Severity for multi-class, AHI_5 for binary cut-off at 5')
args = parser.parse_args()

# Define model
model = args.model
target_col = args.target_col

# Define paths
path = 'data/OSA_complete_patients.csv'
model_pkl_path = f'results_imb_dl/{target_col}/SMOTE/{model}/best_model_fold_5.pkl'
model_path = f'data_visualization/intepretability/Lime/{target_col}/{model}/'

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
# Load the pre-trained model
with open(model_pkl_path, 'rb') as file:
    best_model = pickle.load(file)

# Ensure the model is in evaluation mode
best_model.eval()

# Cross-validation
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
fold_idx = 1
results = []

for train_ids, test_ids in sss.split(df[df.columns[:49]].values, df['Severity'].values):
    print('Fold: ', fold_idx)
    X_train, X_test = df.iloc[train_ids, :49].values, df.iloc[test_ids, :49].values
    y_train, y_test = df.iloc[train_ids]['Severity'].values, df.iloc[test_ids]['Severity'].values

    # Scaling within CV loop
    scaler = preprocessing.MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    X_test_df = pd.DataFrame(X_test, columns=df.columns[:49])
    
    # Define predict function
    def predict_fn(data):
        if len(data.shape) == 2:
            data = data.reshape(data.shape[0], data.shape[1], 1)
        elif len(data.shape) == 1:
            data = data.reshape(1, -1, 1)
        
        data = torch.tensor(data, dtype=torch.float32).to(device)
        with torch.no_grad():
            output = best_model(data)
        return torch.softmax(output, dim=1).cpu().numpy()

    # LIME explanation for each fold 
    lime_explainer = LimeTabularExplainer(X_train, mode='classification', training_labels=y_train, feature_names=df.columns[:49].tolist(), discretize_continuous=True)
    random_idx = np.random.randint(0, len(X_test))
    while np.argmax(predict_fn(X_test[random_idx:random_idx+1])) != 3:
        random_idx = np.random.randint(0, len(X_test))

    explanation = lime_explainer.explain_instance(X_test[random_idx], predict_fn)
    lime_explanation_dir = os.path.join(model_path)
    Path(lime_explanation_dir).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(lime_explanation_dir, f'lime_explanation_fold_{fold_idx}_{model}_{np.argmax(predict_fn(X_test[random_idx]))}.pkl'), 'wb') as f:
        pickle.dump(explanation, f)
    
    explanation_list = explanation.as_list()

    # Save explanation as a plot
    with plt.style.context("ggplot"):
        fig = explanation.as_pyplot_figure()
        plt.title(f"LIME explanation for {model} - {target_col} - prediction: {np.argmax(predict_fn(X_test[random_idx]))}", fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(lime_explanation_dir, f'lime_explanation_fold_{fold_idx}_{model}_{np.argmax(predict_fn(X_test[random_idx]))}.png'))
        plt.close()

    explanation.save_to_file(os.path.join(lime_explanation_dir, f'lime_explanation_fold_{fold_idx}_{model}_{np.argmax(predict_fn(X_test[random_idx]))}.html'))
            
    fold_idx += 1