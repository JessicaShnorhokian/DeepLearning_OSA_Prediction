import torch
import shap
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

# ... (your data loading code remains the same)
data = pd.read_csv('data/OSA_reduced_patients.csv', index_col=['PatientID'])
data.drop(data.columns[[0]], axis=1, inplace=True)
data['AHI_5'] = data['Severity'].apply(lambda x: 1 if x >= 1 else 0)
data['AHI_15'] = data['Severity'].apply(lambda x: 1 if x >= 2 else 0)
data['AHI_30'] = data['Severity'].apply(lambda x: 1 if x >= 3 else 0)


X = data.drop(['AHI_5', 'AHI_15', 'AHI_30', 'Severity'], axis=1)
y = data['Severity']

# Convert X to a PyTorch tensor
X_tensor = torch.tensor(X.values, dtype=torch.float32)

# Load your model
model = pickle.load(open('results_imb_dl/severity/SMOTE/dbn/best_model_fold_5.pkl', 'rb'))
model.eval()  # Set the model to evaluation mode

# Create a wrapper function for your model that accepts numpy arrays
def model_wrapper(x):
    x_tensor = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        return model(x_tensor).cpu().numpy()

# Create the explainer using the wrapper function
masker = shap.maskers.Independent(data=X.values)
explainer = shap.Explainer(model_wrapper, masker)

# Compute SHAP values
shap_values = explainer(X.values)

# Get feature names
feature_names = X.columns.tolist()

# Plot SHAP summary
shap.summary_plot(shap_values.values, X.values, feature_names=feature_names, plot_type="bar")