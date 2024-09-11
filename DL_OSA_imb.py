


import numpy as np
import pandas as pd
import warnings
import time
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, cross_val_score, StratifiedShuffleSplit, train_test_split
from sklearn.metrics import (
    confusion_matrix, accuracy_score, classification_report, f1_score, roc_auc_score, precision_score,
    recall_score, average_precision_score, precision_recall_curve, make_scorer, balanced_accuracy_score
)
from sklearn import preprocessing
import torchvision.models as models
import torch
import torch.nn as nn
from easydict import EasyDict as edict
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import json
from datetime import datetime
import argparse
import itertools
from pathlib import Path
from models import DBN, MV_GRU, MV_RNN, MV_GCN
import os, pickle
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTETomek
from imblearn.metrics import geometric_mean_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define Args
def parse_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model', type=str, help='Model type, current support lstm, cnn, rnn')
    parser.add_argument('--target_col', type=str, help='The column of the target label, choose Severity for multi-class, AHI_5 for binary cut-off at 5')
    parser.add_argument('--batch_size', type=int, help='batch_size', default=256)
    parser.add_argument('--epochs', type=int, help='batch_size', default=70)
    parser.add_argument('--display', type=int, help='batch_size', default=256)
    parser.add_argument('--imb', type=str, choices=['SMOTE', 'ADASYN','BorderlineSMOTE','SMOTETomek'], help='Resampling technique to use for imbalanced data')
    return parser

parser = parse_arguments()
args = parser.parse_args()

# Load dataset
path = 'data/OSA_complete_patients.csv'
df = pd.read_csv(path, index_col=['PatientID'])
df.drop(df.columns[[0]], axis=1, inplace=True)
df.head(5)

# Add columns AHI5, AHI15, and AHI30 
df['AHI_5'] = df['Severity'].apply(lambda x: 1 if x >= 1 else 0)
df['AHI_15'] = df['Severity'].apply(lambda x: 1 if x >= 2 else 0)
df['AHI_30'] = df['Severity'].apply(lambda x: 1 if x >= 3 else 0)


seq_length = 49
# Scaling
x = preprocessing.MinMaxScaler().fit_transform(df.values[:, :seq_length])
y = df[args.target_col].values
num_class = len(set(y))
model_path = 'results_imb_dl/{}/{}/{}'.format(args.target_col, args.imb ,args.model )
Path(model_path).mkdir(parents=True, exist_ok=True)

# Resampling
def apply_resampling(X_train, y_train, technique):
    if technique == 'SMOTE':
        resampler = SMOTE()
    elif technique == 'BorderlineSMOTE':
        resampler = BorderlineSMOTE()
    elif technique == 'SMOTETomek':
        resampler = SMOTETomek()
    elif technique == 'ADASYN':
        resampler = ADASYN()
    else:
        raise ValueError("Unsupported resampling technique. Choose 'SMOTE' or 'ADASYN'.")
    X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

# Get data
def get_data(x_inp, y_inp):
    train_tensor = TensorDataset(torch.from_numpy(x_inp).float(), torch.from_numpy(y_inp).long())
    train_loader = DataLoader(train_tensor, batch_size=args.batch_size, shuffle=True)
    return train_loader


# Evaluate model
def eval_model(model, test_loader):
    
    model.eval()
    total_loss = 0.0
    out = []
    out_label = []
    for (inp, target) in test_loader:
        x_batch = inp.float().to(device).unsqueeze(dim=2)
        y_batch = target.long().to(device)
        output = model(x_batch)
        loss = criterion(output, y_batch)
        total_loss += loss
        preds = F.log_softmax(output, dim=1).argmax(dim=1).detach().cpu().numpy()
        output_label = y_batch.detach().cpu().numpy()
        for val in preds:
            out.append(val)
        for val in output_label:
            out_label.append(val)
    eval_loss = total_loss / len(test_loader)
    print(f"Sample of predictions: {out[:10]}")
    print(f"Sample of true labels: {out_label[:10]}")
    return eval_loss, out, out_label

# Train model
def train_model(args, model, train_loader, test_loader, criterion, optimizer):
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        total_acc = 0.0
        for (inp, target) in train_loader:
            x_batch = inp.float()
            y_batch = target.long()
            x_batch = x_batch.to(device).unsqueeze(dim=2)
            y_batch = y_batch.to(device)
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()    
            optimizer.zero_grad()
            total_loss += loss
            preds = F.log_softmax(output, dim=1).argmax(dim=1).detach().cpu().numpy()
            batch_acc = accuracy_score(preds, y_batch.detach().cpu().numpy())
            total_acc += batch_acc
        train_loss = total_loss / len(train_loader)
        train_acc = total_acc / len(train_loader)
       
        eval_loss, out, out_label = eval_model(model, test_loader)
        val_acc = accuracy_score(out_label, out)
        print(f"Epoch {epoch}, Train Loss: {train_loss.item():.4f}, Train Acc: {train_acc:.4f}, Val Loss: {eval_loss.item():.4f}, Val Acc: {val_acc:.4f}")
        print(f"Prediction distribution: {np.unique(out, return_counts=True)}")
        
        if epoch % args.display == 0:
            print('epoch : ', epoch, '|train loss : ', train_loss.item(), '|train acc : ', train_acc)      
    eval_loss, out, out_label = eval_model(model, test_loader)

    return model, out, out_label

# Tune model
def tune_LSTM(args, params, train_loader, test_loader, fold_num=1):
    torch.manual_seed(fold_num)
    np.random.seed(fold_num)
    best_f1 = 0.0
    best_result = []
    best_model = []
    best_param = []

    # Train model with different hyperparameters
    for model_param in params:
        if args.model == 'rnn':
            model = MV_RNN(device=device, n_features=1, seq_length=seq_length, hidden_dim=model_param[0], n_layers=model_param[2], num_class=num_class)
        if args.model == 'dbn':
            model = DBN(n_visible=seq_length, n_hidden=model_param[0], n_classes=num_class, n_layers=model_param[2]) 
        if args.model == 'gru':
            model = MV_GRU(device=device, n_features=1, seq_length=seq_length, hidden_dim=model_param[0], n_layers=model_param[2], num_class=num_class)
        if args.model == 'gcn':
            model = MV_GCN(n_features=1,seq_length=seq_length, hidden_dim=model_param[0], n_layers=model_param[2], num_class=num_class)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=model_param[1])
        model, out, out_label = train_model(args, model, train_loader, test_loader, criterion, optimizer)
        
        print(f"Hyperparameters: {model_param}")
        print(f"Sample of predictions: {out[:10]}")
        print(f"Sample of true labels: {out_label[:10]}")
        print(f"Prediction distribution: {np.unique(out, return_counts=True)}")

        result = {
            'accuracy': accuracy_score(out_label, out),
            'recall': recall_score(out_label, out, average='weighted'),
            'f1_weighted': f1_score(out_label, out, average='weighted'),
            'f1_macro': f1_score(out_label, out, average='macro'),
            'bal_acc': balanced_accuracy_score(out_label, out),
            'precision': precision_score(out_label, out, average='weighted', zero_division=1),
            'g_mean': geometric_mean_score(out_label, out)
        }

    # Save best parameters
        if result['f1_macro'] > best_f1:
            best_f1 = result['f1_macro']
            best_model = model
            best_result = result
            best_param = model_param
            best_predictions = out
            best_true_labels = out_label
    
    # Save predictions
        predictions_df = pd.DataFrame({
                    'true_labels': best_true_labels,
                    'predicted_labels': best_predictions
                })
        predictions_df.to_csv(os.path.join(model_path, f'predictions_fold_{fold_num}.csv'), index=False)

    torch.save(best_model.state_dict(), os.path.join(model_path, 'best_model_fold_{}.pth'.format(fold_num)))

    # Save the best model and best parameters
    with open(os.path.join(model_path, 'best_params_fold_{}.json'.format(fold_num)), 'w') as f:
        json.dump(best_param, f)
    with open(os.path.join(model_path, 'best_model_fold_{}.pkl'.format(fold_num)), 'wb') as file:
        pickle.dump(best_model, file)

    print(f"Fold {fold_idx} - Best parameters: {best_param}")
    return best_result

with open('config/{}.json'.format(args.model), "r") as f:
    search_space = json.load(f)

all_perm = []
for k, v in search_space.items():
    all_perm.append(v)
all_params = list(itertools.product(*all_perm))
start_time = datetime.now()
criterion = nn.CrossEntropyLoss()
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
fold_idx = 1
results = []

# cross-validation
for train_ids, test_ids in sss.split(x, y):
    print('Fold: ', fold_idx)
    X_train, X_test = x[train_ids], x[test_ids]
    y_train, y_test = y[train_ids], y[test_ids]
    if args.imb:
        X_train, y_train = apply_resampling(X_train, y_train, args.imb)
        print(f"Fold {fold_idx} - Resampled train set shape: {X_train.shape}")
        print(f"Fold {fold_idx} - Resampled train set class distribution: {np.unique(y_train, return_counts=True)}")
    else:
        print("Skipping resampling")
    train_loader, test_loader = get_data(X_train, y_train), get_data(X_test, y_test)
    best_result = tune_LSTM(args, all_params, train_loader, test_loader, fold_num=fold_idx)
    best_result['fold'] = fold_idx
    results.append(best_result)
    fold_df = pd.DataFrame([best_result])
    fold_df.to_csv(os.path.join(model_path, f'fold_{fold_idx}_results.csv'), index=False)

    fold_idx += 1

results_df = pd.DataFrame(results)

# Save results to CSV
results_df.to_csv(os.path.join(model_path, 'evaluation_results.csv'), index=False)

print('Total time: ', (datetime.now() - start_time))
