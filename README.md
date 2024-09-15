
# Deep Learning and Neural Networks for the Classification of Obstructive Sleep Apnea

This project focuses on using artificial neural networks (ANNs) to accurately classify the severity of Obstructive Sleep Apnea (OSA) using readily available data. The models used in this study include Deep Belief Networks (DBNs), Recurrent Neural Networks (RNNs), Gated Recurrent Units (GRUs), and Graph Convolutional Networks (GCNs).

The dataset for this project was pre-processed to incorporate a range of features, including demographic data, measurements, comorbidities, symptoms, and responses from questionnaires. Both binary and multiclass classification tasks were explored. To address data imbalance, advanced resampling methods such as SMOTE, Borderline SMOTE, ADASYN, and SMOTE-Tomek were applied.

Hyperparameter tuning was employed to enhance model performance, with various metrics like accuracy, precision, recall, F1-score, confusion matrices, precision-recall curves, and AUC-ROC curves used to assess effectiveness.

To interpret the models' decision-making, SHAP and LIME were used for feature importance analysis. Additionally, statistical tests like Chi-square and Kruskal-Wallis were conducted to confirm the significance of features related to OSA severity.

## Installation


 Ensure you have `Python 3.11.2` or higher and `R 4.3.1` or higher installed on your system.




Start by cloning this repository to your local machine using the following command:

```bash
git clone https://github.com/your-username/your-repository-name.git
```

Install all necessary python and R packages and dependencies.


## Running the Project

For all command line arguments , choose the appropriate parameters depending on the task at hand. 

- target_col: Severity, AHI_5 , AHI_15, AHI_30
- model: GCN, RNN, GRU, DBN
- imb: SMOTE, SMOTETomek, BorderlineSMOTE, ADASYN 

#### 1. Hyperparameter Optimization 

Prepare your config for grid search in folder `config`

Then run, for example:

```bash
python DL_OSA.py --model DBN --batch_size 256 --epochs 10  --target_col AHI_5
```



#### 2. Imbalance Correction 


```bash
python DL_OSA_imb.py --model DBN --batch_size 256 --epochs 10  --target_col AHI_5 --imb ADASYN
```
#### 3. Model Interpretebility

- SHAP 

    ```bash
    python DL_OSA_interpretebility_SHAP.py --model DBN --target_col AHI_5
    ```

- Lime 

    ```bash
    python DL_OSA_interpretebility_LIME.py --model DBN --target_col AHI_5
    ```

#### 4. Statistical Tests

Statistical Tests are implemented using R in the `stats.r` file. Chi-Square is used for the categorical features and Kruskal-wallis is used for the continuos features. 


## Support

For support, email jessicashnorhokian@gmail.com.


