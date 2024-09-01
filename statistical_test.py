
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import chi2_contingency
import warnings
import time
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scikit_posthocs as sp

from sklearn.model_selection import KFold

path = 'data/OSA_complete_patients.csv'
df= pd.read_csv(path, index_col = ['PatientID'])
df.drop(df.columns[[0]], axis=1, inplace=True)
df.head(5)

# I am going to add columns AHI5, AHI15, and AHI30
df['AHI_5'] = df['Severity'].apply(lambda x: 1 if x >= 1 else 0)
df['AHI_15'] = df['Severity'].apply(lambda x: 1 if x >= 2 else 0)
df['AHI_30'] = df['Severity'].apply(lambda x: 1 if x >= 3 else 0)


#Sex
contingency_table = pd.crosstab(df['Sex'], df['Severity'])
chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

print(f"Chi2 Stat: {chi2_stat}")
print(f"P-value: {p_value}")
print("Contingency Table:")
print(contingency_table)

#respiratory_arrest
contingency_table = pd.crosstab(df['Respiratory_arrest'], df['Severity'])
chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

print(f"Chi2 Stat: {chi2_stat}")
print(f"P-value: {p_value}")
print("Contingency Table:")
print(contingency_table)


# # Group the 'Weight' data by 'Severity'
# groups = [df['Weight'][df['Severity'] == severity] for severity in df['Severity'].unique()]

# # Perform the Kruskal-Wallis test
# kruskal_stat, p_value = stats.kruskal(*groups)
# print(f"Kruskal-Wallis H-statistic: {kruskal_stat}")
# print(f"P-value: {p_value}")


# Group the 'Age' data by 'Severity'
groups = [df['Age'][df['Severity'] == severity] for severity in df['Severity'].unique()]

# Print the number of samples and summary statistics for each group
for idx, group in enumerate(groups):
    print(f"Severity level {df['Severity'].unique()[idx]}: {len(group)} samples")
    print(f"Group summary statistics: Mean = {group.mean()}, Std = {group.std()}")

# Perform the Kruskal-Wallis test
kruskal_stat, p_value = stats.kruskal(*groups)
print(f"Kruskal-Wallis H-statistic: {kruskal_stat}")
print(f"P-value: {p_value}")
dunn_result = sp.posthoc_dunn(df, val_col='Age', group_col='Severity')
print(dunn_result)










