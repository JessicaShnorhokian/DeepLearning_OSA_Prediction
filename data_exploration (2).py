#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

# Load the data
data = 'data/OSA_complete_patients.csv'
df = pd.read_csv(data, index_col=0)

# Data Overview
print("Dataset Overview:")
print(f"Number of samples: {df.shape[0]}")
print(f"Number of features: {df.shape[1]}")
print(f"Number of classes: {df['Severity'].nunique()}")


# In[3]:


# OSA Severity Classes Distribution

colors = [ '#FF6F91', '#FFD15F', '#E8F898', '#8CF5E3']

#plt.figure(figsize=(10, 6))
df['Severity'].value_counts().plot(kind='pie', autopct='%1.1f%%', labels=[0, 1, 2, 3], colors=colors, startangle=90)
#plt.title('Distribution of OSA Severity Classes')
plt.ylabel('')

# Create custom legend labels
legend_labels = ['Severity 0: AHI < 5', 'Severity 1: AHI <= 5 < 15', 'Severity 2: 15 <= AHI < 30', 'Severity 3: AHI >=30']

# Add legend
plt.legend(legend_labels, loc='upper left', bbox_to_anchor=(1, 1))
plt.axis('equal')
plt.show()


# In[4]:


numerical_columns = []
categorical_columns = []

cutoff_val = 10  # If more than 10 unique values, consider it as continuous

# Identify column types
for column in df.columns:
    unique_vals = df[column].nunique()
    
    # Check if the column is categorical based on unique values and data type
    if df[column].dtype == 'object' or unique_vals <= cutoff_val:
        categorical_columns.append(column)
    else:
        numerical_columns.append(column)

# Summarize numerical columns
#df[numerical_columns].describe().to_csv('data_exploration/numerical_columns.csv')

# Initialize a list to store value counts
all_value_counts = []

# Append value counts for each categorical column to the list
for column in categorical_columns:
    value_counts = df[column].value_counts().reset_index()
    value_counts.columns = ['value', 'count']
    value_counts['column_name'] = column
    all_value_counts.append(value_counts)

# Concatenate all value counts into a single DataFrame
all_value_counts_df = pd.concat(all_value_counts, ignore_index=True)

# Reorder columns for better readability
all_value_counts_df = all_value_counts_df[['column_name', 'value', 'count']]

# Save the combined value counts to a single CSV file
#all_value_counts_df.to_csv('data_exploration/categorical_columns_value_counts.csv', index=False)




# In[5]:


# plot the distribution of numerical features
for feature in numerical_columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(df[feature], kde=True, color="#D65DB1")
    plt.title(f'Distribution of {feature}')
    plt.savefig(f'Distribution_of_{feature}.png') 
    plt.close


# In[6]:


# plot the value counts of categorical features
for feature in categorical_columns:
    #plt.figure(figsize=(4, 5))
    df[feature].value_counts().plot(kind='bar', color="#FFD15F", edgecolor="orange")
    plt.title(f'Distribution of {feature}')
    output_dir = 'data_exploration/charts/categorical/'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}Distribution_of_{feature}.png')
    plt.close() 


# In[7]:


# Age and BMI Distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
sns.boxplot(x='Severity', y='Age', data=df, ax=ax1)
ax1.set_title('Age Distribution Across OSA Severity Levels')
sns.boxplot(x='Severity', y='BMI', data=df, ax=ax2)
ax2.set_title('BMI Distribution Across OSA Severity Levels')
plt.tight_layout()
plt.show()


# In[35]:


# Variables organized into their respective subsets

# Variables organized into their respective numerical and categorical subsets

# Demographic
Demographic_numerical = ['Age']
Demographic_categorical = ['Sex', 'Current_smoker', 'Former_smoker', 'Sedentary']

# Measurements
Measurements_numerical = [
    'Height', 'Weight', 'Cervical_perimeter', 'Abdominal_perimeter', 
    'Systolic_BP', 'Diastolic_BP', 'BMI'
]
Measurements_categorical = ['Maxillofacial_profile', 'High_BP']

# Comorbidities
Comorbidities_numerical = []
Comorbidities_categorical = [
    'Asthma', 'Rhinitis', 'COPD', 'Respiratory_fail', 'Myocardial_infarct', 
    'Coronary_fail', 'Arrhythmias', 'Stroke', 'Heart_fail', 'Arteriopathy', 
    'Gastric_reflux', 'Glaucoma', 'Diabetes', 'Hypercholesterolemia', 
    'Hypertriglyceridemia', 'Hypo(er)thyroidism', 'Depression', 'Obesity', 
    'Dysmorphology', 'Restless_Leg_Syndrome'
]

# Symptoms
Symptoms_numerical = []
Symptoms_categorical = [
    'Snoring', 'Diurnal_somnolence', 'Driving_drowsiness', 
    'Morning_fatigue', 'Morning_headache', 'Memory_problem', 
    'Nocturnal_perspiration', 'Shortness_of_breath_on_exertion', 
    'Nocturia', 'Drowsiness_accident', 'Near_miss_accident', 'Respiratory_arrest'
]

# Questionnaires
Questionnaires_numerical = ['Epworth_scale', 'Pichots_scale', 'Depression_scale']
Questionnaires_categorical = []


Questionnaires_categorical = []
def plot_categorical_data(categorical_vars, title, df):
    if len(categorical_vars) == 0:
        print(f"No categorical variables to plot for {title}")
        return
    
    category_data = df.groupby('Severity')[categorical_vars].mean()
    plt.figure(figsize=(12, 6))
    ax = category_data.plot(kind='bar', stacked=True)
    plt.title(f'{title} Across OSA Severity Levels')
    plt.xlabel('OSA Severity')
    plt.ylabel('Proportion')
    plt.legend(title=title, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adding value annotations on the bars
    for p in ax.patches:
        width = p.get_width()
        height = p.get_height()
        x = p.get_x() + width / 2
        y = p.get_y() + height / 2
        ax.annotate(f'{height:.2f}', (x, y), ha='center', va='center')

    plt.tight_layout()
    plt.show()


# Plotting for each category
plot_categorical_data(numerical_columns[1:9], 'Demographic', df)
plot_categorical_data(Demographic_categorical, 'Demo', df)
# plot_categorical_data(Comorbidities_categorical, 'Comorbidities')
# plot_categorical_data(Symptoms_categorical, 'Symptoms')
plot_categorical_data(Questionnaires_numerical, 'Questionnaires',df)

# def plot_grouped_bar_chart(categorical_vars, title, df):
#     if len(categorical_vars) == 0:
#         print(f"No categorical variables to plot for {title}")
#         return
    
#     category_data = df.groupby('Severity')[categorical_vars].sum()
#     category_proportions = category_data.div(df['Severity'].value_counts(), axis=0)
    
#     ax = category_proportions.plot(kind='bar', figsize=(14, 8))
#     plt.title(f'{title} Across OSA Severity Levels')
#     plt.xlabel('OSA Severity')
#     plt.ylabel('Proportion')
#     plt.legend(title=title, bbox_to_anchor=(1.05, 1), loc='upper left')
    
#     # Adding value annotations on the bars
#     for p in ax.patches:
#         height = p.get_height()
#         if height > 0:
#             ax.annotate(f'{height:.2f}', 
#                         (p.get_x() + p.get_width() / 2., height), 
#                         ha='center', va='center', 
#                         xytext=(0, 10), 
#                         textcoords='offset points')
    
#     plt.tight_layout()
#     plt.show()

# plot_grouped_bar_chart(Comorbidities_categorical, 'Comorbidities', df)


# In[ ]:





# In[ ]:






# In[44]:


#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
import os

# Load the data
data = 'OSA_complete_patients.csv'
df = pd.read_csv(data, index_col=0)

# Data Overview
print("Dataset Overview:")
print(f"Number of samples: {df.shape[0]}")
print(f"Number of features: {df.shape[1]}")
print(f"Number of classes: {df['Severity'].nunique()}")


# save_folder = 'plots'

# if not os.path.exists(save_folder):
#     os.makedirs(save_folder)


# In[3]:


# OSA Severity Classes Distribution

colors = [ '#FF6F91', '#FFD15F', '#E8F898', '#8CF5E3']

#plt.figure(figsize=(10, 6))
df['Severity'].value_counts().plot(kind='pie', autopct='%1.1f%%', labels=[0, 1, 2, 3], colors=colors, startangle=90)
#plt.title('Distribution of OSA Severity Classes')
plt.ylabel('')

# Create custom legend labels
legend_labels = ['Severity 0: AHI < 5', 'Severity 1: AHI <= 5 < 15', 'Severity 2: 15 <= AHI < 30', 'Severity 3: AHI >=30']

# Add legend
plt.legend(legend_labels, loc='upper left', bbox_to_anchor=(1, 1))
plt.axis('equal')
#
#plt.savefig('data_exploration/charts/severity_distribution_pie_chart.png', dpi=300, bbox_inches='tight')
plt.show()



# In[4]:


numerical_columns = []
categorical_columns = []

cutoff_val = 10  # If more than 10 unique values, consider it as continuous

# Identify column types
for column in df.columns:
    unique_vals = df[column].nunique()
    
    # Check if the column is categorical based on unique values and data type
    if df[column].dtype == 'object' or unique_vals <= cutoff_val:
        categorical_columns.append(column)
    else:
        numerical_columns.append(column)

print(len(numerical_columns))
print(len(categorical_columns))


# Summarize numerical columns
#df[numerical_columns].describe().to_csv('data_exploration/numerical_columns.csv')

# Initialize a list to store value counts
all_value_counts = []

# Append value counts for each categorical column to the list
for column in categorical_columns:
    value_counts = df[column].value_counts().reset_index()
    value_counts.columns = ['value', 'count']
    value_counts['column_name'] = column
    all_value_counts.append(value_counts)

# Concatenate all value counts into a single DataFrame
all_value_counts_df = pd.concat(all_value_counts, ignore_index=True)

# Reorder columns for better readability
all_value_counts_df = all_value_counts_df[['column_name', 'value', 'count']]

# Save the combined value counts to a single CSV file
#all_value_counts_df.to_csv('data_exploration/categorical_columns_value_counts.csv', index=False)




# In[5]:


# plot the distribution of numerical features
for feature in numerical_columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(df[feature], kde=True, color="#D65DB1")
    plt.title(f'Distribution of {feature}')
    plt.tight_layout()

    save_path = os.path.join('Q:\distribution\Continuous', f'Distribution_of_{feature}.png')
    plt.savefig(save_path)
    plt.close()





# plot the value counts of categorical features
for feature in categorical_columns:
    plt.figure(figsize=(10, 6))
    df[feature].value_counts().plot(kind='bar', color="#FFD15F", edgecolor="orange")
    plt.title(f'Distribution of {feature}')
    plt.tight_layout()
    save_path = os.path.join('Q:\distribution\Categorical', f'Distribution_of_{feature}.png')
    plt.savefig(save_path)
    plt.close()

for feature in numerical_columns:
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x='Severity', y=feature, data=df, palette=colors, width=0.7)

    # Calculate median values for each Severity group
    medians = df.groupby(['Severity'])[feature].median()

    # Calculate dynamic y-offset based on IQR or range of the data
    q1 = df.groupby(['Severity'])[feature].quantile(0.25)
    q3 = df.groupby(['Severity'])[feature].quantile(0.75)
    iqr = q3 - q1
    y_offset = 0.05 * iqr  # You can adjust the multiplier as needed

    # Add median labels to the plot
    for xtick in ax.get_xticks():
        median_value = medians[xtick]
        offset = y_offset[xtick] if y_offset[xtick] > 0 else 0.2  # fallback to fixed offset if needed
        ax.text(xtick, median_value + offset, f'{median_value:.2f}',
                horizontalalignment='center', size='small', color='black', weight='semibold')

    plt.title(f'{feature} Distribution Across OSA Severity Levels')
    plt.tight_layout()
    save_path = os.path.join('Q:\\distribution\\Continuous', f'Distribution_of_{feature}_across_severity.png')
    plt.savefig(save_path)
    plt.close()


# In[35]:


# Variables organized into their respective subsets

# Variables organized into their respective numerical and categorical subsets

# Demographic
Demographic_numerical = ['Age']
Demographic_categorical = ['Sex', 'Current_smoker', 'Former_smoker', 'Sedentary']

# Measurements
Measurements_numerical = [
    'Height', 'Weight', 'Cervical_perimeter', 'Abdominal_perimeter', 
    'Systolic_BP', 'Diastolic_BP', 'BMI'
]
Measurements_categorical = ['Maxillofacial_profile', 'High_BP']

# Comorbidities
Comorbidities_numerical = []
Comorbidities_categorical = [
    'Asthma', 'Rhinitis', 'COPD', 'Respiratory_fail', 'Myocardial_infarct', 
    'Coronary_fail', 'Arrhythmias', 'Stroke', 'Heart_fail', 'Arteriopathy', 
    'Gastric_reflux', 'Glaucoma', 'Diabetes', 'Hypercholesterolemia', 
    'Hypertriglyceridemia', 'Hypo(er)thyroidism', 'Depression', 'Obesity', 
    'Dysmorphology', 'Restless_Leg_Syndrome'
]

# Symptoms
Symptoms_numerical = []
Symptoms_categorical = [
    'Snoring', 'Diurnal_somnolence', 'Driving_drowsiness', 
    'Morning_fatigue', 'Morning_headache', 'Memory_problem', 
    'Nocturnal_perspiration', 'Shortness_of_breath_on_exertion', 
    'Nocturia', 'Drowsiness_accident', 'Near_miss_accident', 'Respiratory_arrest'
]

# Questionnaires
Questionnaires_numerical = ['Epworth_scale', 'Pichots_scale', 'Depression_scale']
Questionnaires_categorical = []


def plot_categorical_data(categorical_vars, title, df):
    if len(categorical_vars) == 0:
        print(f"No categorical variables to plot for {title}")
        return
    
    category_data = df.groupby('Severity')[categorical_vars].mean()
    #plt.figure(figsize=(12, 6))
    ax = category_data.plot(kind='bar', stacked=True)
    plt.title(f'{title} Across OSA Severity Levels')
    plt.xlabel('OSA Severity')
    plt.ylabel('Proportion')
    plt.legend(title=title, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adding value annotations on the bars
    for p in ax.patches:
        width = p.get_width()
        height = p.get_height()
        x = p.get_x() + width / 2
        y = p.get_y() + height / 2
        ax.annotate(f'{height:.2f}', (x, y), ha='center', va='center')

    #plt.tight_layout()
    plt.show()


# Plotting for each category
plot_categorical_data(numerical_columns[1:12], 'numerical', df)
plot_categorical_data(categorical_columns, 'categorical', df)




# In[ ]:




