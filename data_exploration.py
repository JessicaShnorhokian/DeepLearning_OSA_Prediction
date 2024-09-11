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
plt.savefig('data_exploration/charts/severity_distribution_pie_chart.png', dpi=300, bbox_inches='tight')
plt.show()


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
df[numerical_columns].describe().to_csv('data_exploration/numerical_columns.csv')


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
all_value_counts_df.to_csv('data_exploration/categorical_columns_value_counts.csv', index=False)


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
    q1 = df.groupby(['Severity'])[feature].quantile(0.25)
    q3 = df.groupby(['Severity'])[feature].quantile(0.75)
    iqr = q3 - q1
    y_offset = 0.05 * iqr 

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


def plot_categorical_data(categorical_vars, title, df):
    if len(categorical_vars) == 0:
        print(f"No categorical variables to plot for {title}")
        return
    
    category_data = df.groupby('Severity')[categorical_vars].mean()
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

    plt.show()

# Plotting for each category
plot_categorical_data(numerical_columns[1:12], 'numerical', df)
plot_categorical_data(categorical_columns, 'categorical', df)



