## Importing Libraries ##
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


## Loading Dataset ##
data = pd.read_csv("C:/Users/nrdev/Downloads/greendestination (1).csv") 


## Information about Dataset ##
print("Shape of data:", data.shape)
data.columns
print(data.info())


## Looking for Missing Values ##
missing = data.isnull().sum()


## Looking for duplicates ##
duplicates = data.duplicated().sum()


## Looking for Outliers ##
outlier_cols = [
    'Age',
    'DailyRate',
    'DistanceFromHome',
    'HourlyRate',
    'MonthlyIncome',
    'MonthlyRate',
    'NumCompaniesWorked',
    'PercentSalaryHike',
    'TotalWorkingYears',
    'TrainingTimesLastYear',
    'YearsAtCompany',
    'YearsInCurrentRole',
    'YearsSinceLastPromotion',
    'YearsWithCurrManager'
]

for col in outlier_cols:
    plt.figure(figsize=(6,3))
    sns.boxplot(x=data[col])
    plt.title(f'Boxplot of {col}')
    plt.show()


## Four Business Moments ##
num_cols = data.select_dtypes(include=['int64', 'float64']).columns
business_moment = pd.DataFrame({
    'Mean': data[num_cols].mean(),
    'Variance': data[num_cols].var(),
    'Std Dev': data[num_cols].std(),
    'Skewness': data[num_cols].skew(),
    'Kurtosis': data[num_cols].kurt()
})


## Graphical Representation ##
sns.countplot(x='Attrition', data=data)
plt.title('Attrition Count')
plt.show()

sns.histplot(data['Age'], kde=True)
plt.title('Age Distribution')
plt.show()

## Bivariate
sns.boxplot(x='Attrition', y='Age', data=data)
plt.title('Age vs Attrition')
plt.show()

sns.boxplot(x='Attrition', y='MonthlyIncome', data=data)
plt.title('Monthly Income vs Attrition')
plt.show()

sns.boxplot(x='Attrition', y='YearsAtCompany', data=data)
plt.title('Years at Company vs Attrition')
plt.show()

plt.figure(figsize=(10,5))
sns.countplot(x='JobRole', hue='Attrition', data=data)
plt.xticks(rotation=45)
plt.title('Attrition by Job Role')
plt.show()

num_df = data.select_dtypes(include=['int64', 'float64'])

# Plot correlation heatmap
plt.figure(figsize=(12,6))
sns.heatmap(num_df.corr(), cmap='coolwarm', annot=False)
plt.title('Correlation Heatmap')
plt.show()


## Attrition Rate Calculation ##
attrition_rate = data['Attrition'].value_counts(normalize=True).get('Yes', 0) * 100
print(f"\n Attrition Rate: {attrition_rate:.2f}%")





















