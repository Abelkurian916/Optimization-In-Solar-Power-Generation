# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 14:53:25 2025

@author: abelk
"""

import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine

engine = create_engine("mysql+pymysql://root:abelkurian123@localhost:3306/solar_energy")

conn = engine.connect()

sql = "SELECT * FROM solar_energy.`inverter dataset`;"

df = pd.read_sql_query(sql, conn)


from scipy.stats import skew, kurtosis
from statistics import mode
from statistics import mean
from statistics import median


mean_value = df['Power of inverter 1 in unit 1'].mean()

median_value = df['Power of inverter 1 in unit 1'].median()

mode_value = df['Power of inverter 1 in unit 1'].mode()

print('Mean =',mean_value)
print('Median =',median_value)
print('Mode=',mode_value)


mean2_value = df['Power of inverter 2 in unit 1'].mean()

median2_value = df['Power of inverter 2 in unit 1'].median()

mode2_value = df['Power of inverter 2 in unit 1'].mode()

print('Mean =',mean2_value)
print('Median =',median2_value)
print('Mode=',mode2_value)

mean3_value = df['Power of inverter 1 in unit 2'].mean()

median3_value = df['Power of inverter 1 in unit 2'].median()

mode3_value = df['Power of inverter 1 in unit 2'].mode()


print('Mean =',mean3_value)
print('Median =',median3_value)
print('Mode=',mode3_value)


mean4_value = df['Power of inverter 2 in unit 2'].mean()

median4_value = df['Power of inverter 2 in unit 2'].median()

mode4_value = df['Power of inverter 2 in unit 2'].mode()


print('Mean =',mean4_value)
print('Median =',median4_value)
print('Mode=',mode4_value)

# variance of 'Power of inverter 1 in unit 1'

variance1_value = df['Power of inverter 1 in unit 1'].var()
print('variance of Power of inverter 1 in unit 1 = ',variance1_value)


# variance of Power of inverter 2 in unit 1

variance2_value = df['Power of inverter 2 in unit 1'].var()
print('variance of power of inverter 2 in 1 = ', variance2_value)

# variance of 'Power of inverter 1 in unit 2'

variance3_value = df['Power of inverter 1 in unit 2'].var()
print('variance of Power of inverter 1 in unit 2 = ',variance3_value)

# variance of 'Power of inverter 2 in unit 2'
variance4_value = df['Power of inverter 2 in unit 2'].var()
print('variance of Power of inverter 2 in unit 2 = ',variance4_value )



# Standared Deviation of Power of inverter 1 in unit 1

std_dev_value1 = df['Power of inverter 1 in unit 1'].std()
print('Standared Deviation of Power of inverter 1 in unit 1 = ',std_dev_value1)

# Standared Deviation of Power of inverter 2 in unit 1

std_dev_value2 = df['Power of inverter 2 in unit 1'].std()
print('Standared Deviation of Power of inverter 2 in unit 1 = ',std_dev_value2)


# Standared Deviation of Power of inverter 1 in unit 2


std_dev_value3 = df['Power of inverter 1 in unit 2'].std()
print('Standared Deviation of Power of inverter 1 in unit 2 = ',std_dev_value3)


# Standared Deviation of Power of inverter 2 in unit 2

std_dev_value4 = df['Power of inverter 2 in unit 2'].std()
print('Standared Deviation of Power of inverter 2 in unit 2 = ',std_dev_value4)


#Range of Power of inverter 1 in unit 1
range1_value = df['Power of inverter 1 in unit 1'].max()-df['Power of inverter 1 in unit 1'].min()
print('Range of Power of inverter 1 in unit 1 = ',range1_value)


#Range of Power of inverter 2 in unit 1

range2_value = df['Power of inverter 2 in unit 1'].max()-df['Power of inverter 2 in unit 1'].min()
print('Range of Power of inverter 2 in unit 1 = ',range2_value)


#Range of Power of inverter 1 in unit 2

range3_value = df['Power of inverter 1 in unit 2'].max()-df['Power of inverter 1 in unit 2'].min()
print('Range of Power of inverter 1 in unit 2 = ',range3_value)


#Range of Power of inverter 2 in unit 2

range4_value = df['Power of inverter 2 in unit 2'].max()-df['Power of inverter 2 in unit 2'].min()
print('Power of inverter 2 in unit 2 = ',range4_value)


# Skewness of 'Power of inverter 1 in unit 1'
skew1 = skew(df['Power of inverter 1 in unit 1'])
print('Skewness of Power of inverter 1 in unit 1 =', skew1)

# Skewness of 'Power of inverter 2 in unit 1'
skew2 = skew(df['Power of inverter 2 in unit 1'])
print('Skewness of Power of inverter 2 in unit 1 =', skew2)

# Skewness of 'Power of inverter 1 in unit 2'
skew3 = skew(df['Power of inverter 1 in unit 2'])
print('Skewness of Power of inverter 1 in unit 2 =', skew3)

# Skewness of 'Power of inverter 2 in unit 2'
skew4 = skew(df['Power of inverter 2 in unit 2'])
print('Skewness of Power of inverter 2 in unit 2 =', skew4)


# Kurtosis of 'Power of inverter 1 in unit 1'
kurt1 = kurtosis(df['Power of inverter 1 in unit 1'])
print('Kurtosis of Power of inverter 1 in unit 1 =', kurt1)

# Kurtosis of 'Power of inverter 2 in unit 1'
kurt2 = kurtosis(df['Power of inverter 2 in unit 1'])
print('Kurtosis of Power of inverter 2 in unit 1 =', kurt2)

# Kurtosis of 'Power of inverter 1 in unit 2'
kurt3 = kurtosis(df['Power of inverter 1 in unit 2'])
print('Kurtosis of Power of inverter 1 in unit 2 =', kurt3)

# Kurtosis of 'Power of inverter 2 in unit 2'
kurt4 = kurtosis(df['Power of inverter 2 in unit 2'])
print('Kurtosis of Power of inverter 2 in unit 2 =', kurt4)



#GRAPHICAL REPRESENTATION OF Power of inverter 1 in unit 1


import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats


#univariate
##Histogram 


sns.histplot(df["Power of inverter 1 in unit 1"], bins=20, kde=True, color="green")
sns.histplot(df["Power of inverter 2 in unit 1"], bins=20, kde=True, color="green")
sns.histplot(df["Power of inverter 1 in unit 2"], bins=20, kde=True, color="green")
sns.histplot(df["Power of inverter 2 in unit 2"], bins=20, kde=True, color="green")

##Box Plot 


sns.boxplot(y=df["Power of inverter 1 in unit 1"], color="orange")
sns.boxplot(y=df["Power of inverter 2 in unit 1"], color="orange")
sns.boxplot(y=df["Power of inverter 1 in unit 2"], color="orange")
sns.boxplot(y=df["Power of inverter 2 in unit 2"], color="orange")


##Q-Q (Quantile-Quantile) Plot


stats.probplot(df["Power of inverter 1 in unit 1"], dist="norm", plot=plt)
stats.probplot(df["Power of inverter 2 in unit 1"], dist="norm", plot=plt)
stats.probplot(df["Power of inverter 1 in unit 2"], dist="norm", plot=plt)
stats.probplot(df["Power of inverter 2 in unit 2"], dist="norm", plot=plt)


##Density plot 


sns.kdeplot(df['Power of inverter 1 in unit 1'], shade=True, color="blue", linewidth=2)
sns.kdeplot(df['Power of inverter 2 in unit 1'], shade=True, color="blue", linewidth=2)
sns.kdeplot(df['Power of inverter 1 in unit 2'], shade=True, color="blue", linewidth=2)
sns.kdeplot(df['Power of inverter 2 in unit 2'], shade=True, color="blue", linewidth=2)


#BIVARIATE ANALYSIS

#SCATTER PLOTS AND CORRELATION OF Power of inverter 1 in unit 1

plt.scatter(df["Power of inverter 1 in unit 1"],df["Power of inverter 2 in unit 1"], color ='red')
df['Power of inverter 1 in unit 1'].corr(df['Power of inverter 2 in unit 1']) #   Correlation Coefficient = 0.9883180626693996

plt.scatter(df["Power of inverter 1 in unit 1"],df["Power of inverter 1 in unit 2"], color ='red')
df['Power of inverter 1 in unit 1'].corr(df['Power of inverter 1 in unit 2']) #   Correlation Coefficient = 0.987442612006051

plt.scatter(df["Power of inverter 1 in unit 1"],df["Power of inverter 2 in unit 2"], color ='red')
df['Power of inverter 1 in unit 1'].corr(df['Power of inverter 2 in unit 2']) #   Correlation Coefficient = 0.9888663802697253

#SCATTER PLOTS AND CORRELATION OF Power of inverter 2 in unit 1

plt.scatter(df["Power of inverter 2 in unit 1"],df["Power of inverter 1 in unit 1"], color ='red')
df['Power of inverter 2 in unit 1'].corr(df['Power of inverter 1 in unit 1']) # Correlation Coefficient = 0.9883180626693995

plt.scatter(df["Power of inverter 2 in unit 1"],df["Power of inverter 1 in unit 2"], color ='red')
df['Power of inverter 2 in unit 1'].corr(df['Power of inverter 1 in unit 2']) # Correlation Coefficient = 0.984097626125095

plt.scatter(df["Power of inverter 2 in unit 1"],df["Power of inverter 2 in unit 2"], color ='red')
df['Power of inverter 2 in unit 1'].corr(df['Power of inverter 2 in unit 2']) # Correlation Coefficient = 0.985222898384228

#SCATTER PLOTS AND CORRELATION OF Power of inverter 1 in unit 2

plt.scatter(df["Power of inverter 1 in unit 2"],df["Power of inverter 1 in unit 1"], color ='red')
df['Power of inverter 1 in unit 2'].corr(df['Power of inverter 2 in unit 2'])#   Correlation Coefficient = 0.998616327975247

plt.scatter(df["Power of inverter 1 in unit 2"],df["Power of inverter 2 in unit 1"], color ='red')
df['Power of inverter 1 in unit 2'].corr(df['Power of inverter 2 in unit 1'])#   Correlation Coefficient = 0.9840976261250951

plt.scatter(df["Power of inverter 1 in unit 2"],df["Power of inverter 2 in unit 2"], color ='red')
df['Power of inverter 1 in unit 2'].corr(df['Power of inverter 2 in unit 2'])#   Correlation Coefficient = 0.998616327975247

#SCATTER PLOTS AND CORRELATION OF Power of inverter 2 in unit 2

plt.scatter(df["Power of inverter 2 in unit 2"],df["Power of inverter 1 in unit 1"], color ='red')
df['Power of inverter 2 in unit 2'].corr(df['Power of inverter 1 in unit 1'])#   Correlation Coefficient = 0.9888663802697252

plt.scatter(df["Power of inverter 2 in unit 2"],df["Power of inverter 2 in unit 1"], color ='red')
df['Power of inverter 2 in unit 2'].corr(df['Power of inverter 2 in unit 1'])#   Correlation Coefficient = 0.985222898384228

plt.scatter(df["Power of inverter 2 in unit 2"],df["Power of inverter 1 in unit 2"], color ='red')
df['Power of inverter 2 in unit 2'].corr(df['Power of inverter 1 in unit 2'])#   Correlation Coefficient = 0.998616327975247



# DATA PREPROCESSING---------------------
 

print(df.columns)    # column names


df = df.rename(columns={"ï»¿DATE/TIME": "DATE/TIME"})       #rename column name


# No: of rows and columns

print(df.shape)

#HANDLING MISSING VALUES

df.isnull().sum()
df.fillna(method='ffill', inplace=True)


#TYPE CASTING-----------------

df['DATE/TIME'] = pd.to_datetime(df['DATE/TIME'], dayfirst=True, errors='coerce')
df['Power of inverter 1 in unit 1'] = df['Power of inverter 1 in unit 1'].astype(float)
df['Power of inverter 2 in unit 1'] = df['Power of inverter 2 in unit 1'].astype(float)
df['Power of inverter 1 in unit 2'] = df['Power of inverter 1 in unit 2'].astype(float)
df['Power of inverter 2 in unit 2'] = df['Power of inverter 2 in unit 2'].astype(float)
print(df.dtypes)

#HANDLING DUPLICATE

print(df.duplicated().sum()) 
df.drop_duplicates(inplace=True)

#ZERO VARIANCE-----------------

# Select numerical columns only-------------------

numeric_columns = df.select_dtypes(include=['number'])

# Identifying zero variance columns------------
zero_variance_cols = numeric_columns.var() == 0

# Print columns with zero variance
print("Zero Variance Columns:\n", numeric_columns.columns[zero_variance_cols])


#HANDLING OUTLIERS

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

# Define lower and upper bounds

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR


# Detect outliers

outliers = ((df < lower_bound) | (df > upper_bound)).sum()
print("Number of outliers in each column:\n", outliers)

#TRANSFORMATION--------------


import pylab
from sklearn.preprocessing import PowerTransformer


#SELECTING NUMERICAL COLUMNS------------


numerical_column = df.select_dtypes(include=['float64']).columns

#Q-Q Plots BEFORE transformation--------------------



for col in numerical_column:
    plt.figure(figsize=(8, 6))
    stats.probplot(df[col], dist=stats.norm, plot=pylab)
    plt.title(f"Q-Q Plot for {col} BEFORE Transformation")
    plt.show()

#Apply Yeo-Johnson Transformation----------------------


p_t = PowerTransformer(method='yeo-johnson')
df[numerical_column] = p_t.fit_transform(df[numerical_column])

#Q-Q Plots AFTER transformation--------------

for col in numerical_column:
    plt.figure(figsize=(8, 6))
    stats.probplot(df[col], dist=stats.norm, plot=pylab)
    plt.title(f"Q-Q Plot for {col} AFTER Transformation")
    plt.show()
    
    
transformed_inverter = "Transformed_Inverter_Data.csv"
df.to_csv(transformed_inverter, index=False)

print(f"Transformed dataset saved as: {transformed_inverter}")





