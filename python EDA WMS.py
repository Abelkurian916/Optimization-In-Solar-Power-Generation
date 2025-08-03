# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 22:08:53 2025

@author: abelk
"""

import pandas as pd
import numpy as np
import sqlalchemy
from sqlalchemy import create_engine

engine = create_engine("mysql+pymysql://root:abelkurian123@localhost:3306/solar_energy")

conn = engine.connect()

sql = "SELECT * FROM solar_energy.`wms report`;"

df = pd.read_sql_query(sql, conn)

from scipy.stats import skew, kurtosis


# mean, median, mode of GII coloumn

mean_value = df['GII'].mean()

median_value = df['GII'].median()

mode_value = df['GII'].mode()


print('Mean =',mean_value)
print('Median =',median_value)
print('Mode=',mode_value)


# mean, median, mode of MODULE TEMP.1 coloumn

mean_value2 = df['MODULE TEMP.1'].mean()

median_value2 = df['MODULE TEMP.1'].median()

mode_value2 = df['MODULE TEMP.1'].mode()


print('Mean =',mean_value2)
print('Median =',median_value2)
print('Mode=',mode_value2)


# mean, median, mode of RAIN coloumn

mean_value3 = df['RAIN'].mean()

median_value3 = df['RAIN'].median()

mode_value3 = df['RAIN'].mode()


print('Mean =',mean_value3)
print('Median =',median_value3)
print('Mode=',mode_value3)



# mean, median, mode of AMBIENT TEMPRETURE coloumn

mean_value4 = df['AMBIENT TEMPRETURE'].mean()

median_value4 = df['AMBIENT TEMPRETURE'].median()

mode_value4 = df['AMBIENT TEMPRETURE'].mode()


print('Mean =',mean_value4)
print('Median =',median_value4)
print('Mode=',mode_value4)


# variance of GII

variance1_value = df['GII'].var()
print('variance of GII = ',variance1_value)


# variance of MODULE TEMP.1

variance2_value = df['MODULE TEMP.1'].var()
print('variance of MODULE TEMP.1 = ', variance2_value)

# variance of RAIN

variance3_value = df['RAIN'].var()
print('variance of RAIN = ',variance3_value)

# variance of AMBIENT TEMPRETURE
variance4_value = df['AMBIENT TEMPRETURE'].var()
print('variance of AMBIENT TEMPRETURE = ',variance4_value )


# Standared Deviation of GII

std_dev_value1 = df['GII'].std()
print('Standared Deviation of GII = ',std_dev_value1)

# Standared Deviation of MODULE TEMP.1

std_dev_value2 = df['MODULE TEMP.1'].std()
print('Standared Deviation of MODULE TEMP.1 = ',std_dev_value2)


# Standared Deviation of RAIN

std_dev_value3 = df['RAIN'].std()
print('Standared Deviation of RAIN = ',std_dev_value3)


# Standared Deviation of AMBIENT TEMPRETURE

std_dev_value4 = df['AMBIENT TEMPRETURE'].std()
print('Standared Deviation of AMBIENT TEMPRETURE = ',std_dev_value4)



#Range of Power of GII
range1_value = df['GII'].max()-df['GII'].min()
print('Range of GII = ',range1_value)


#Range of MODULE TEMP.1

range2_value = df['MODULE TEMP.1'].max()-df['MODULE TEMP.1'].min()
print('Range of MODULE TEMP.1 = ',range2_value)


#Range of RAIN

range3_value = df['RAIN'].max()-df['RAIN'].min()
print('RAIN = ',range3_value)


#Range of AMBIENT TEMPRETURE

range4_value = df['AMBIENT TEMPRETURE'].max()-df['AMBIENT TEMPRETURE'].min()
print('AMBIENT TEMPRETURE = ',range4_value)


#skewness of GII
skew_value1 = skew(df['GII'], nan_policy='omit')
print('GII =',skew_value1)


#skewness of MODULE TEMP.1
skew_value2 = skew(df['MODULE TEMP.1'], nan_policy='omit')
print('skewness of MODULE TEMP.1 =',skew_value2)


#skewness of RAIN

skew_value3 = skew(df['RAIN'], nan_policy='omit')
print('skewness of RAIN =',skew_value3)


#skewness of AMBIENT TEMPRETURE

skew_value4 = skew(df['AMBIENT TEMPRETURE'], nan_policy='omit')
print('skewness of AMBIENT TEMPRETURE =',skew_value4)


# Kurtosis of GII
kurt1_value = kurtosis(df['GII'])
print('Kurtosis of GII =', kurt1_value)

# Kurtosis of MODULE TEMP.1
kurt2_value = kurtosis(df['MODULE TEMP.1'])
print('Kurtosis of MODULE TEMP.1 =', kurt2_value)

# Kurtosis of RAIN
kurt3_value = kurtosis(df['RAIN'])
print('RAIN =', kurt3_value)

# Kurtosis of AMBIENT TEMPRETURE
kurt4_value = kurtosis(df['AMBIENT TEMPRETURE'])
print('AMBIENT TEMPRETURE =', kurt4_value)




#GRAPHICAL REPRESENTATION 
##UNIVARIATE

import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

##Histogram 


sns.histplot(df["GII"], bins=20, kde=True, color="green")
sns.histplot(df["MODULE TEMP.1"], bins=20, kde=True, color="blue")
sns.histplot(df["RAIN"], bins=20, kde=True, color="blue")
sns.histplot(df["AMBIENT TEMPRETURE"], bins=20, kde=True, color="blue")

##Box Plot 


sns.boxplot(y=df["GII"], color="blue")
sns.boxplot(y=df["MODULE TEMP.1"], color="orange")
sns.boxplot(y=df["RAIN"], color="orange")
sns.boxplot(y=df["AMBIENT TEMPRETURE"], color="orange")


##Q-Q (Quantile-Quantile) Plot


stats.probplot(df["GII"], dist="norm", plot=plt)
stats.probplot(df["MODULE TEMP.1"], dist="norm", plot=plt)
stats.probplot(df["RAIN"], dist="norm", plot=plt)
stats.probplot(df["AMBIENT TEMPRETURE"], dist="norm", plot=plt)


##Density plot 


sns.kdeplot(df['GII'], shade=True, color="blue", linewidth=2)
sns.kdeplot(df['MODULE TEMP.1'], shade=True, color="blue", linewidth=2)
sns.kdeplot(df['RAIN'], shade=True, color="blue", linewidth=2)
sns.kdeplot(df['AMBIENT TEMPRETURE'], shade=True, color="blue", linewidth=2)



#BIVARIATE ANALYSIS

##SCATTER PLOTS AND CORRELATION OF GII

plt.scatter(df["GII"],df["MODULE TEMP.1"], color ='red')
df['GII'].corr(df['MODULE TEMP.1']) #   Correlation Coefficient = 0.8368293330731253

plt.scatter(df["GII"],df["RAIN"], color ='red')
df['GII'].corr(df['RAIN']) #   Correlation Coefficient = -0.0012220475274169335

plt.scatter(df["GII"],df["AMBIENT TEMPRETURE"], color ='red')
df['GII'].corr(df['AMBIENT TEMPRETURE']) #   Correlation Coefficient = 0.48365741272978063

##SCATTER PLOTS AND CORRELATION OF MODULE TEMP.1

plt.scatter(df["MODULE TEMP.1"],df["GII"], color ='red')
df['MODULE TEMP.1'].corr(df['GII']) # Correlation Coefficient = 0.8368293330731255

plt.scatter(df["MODULE TEMP.1"],df["RAIN"], color ='red')
df['MODULE TEMP.1'].corr(df['RAIN']) # Correlation Coefficient = -0.01849828932212484

plt.scatter(df["MODULE TEMP.1"],df["AMBIENT TEMPRETURE"], color ='red')
df['MODULE TEMP.1'].corr(df['AMBIENT TEMPRETURE']) # Correlation Coefficient = 0.705913949887113

##SCATTER PLOTS AND CORRELATION OF RAIN

plt.scatter(df["RAIN"],df["GII"], color ='red')
df['RAIN'].corr(df['GII'])#   Correlation Coefficient = -0.0012220475274169332

plt.scatter(df["RAIN"],df["MODULE TEMP.1"], color ='red')
df['RAIN'].corr(df['MODULE TEMP.1'])#   Correlation Coefficient = -0.018498289322124835

plt.scatter(df["RAIN"],df["AMBIENT TEMPRETURE"], color ='red')
df['RAIN'].corr(df['AMBIENT TEMPRETURE'])# Correlation Coefficient = -0.019944507668646288

##SCATTER PLOTS AND CORRELATION OF AMBIENT TEMPRETURE

plt.scatter(df["AMBIENT TEMPRETURE"],df["GII"], color ='red')
df['AMBIENT TEMPRETURE'].corr(df['GII'])#   Correlation Coefficient = 0.4836574127297806

plt.scatter(df["AMBIENT TEMPRETURE"],df["MODULE TEMP.1"], color ='red')
df['AMBIENT TEMPRETURE'].corr(df['MODULE TEMP.1'])#   Correlation Coefficient = 0.705913949887113

plt.scatter(df["AMBIENT TEMPRETURE"],df["RAIN"], color ='red')
df['AMBIENT TEMPRETURE'].corr(df['RAIN'])#   Correlation Coefficient = -0.019944507668646288




#DATA PROCESSING-------------


df = df.rename(columns={"AMBIENT TEMPRETURE":"AMBIENT TEMPERATURE"}) #CHANGED COLUMN NAME
print(df.columns)    # column names

# No: of rows and columns
print(df.shape)

#HANDLING MISSING VALUES

df.isnull().sum()
df.fillna(method='ffill', inplace=True)

#TYPE CASTING-----------------

df['DATE & TIME'] = pd.to_datetime(df['DATE & TIME'], dayfirst=True, errors='coerce')
df['GII'] = df['GII'].astype(int)
df['MODULE TEMP.1'] = df['MODULE TEMP.1'].astype(float)
df['RAIN'] = df['RAIN'].astype(float)
df['AMBIENT TEMPERATURE'] = df['AMBIENT TEMPERATURE'].astype(float)
print(df.dtypes)

#HANDLING DUPLICATE

print(df.duplicated().sum()) 
df.drop_duplicates(inplace=True)

#ZERO VARIANCE---------

# Select numeric columns----------------


numeric_cols = df.select_dtypes(include=['float64', 'int64'])

# Find columns with zero variance---------------

zero_variance_cols = numeric_cols.var() == 0

# Print zero variance columns------------

print("Zero Variance Columns:\n", zero_variance_cols[zero_variance_cols].index.tolist())



#DETECTING OUTLIERS USING BOXPLOT

sns.boxplot(y=df["GII"], color="blue")
sns.boxplot(y=df["MODULE TEMP.1"], color="orange")
sns.boxplot(y=df["RAIN"], color="orange")
sns.boxplot(y=df["AMBIENT TEMPERATURE"], color="orange")



#HANDLING OUTLIERS

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

# Define lower and upper bounds

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# FILTER OUTLIERS...........
df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]


# Detect outliers

outliers = ((df < lower_bound) | (df > upper_bound)).sum()
print("Number of outliers in each column:\n", outliers)


#Discretization --------------------

# Check unique values and summary statistics.............

print(df['RAIN'].describe())
print(df['RAIN'].unique())
# Define bin edges---------------

bin_edges = [df['RAIN'].min(), 0.0001, df['RAIN'].max()]  # Use small positive value instead of 0
bin_labels = ['No Rain', 'Rain']

# Apply discretization
df['RAIN_category'] = pd.cut(df['RAIN'], bins=bin_edges, labels=bin_labels, include_lowest=True)

# Check output
print(df[['RAIN', 'RAIN_category']].head(10))

print(df['RAIN_category'].value_counts())


# Scaling (if needed)--------------------------
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])


#TRANSFORMATION--------------------

from sklearn.preprocessing import StandardScaler, PowerTransformer
import pylab
import scipy.stats as stats
import matplotlib.pyplot as plt

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

#Q-Q Plots BEFORE transformation-----------------------
for col in numeric_cols:
    plt.figure(figsize=(8, 6))
    stats.probplot(df[col], dist=stats.norm, plot=pylab)
    plt.title(f"Q-Q Plot for {col} BEFORE Transformation")
    plt.show()

#Apply Yeo-Johnson Transformation
pt = PowerTransformer(method='yeo-johnson')
df[numeric_cols] = pt.fit_transform(df[numeric_cols])

#Q-Q Plots AFTER transformation
for col in numeric_cols:
    plt.figure(figsize=(8, 6))
    stats.probplot(df[col], dist=stats.norm, plot=pylab)
    plt.title(f"Q-Q Plot for {col} AFTER Transformation")
    plt.show()

transformed_WMS = "Transformed_WMS_REPORT.csv"
df.to_csv(transformed_WMS, index=False)

print(f"Transformed dataset saved as: {transformed_WMS}")














