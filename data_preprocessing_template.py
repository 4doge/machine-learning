import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Import the dataset from .csv file
dataset = pd.read_csv('data.csv')

# x is features or independent variables,
# for the current dataset it's "Country,Age,Salary" columns,
# without "Purchased" column
x = dataset.iloc[:, :-1].values

# y is dependent variable
# for the current dataset it's "Purchased" column
y = dataset.iloc[:, 3].values

# Transformer for filling the missing data
# missing_values - placeholder for the missing data
# strategy - logic for filling, mean by axis
# axis - 0 for the column, 1 for the row
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

# Encoding the categorical data in the dataset
# first column - "Country", because has only 3 possible values (France/Germany/Spain)
# encoding mean that we transform our possible values(strings) into possible values(numbers)
# France - 0 / Germany - 1 / Spain - 2
label_encoder_x = LabelEncoder()
x[:, 0] = label_encoder_x.fit_transform(x[:, 0])

# Replacing the 1 first column(categorical feature) with country value with
# 3(total count of possible values) columns with numbers 1/0
one_hot_encoder = OneHotEncoder(categorical_features=[0])
x = one_hot_encoder.fit_transform(x).toarray()

# Encoding the categorical data in the dataset
# so our "Purchased" column with Yes/No values will be 1/0
label_encoder_y = LabelEncoder()
y = label_encoder_x.fit_transform(y)

# Split our dataset to training and test sets
# test_size - declaring what part of our dataset we will use as test data in percents
# So if we have 100 objects in our dataset and test_size 0.2 we will get the 20 objects for test and 80 as training
# good values 0.2-0.25
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# Feature scaling
# The "Age" and "Salary" fields not in the same scale rank
# So if we want to do the clear math we need to put both of the into the same scale rank
# e.g. from -1 to 1
# There two ways: normalization & standardisation
standard_scaler_x = StandardScaler()
x_train = standard_scaler_x.fit_transform(x_train)
x_test = standard_scaler_x.transform(x_test)