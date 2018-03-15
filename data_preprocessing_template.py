import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder

# import the dataset from .csv file
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

# encoding the categorical data in the dataset
# first column - "Country", because has only 3 possible values (France/Germany/Spain)
# encoding mean that we transform our possible values(strings) into possible values(numbers)
# France - 0 / Germany - 1 / Spain - 2
label_encoder_x = LabelEncoder()
x[:, 0] = label_encoder_x.fit_transform(x[:, 0])

# Replacing the 1 first column(categorical feature) with country value with
# 3(total count of possible values) columns with numbers 1/0
one_hot_encoder = OneHotEncoder(categorical_features=[0])
x = one_hot_encoder.fit_transform(x).toarray()


# encoding the categorical data in the dataset
# so our "Purchased" column with Yes/No values will be 1/0
label_encoder_y = LabelEncoder()
y = label_encoder_x.fit_transform(y)