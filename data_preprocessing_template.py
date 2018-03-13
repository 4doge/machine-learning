import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer

# import the dataset from .csv file
dataset = pd.read_csv('data.csv')

# x is features or independent variables,
# for the current dataset it's "Country,Age,Salary" columns,
# without "Purchased" column
x = dataset.iloc[:, :-1].values

# y is dependent variable
# for the current dataset it's "Purchased" column
y = dataset.iloc[:, 3].values


imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])