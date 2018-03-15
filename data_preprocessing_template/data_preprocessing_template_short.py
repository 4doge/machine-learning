import pandas as pd
from sklearn.model_selection import train_test_split

# Import dataset
dataset = pd.read_csv('data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Split the dataset to train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
