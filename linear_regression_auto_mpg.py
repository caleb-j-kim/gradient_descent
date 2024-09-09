# CS 4375.001
# Linear Regression using Gradient Descent on the Auto MPG Dataset
# Saidarsh Tukkadi / SXT200072
# Caleb Kim / CJK200004

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

# load the dataset from the github raw url
url = "https://raw.githubusercontent.com/saidarsht/Auto-MPG-Dataset/main/auto-mpg.data"

# define column names as per the dataset description
columns = [
    'mpg', 'cylinders', 'displacement', 'horsepower', 
    'weight', 'acceleration', 'model_year', 'origin', 'car_name'
]

# read the dataset, specifying the delimiter and column names
data = pd.read_csv(url, delim_whitespace=True, names=columns, na_values='?')

# data preprocessing
data = data.dropna()
data['origin'] = data['origin'].astype('int')
data = pd.get_dummies(data, columns=['origin'], drop_first=True)
data = data.drop(['car_name'], axis=1)

# separate features and target variable
X = data.drop('mpg', axis=1)
y = data['mpg']

# standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# hyperparameter tuning: try different learning rates and iterations
learning_rates = [0.01, 0.001, 0.0001]
n_iterations = [1000, 5000, 10000]

log_file = 'results_log.txt'
with open(log_file, 'w') as log:
    trial = 1
    for lr in learning_rates:
        for n_iter in n_iterations:
            model = SGDRegressor(max_iter=n_iter, tol=1e-3, learning_rate='constant', eta0=lr)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # log the results
            log.write(f'Trial {trial}: Learning Rate: {lr}, Iterations: {n_iter}, MSE: {mse}, R-squared: {r2}\n')
            print(f'Trial {trial}: Learning Rate: {lr}, Iterations: {n_iter}, MSE: {mse}, R-squared: {r2}')
            
            trial += 1

# residuals plot (predicted vs residuals)
plt.figure(figsize=(10, 6))
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, color='purple')
plt.hlines(0, y_pred.min(), y_pred.max(), color='black', linestyles='dashed')
plt.xlabel('Predicted MPG')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Residuals Plot: Predicted vs Residuals')
plt.grid(True)

# final model visualization (actual vs predicted mpg)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
plt.xlabel('Actual MPG')
plt.ylabel('Predicted MPG')
plt.title('Actual vs Predicted MPG')

# show all plots at once
plt.show()
