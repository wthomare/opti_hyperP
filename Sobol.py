# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from SALib.sample import saltelli
from SALib.analyze import sobol

import concurrent.futures
import seaborn as sns
import matplotlib.pyplot as plt


# Input data
boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.Series(boston.target)


X_train, X_test, y_train, y_test = train_test_split(X, y)


def models(X):
    params = X.tolist()
    # splitting
    regressor = xgb.XGBRegressor(
        base_score=params[0], 
        colsample_bylevel=params[1], 
        colsample_bytree=params[2],
        gamma=params[3], 
        learning_rate=params[4], 
        max_delta_step=params[5], 
        max_depth=int(params[6]),
        n_estimators=int(params[7]), 
        reg_alpha=params[8], 
        reg_lambda=params[9],
        scale_pos_weight=params[10], 
        seed=int(params[11]),  
        subsample=params[12],
        max_leaves=int(params[13]))

    regressor.fit(X_train, y_train)
    
    
    y_pred = regressor.predict(X_test)
    err = mean_squared_error(y_test, y_pred)
    return err



def main(param_values):
    
    np.savetxt("param_values.txt", param_values)
    #Y = np.zeros([param_values.shape[0]])
    with concurrent.futures.ProcessPoolExecutor() as executor:
        test = executor.map(models, param_values)
    return test

if __name__ == '__main__':

    problem = {
        'num_vars': 14,
        'names': ["base_score",
                "colsample_bylevel", 
                "colsample_bytree",
                "gamma", 
                "learning_rate", 
                "max_delta_step", 
                "max_depth",
                "n_estimators", 
                "reg_alpha", 
                "reg_lambda",
                "scale_pos_weight", 
                "seed", 
                "subsample",
                "max_leaves"],
        'bounds': [[0.1, 0.9],
                   [0.1, 0.9],
                   [0.1, 0.9],
                   [0,0.9],
                   [0.1,0.9],
                   [0,0.9],
                   [5,15],
                   [10,1000],
                   [0,1],
                   [0,1],
                   [0,1],
                   [0,10],
                   [0,1],
                   [1,1000]]}
    
    param_values = saltelli.sample(problem, 1024)
        
    Y = np.zeros(param_values.shape[0])
    if not os.path.exists("outputs.txt"): 
        res = main(param_values)
    
        for i, r in enumerate(res):
            Y[i] = r
            np.savetxt("outputs.txt", Y)

    Y = np.loadtxt("outputs.txt", float)
    Si = sobol.analyze(problem, Y)
    total_Si, first_Si, second_Si = Si.to_df()
    
    composant_principale = pd.DataFrame({
        "nom" : problem['names'],
        "value" : Si['S1']
        })


df_params = pd.DataFrame(param_values)

matrix = df_params.corr().round(2)
mask = np.triu(np.ones_like(matrix, dtype=bool))        
                 
sns.heatmap(matrix, annot=True, vmax=1, vmin=-1, center=0, mask=mask)
plt.show()
