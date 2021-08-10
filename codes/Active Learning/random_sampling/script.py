import pandas as pd 
import numpy as np 
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, ConstantKernel as C,
                                              Matern, WhiteKernel, DotProduct)
import seaborn as sns
import matplotlib.pyplot as plt

data = np.load(Path(__file__).resolve().parents[3] / f"data/chrome_data.npy", 
               allow_pickle=True)[()]

kernel = C(1.0) * Matern(length_scale=1.0) + WhiteKernel(noise_level=1.0) + C(1.0) * DotProduct(sigma_0=1.0)
PCC = []
R2 = []

for i in range(10):
    if i == 0:
        X_train, X_test, y_train, y_test = train_test_split(
            data['X'], data['y'], train_size=0.2, shuffle=True)
    else:
        X_test, X_add, y_test, y_add = train_test_split(X_test, y_test, 
                            test_size=0.05, shuffle=True)
        X_train = np.append(X_train, X_add, axis=0)
        y_train = np.append(y_train, y_add, axis=0)
    scale = StandardScaler()
    X_train_scaled = scale.fit_transform(X_train)
    X_test_scaled = scale.transform(X_test)
    
    model = GaussianProcessRegressor(kernel=kernel, 
                                     n_restarts_optimizer=8, 
                                     normalize_y=True)
    model.fit(X_train_scaled, np.log(y_train))
    mu_z, std_z = model.predict(scale.transform(X_test), return_std=True)
    y_pred = np.exp(mu_z + std_z**2/2)
    y_pred_unc = np.sqrt(std_z**2*y_pred**2)
    print(f"pearsonr: {pearsonr(y_test, y_pred)[0]}")
    PCC.append([len(X_train_scaled), pearsonr(y_test, y_pred)[0]])
    print(f"R2: {r2_score(y_test, y_pred)}")
    R2.append([len(X_train_scaled), r2_score(y_test, y_pred)])
    

n_data = [li[0] for li in R2]
pcc = [li[1] for li in R2]
with plt.style.context('seaborn-white'):
    plt.figure(figsize=(8, 8))
    plt.plot(n_data, pcc, 
                c='r', label='PCC')
    plt.title('R2 over the number of data points')
    plt.legend()
    plt.xlabel("Number of data points")
    plt.ylabel("R2")
    plt.show() 
#df_train.to_csv(Path(__file__).resolve().parent / 'qr_predictions_train.csv', index=False)
#df_test.to_csv(Path(__file__).resolve().parent / 'qr_predictions_test.csv', index=False)       
# np.save(Path(__file__).resolve().parent / 'qr_predictions_train.npy', df_train)
# np.save(Path(__file__).resolve().parent / 'qr_predictions_test.npy', df_test)     