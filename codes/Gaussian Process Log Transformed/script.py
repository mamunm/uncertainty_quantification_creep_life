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

data = np.load(Path(__file__).resolve().parents[2] / f"data/chrome_data.npy", 
               allow_pickle=True)[()]

df_train = {}
df_test = {}

kf = KFold(n_splits=5, shuffle=True, random_state=123)
kernel = C(1.0) * Matern(length_scale=1.0) + WhiteKernel(noise_level=1.0) + C(1.0) * DotProduct(sigma_0=1.0)

for i, (tr, ts) in enumerate(kf.split(data['y'])):
    X_train, X_test = data['X'][tr], data['X'][ts]
    scale = StandardScaler()
    X_train = scale.fit_transform(X_train)
    X_test = scale.transform(X_test)
    y_train, y_test = data['y'][tr], data['y'][ts]
    df_train['y_train (k={})'.format(i)] = y_train
    df_test['y_test (k={})'.format(i)] = y_test
    model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=8, normalize_y=True)
    model.fit(X_train, np.log(y_train))
    mu_z, std_z = model.predict(X_train, return_std=True)
    y_pred_train = np.exp(mu_z + std_z**2/2)
    y_pred_unc_train = np.sqrt(std_z**2*y_pred_train**2)
    mu_z, std_z = model.predict(X_test, return_std=True)
    y_pred_test = np.exp(mu_z + std_z**2/2)
    y_pred_unc_test = np.sqrt(std_z**2*y_pred_test**2)
    print(f'pearsonr: {pearsonr(y_test, y_pred_test)[0]}')
    
    K = model.kernel_(model.X_train_)
    K[np.diag_indices_from(K)] += model.alpha
    sns.heatmap(K, cmap='mako')
    plt.title('Components of Kernel Matrix')
    plt.savefig(Path(__file__).resolve().parent / 'heatmap_{}.png'.format(i))
    plt.clf()

    df_train['y_train_pred (q=0.025) (k={})'.format(i)] = y_pred_train - 2 * y_pred_unc_train
    df_test['y_test_pred (q=0.025) (k={})'.format(i)] = y_pred_test - 2 * y_pred_unc_test
    df_train['y_train_pred (q=0.500) (k={})'.format(i)] = y_pred_train
    df_test['y_test_pred (q=0.500) (k={})'.format(i)] = y_pred_test
    df_train['y_train_pred (q=0.975) (k={})'.format(i)] = y_pred_train + 2 * y_pred_unc_train
    df_test['y_test_pred (q=0.975) (k={})'.format(i)] = y_pred_test + 2 * y_pred_unc_test
    
#df_train.to_csv(Path(__file__).resolve().parent / 'qr_predictions_train.csv', index=False)
#df_test.to_csv(Path(__file__).resolve().parent / 'qr_predictions_test.csv', index=False)       
np.save(Path(__file__).resolve().parent / 'qr_predictions_train.npy', df_train)
np.save(Path(__file__).resolve().parent / 'qr_predictions_test.npy', df_test)     