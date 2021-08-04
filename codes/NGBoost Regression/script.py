import pandas as pd 
import numpy as np 
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from ngboost import NGBRegressor
from sklearn.model_selection import train_test_split

data = np.load(Path(__file__).resolve().parents[2] / f"data/chrome_data.npy", 
               allow_pickle=True)[()]

df_train = {}
df_test = {}

kf = KFold(n_splits=5, shuffle=True, random_state=123)


for i, (tr, ts) in enumerate(kf.split(data['y'])):
    X_train, X_test = data['X'][tr], data['X'][ts]
    scale = StandardScaler()
    X_train = scale.fit_transform(X_train)
    X_test = scale.transform(X_test)
    y_train, y_test = data['y'][tr], data['y'][ts]
    df_train['y_train (k={})'.format(i)] = y_train
    df_test['y_test (k={})'.format(i)] = y_test
    model = NGBRegressor()
    model.fit(X_train, np.log(y_train))
    y_pred = np.exp(model.predict(X_test))
    print(f'pearsonr: {pearsonr(y_test, y_pred)[0]}')
    y_train_ngb = model.pred_dist(X_train)
    y_test_ngb = model.pred_dist(X_test)
    df_train['y_train_pred (q=0.025) (k={})'.format(i)] = np.exp(
        y_train_ngb.interval(0.95)[0])
    df_test['y_test_pred (q=0.025) (k={})'.format(i)] = np.exp(
        y_test_ngb.interval(0.95)[0])
    df_train['y_train_pred (q=0.500) (k={})'.format(i)] = np.exp(
        model.predict(X_train))
    df_test['y_test_pred (q=0.500) (k={})'.format(i)] = np.exp(
        model.predict(X_test))
    df_train['y_train_pred (q=0.975) (k={})'.format(i)] = np.exp(
        y_train_ngb.interval(0.95)[1])
    df_test['y_test_pred (q=0.975) (k={})'.format(i)] = np.exp(
        y_test_ngb.interval(0.95)[1])
    
#df_train.to_csv(Path(__file__).resolve().parent / 'qr_predictions_train.csv', index=False)
#df_test.to_csv(Path(__file__).resolve().parent / 'qr_predictions_test.csv', index=False)       
np.save(Path(__file__).resolve().parent / 'qr_predictions_train.npy', df_train)
np.save(Path(__file__).resolve().parent / 'qr_predictions_test.npy', df_test)     