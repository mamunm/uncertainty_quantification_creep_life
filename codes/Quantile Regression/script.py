import pandas as pd 
import numpy as np 
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

data = np.load(Path(__file__).resolve().parents[2] / f"data/chrome_data.npy", 
               allow_pickle=True)[()]

df_train = {}
df_test = {}

quantiles = [0.025, 0.5, 0.975]
kf = KFold(n_splits=5, shuffle=True, random_state=123)


for i, (tr, ts) in enumerate(kf.split(data['y'])):
    X_train, X_test = data['X'][tr], data['X'][ts]
    scale = StandardScaler()
    X_train = scale.fit_transform(X_train)
    X_test = scale.transform(X_test)
    y_train, y_test = data['y'][tr], data['y'][ts]
    X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.1)
    df_train['y_train (k={})'.format(i)] = y_train
    df_test['y_test (k={})'.format(i)] = y_test
    for q in quantiles:
        parameters = {'loss_function': 'Quantile:alpha={:0.2f}'.format(q),
                  'num_boost_round': 5000}
        model = CatBoostRegressor(**parameters)
        model.fit(X_train, np.log(y_train),
                  eval_set=[(X_val, np.log(y_val))],
                  early_stopping_rounds=20,
                  verbose=False)
        y_pred = np.exp(model.predict(X_test))
        print(f'q: {q} | pearsonr: {pearsonr(y_test, y_pred)[0]}')
        df_train['y_train_pred (q={:0.3f}) (k={})'.format(q, i)] = np.exp(
            model.predict(X_train))
        df_test['y_test_pred (q={:0.3f}) (k={})'.format(q, i)] = np.exp(
            model.predict(X_test))
    
#df_train.to_csv(Path(__file__).resolve().parent / 'qr_predictions_train.csv', index=False)
#df_test.to_csv(Path(__file__).resolve().parent / 'qr_predictions_test.csv', index=False)       
np.save(Path(__file__).resolve().parent / 'qr_predictions_train.npy', df_train)
np.save(Path(__file__).resolve().parent / 'qr_predictions_test.npy', df_test)   