import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import seaborn as sns
from pathlib import Path
import numpy as np 
import matplotlib

plt.style.use('seaborn-white')
# sns.set()
font = {'family' : 'sans-serif',
        'size'   : 16}
matplotlib.rc('font', **font)
sns.set_palette('mako')
# colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def coverage(y, yL, yH):
    return (100 / y.shape[0] * ((y>yL)&(y<yH)).sum())

# df_train = pd.read_csv(Path(__file__).resolve().parent / "qr_predictions_train.csv")
# df_test = pd.read_csv(Path(__file__).resolve().parent / "qr_predictions_test.csv
df_train = np.load(Path(__file__).resolve().parent / "qr_predictions_train.npy",
                   allow_pickle=True)[()]
df_test = np.load(Path(__file__).resolve().parent / "qr_predictions_test.npy",
                  allow_pickle=True)[()]

for i in range(5):
    for set in ['train', 'test']:
        y_true = (df_train['y_train (k={})'.format(i)] 
                  if set == 'train' 
                  else df_test['y_test (k={})'.format(i)])
        y_median = (df_train['y_train_pred (q=0.500) (k={})'.format(i)] 
                  if set == 'train' else 
                  df_test['y_test_pred (q=0.500) (k={})'.format(i)])
        y_low = (df_train['y_train_pred (q=0.025) (k={})'.format(i)] 
                  if set == 'train' else 
                  df_test['y_test_pred (q=0.025) (k={})'.format(i)])
        y_high = (df_train['y_train_pred (q=0.975) (k={})'.format(i)] 
                  if set == 'train' else 
                  df_test['y_test_pred (q=0.975) (k={})'.format(i)])
        plt.figure(figsize=(14, 6))
        plt.plot(np.arange(1, len(y_true)+1), sorted(y_median), 
                 label='predicted median', color='k')
        plt.scatter(np.arange(1, len(y_true)+1), y_true[np.argsort(y_median)], 
                    marker='o', color='darkblue', 
                    label='actual validation data')
        plt.fill_between(np.arange(1, len(y_true)+1), 
                         y_low[np.argsort(y_median)], 
                         y_high[np.argsort(y_median)], alpha=0.3, 
                         color='green',
                         label='95% confidence interval')
        plt.grid(True, which='major', linestyle='-', 
                 linewidth='0.25')#, color='gray')
        plt.ylabel('Rupture Life [hrs]')
        plt.xlabel('Data index in ascending order')
        plt.ticklabel_format(style='sci', scilimits=(-3,4), axis='y')
        plt.legend()
        plt.savefig(Path(__file__).resolve().parent / "{}_parity_{}.png".format(
            set, i), bbox_inches='tight')
        plt.show()
        print(f"Coverage ({set}, {i}): {coverage(y_true, y_low, y_high)}")
        print(f"Upper coverage ({set}, {i}): {coverage(y_true, y_low, np.inf)}")
        print(f"Lower coverage ({set}, {i}): {coverage(y_true, -np.inf, y_high)}")
        print(f"Pearson R2 ({set}, {i}): {pearsonr(y_true, y_median)[0]}")
        print(f"R2 ({set}, {i}): {r2_score(y_true, y_median)}")
        print(f"RMSE ({set}, {i}): {np.sqrt(mean_squared_error(y_true, y_median))}")