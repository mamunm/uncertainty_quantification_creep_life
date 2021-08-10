import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import numpy as np
from .utils import coverage


plt.style.use('seaborn-white')
# sns.set()
font = {'family' : 'sans-serif',
        'size'   : 16}
matplotlib.rc('font', **font)
sns.set_palette('mako')


def plot_error_bounds(self, n_iter: int):
        y_low = sorted(self.y_pred) - 2 * self.y_pred_unc[
            np.argsort(self.y_pred)]
        y_high = sorted(self.y_pred) + 2 * self.y_pred_unc[
            np.argsort(self.y_pred)]
        cov = coverage(self.y, y_low, y_high)
        plt.title(f"Coverage: {cov:0.2f}")
        plt.figure(figsize=(14, 6))
        plt.plot(np.arange(1, len(self.y_pred)+1), sorted(self.y_pred), 
                 label='predicted median', color='k')
        plt.scatter(np.arange(1, len(self.y_pred)+1), 
                    self.y[np.argsort(self.y_pred)], 
                    marker='o', color='darkblue', 
                    label='actual validation data')
        plt.fill_between(np.arange(1, len(self.y_pred)+1), 
                         y_low, y_high, alpha=0.3, 
                         color='green',
                         label='95% confidence interval')
        plt.grid(True, which='major', linestyle='-', 
                 linewidth='0.25')#, color='gray')
        plt.ylabel('Rupture Life [hrs]')
        plt.xlabel('Data index in ascending order')
        plt.ticklabel_format(style='sci', scilimits=(-3,4), axis='y')
        plt.legend()
        plt.savefig(self.plot_path + f'/error_bounds_{n_iter}.png', 
                    bbox_inches='tight')
        plt.close()
    
def plot_score(self, X_train, X_pooled, scores, iteration):
    X_train_pca = self.pca.transform(X_train)
    X_pooled_pca = self.pca.transform(X_pooled)
    with plt.style.context('seaborn-white'):
        plt.figure(figsize=(8, 8))
        plt.scatter(X_pooled_pca[:, 0], X_pooled_pca[:, 1], 
                    c=scores, cmap='viridis')
        plt.colorbar()
        plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], 
                    c='r', label='labeled')
        plt.title(f'Scores at iteration {iteration}')
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.legend()
        plt.savefig(self.plot_path + f'/scores_{iteration}.png', 
                bbox_inches='tight')
        plt.close()
    
def plot_pcc_history(self):
    n_data = [len(self.history[k]['train_index']) for k in self.history]
    pcc = [self.history[k]['PCC'] for k in self.history]
    with plt.style.context('seaborn-white'):
        plt.figure(figsize=(8, 8))
        plt.plot(n_data, pcc, 
                    c='r', label='PCC')
        plt.title('PCC over the number of data points')
        plt.legend()
        plt.xlabel("Number of data points")
        plt.ylabel("PCC")
        plt.savefig(self.plot_path + f'/pcc_history.png', 
                bbox_inches='tight')
        plt.close() 
def plot_r2_history(self):
    n_data = [len(self.history[k]['train_index']) for k in self.history]
    r2 = [self.history[k]['R2'] for k in self.history]
    with plt.style.context('seaborn-white'):
        plt.figure(figsize=(8, 8))
        plt.plot(n_data, r2, 
                    c='r', label='R2')
        plt.title('R2 over the number of data points')
        plt.legend()
        plt.xlabel("Number of data points")
        plt.ylabel("R2")
        plt.savefig(self.plot_path + f'/r2_history.png', 
                bbox_inches='tight')
        plt.close()

def plot_new_queries(self, queries: np.ndarray, iteration: int):
    X_train_pca = self.pca.transform(self.X_train)
    X_batch_pca = self.pca.transform(self.X[queries])
    X_test_pca = self.pca.transform(self.X[self.idx_test])
    with plt.style.context('seaborn-white'):
        plt.figure(figsize=(8, 8))
        plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], 
                    c='0.8', label='unlabeled')
        plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], 
                    c='r', label='labeled')
        plt.scatter(X_batch_pca[:, 0], X_batch_pca[:, 1], 
                    c='k', label='queried')
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.title('The instances selected for labeling')
        plt.legend()
        plt.savefig(self.plot_path + f'/new_queries_{iteration}.png', 
                bbox_inches='tight')
        plt.close() 