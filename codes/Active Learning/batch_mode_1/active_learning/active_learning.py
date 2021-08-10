import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, ConstantKernel as C,
                                              Matern, WhiteKernel, DotProduct)
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from pathlib import Path
from sklearn.metrics import r2_score

plt.style.use('seaborn-white')
# sns.set()
font = {'family' : 'sans-serif',
        'size'   : 16}
matplotlib.rc('font', **font)
sns.set_palette('mako')

def coverage(y, yL, yH):
    return (100 / y.shape[0] * ((y>yL)&(y<yH)).sum())

class ActiveLearning:
    def __init__(self, 
                 data: np.ndarray,
                 plot_path: str):
        
        self.X = data['X']
        self.y = data['y']
        self.plot_path = plot_path
        self.pca = PCA(n_components=2).fit(self.X)
        self.idx = np.arange(len(self.y))
        self.history = {}
        
    def run(self, 
            n_iter: int = 10, 
            initial_samples_ratio: int = 0.2,
            batch_size: int = 10):
        print(f"Performing iteration : 0")
        self.X_train, _, self.y_train, _, self.idx_train, self.idx_test = train_test_split(
            self.X, self.y, self.idx, train_size=initial_samples_ratio)
        scale = StandardScaler()
        self.X_train_scaled = scale.fit_transform(self.X_train)
        kernel = C(1.0) * Matern(length_scale=1.0) + WhiteKernel(noise_level=1.0) + C(1.0) * DotProduct(sigma_0=1.0)
        model = GaussianProcessRegressor(kernel=kernel, 
                                         n_restarts_optimizer=8, 
                                         normalize_y=True)
        model.fit(self.X_train_scaled, np.log(self.y_train))
        self.X_scaled = scale.transform(self.X)
        mu_z, std_z = model.predict(self.X_scaled[self.idx_test], return_std=True)
        self.y_pred_test = np.exp(mu_z + std_z**2/2)
        self.y_pred_unc_test = np.sqrt(std_z**2*self.y_pred_test**2)
        mu_z, std_z = model.predict(self.X_scaled, return_std=True)
        self.y_pred = np.exp(mu_z + std_z**2/2)
        self.y_pred_unc = np.sqrt(std_z**2*self.y_pred**2)
        self.plot_error_bounds(0)
        self.history[0] = {'y_pred': self.y_pred, 
                           'y_pred_unc': self.y_pred_unc,
                           'train_index': self.idx_train,
                           'test_index': self.idx_test,
                           'PCC': pearsonr(self.y[self.idx_test], self.y_pred_test)[0],
                           'R2': r2_score(self.y[self.idx_test], self.y_pred_test)}

        
        for i in range(1, n_iter):
            print(f"Performing iteration : {i}")
            q_points = self.get_query_points(batch_size, i)
            self.plot_new_queries(q_points, i)
            self.X_train = np.vstack((self.X_train, self.X[q_points]))
            self.y_train = np.hstack((self.y_train, self.y[q_points]))
            scale = StandardScaler()
            self.X_train_scaled = scale.fit_transform(self.X_train)
            kernel = C(1.0) * Matern(length_scale=1.0) + WhiteKernel(noise_level=1.0) + C(1.0) * DotProduct(sigma_0=1.0)
            model = GaussianProcessRegressor(kernel=kernel, 
                                             n_restarts_optimizer=8, 
                                             normalize_y=True)
            model.fit(self.X_train_scaled, np.log(self.y_train))
            mu_z, std_z = model.predict(scale.transform(self.X[self.idx_test]), 
                                        return_std=True)
            self.y_pred_test = np.exp(mu_z + std_z**2/2)
            self.y_pred_unc_test = np.sqrt(std_z**2*self.y_pred_test**2)
            mu_z, std_z = model.predict(scale.transform(self.X), 
                                        return_std=True)
            self.y_pred = np.exp(mu_z + std_z**2/2)
            self.y_pred_unc = np.sqrt(std_z**2*self.y_pred**2)
            self.plot_error_bounds(i)
            self.history[i] = {'y_pred': self.y_pred, 
                               'y_pred_unc': self.y_pred_unc,
                               'train_index': self.idx_train,
                               'test_index': self.idx_test,
                               'PCC': pearsonr(self.y[self.idx_test], self.y_pred_test)[0],
                               'R2': r2_score(self.y[self.idx_test], self.y_pred_test)}
        self.plot_pcc_history()
        self.plot_r2_history()
        print("Finished Successfully!")
    
    def get_query_points(self, batch_size: int, iteration: int):
        batch_size = min(batch_size, len(self.idx_test))
        queries = []
        for i in range(batch_size):
            X_pooled = self.X_scaled[self.idx_test]
            X_train = self.X_scaled[self.idx_train]
            alpha = len(X_train) / len(self.X_scaled)
            U = self.y_pred_unc[self.idx_test]
            sim = pairwise_distances(
                X_pooled, X_train).min(1)
            sim_scores = 1 / (1 + sim)
            U_scores = np.tanh(U)
            scores = alpha * sim_scores + (1 - alpha) * U_scores
            if i == 0:
                self.plot_score(self.X[self.idx_train], 
                                self.X[self.idx_test], 
                                scores, 
                                iteration)
            q_ind_ind = np.argmax(scores)
            self.idx_train = np.append(self.idx_train, 
                                       self.idx_test[q_ind_ind])
            queries.append(self.idx_test[q_ind_ind])
            self.idx_test = np.delete(self.idx_test, q_ind_ind)
        return np.array(queries)
            
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
        
        