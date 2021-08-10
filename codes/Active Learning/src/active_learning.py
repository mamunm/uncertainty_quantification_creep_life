import sys
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, ConstantKernel as C,
                                              Matern, WhiteKernel, DotProduct)
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from pathlib import Path
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans

class ActiveLearning:
    def __init__(self, 
                 data: np.ndarray = None,
                 f_path: str = None,
                 query_strategy: str = 'random',
                 test_size: float = 0.2,
                 n_cluster: int = 10,
                 verbose: bool = True):
        
        if query_strategy == 'cluster_variance_reduction':
            self.n_cluster = n_cluster
        self.X, self.X_test, self.y, self.y_test = train_test_split(
            data['X'], data['y'], shuffle=True, test_size=test_size)
        self.plot_path = Path(__file__).parents[1] / f_path
        if not self.plot_path.exists():
            self.plot_path.mkdir(parents=True)
        self.idx = np.arange(len(self.y))
        self.history = {}
        self.query_strategy = query_strategy
        self.verbose = verbose
        
    def run(self, 
            n_iter: int = 10,
            train_ratio: float = 0.2):
        self.X_train, _, self.y_train, _, self.idx_train, self.idx_pool = train_test_split(
            self.X, self.y, self.idx, train_size=train_ratio, shuffle=True)
        
        for i in range(n_iter):
            print(f"Performing iteration : {i}")
            if i != 0:
                q_points, idx_pool_train = self.get_query_points()
                self.X_train = np.vstack((self.X_train, self.X[q_points]))
                self.y_train = np.hstack((self.y_train, self.y[q_points]))
                self.idx_train = np.append(self.idx_train, idx_pool_train)
                self.idx_pool = np.delete(self.idx_pool, idx_pool_train)
                
            scale = StandardScaler()
            self.X_train_scaled = scale.fit_transform(self.X_train)
            kernel = C(1.0) * Matern(length_scale=1.0) + WhiteKernel(noise_level=1.0) + C(1.0) * DotProduct(sigma_0=1.0)
            model = GaussianProcessRegressor(kernel=kernel, 
                                             n_restarts_optimizer=8, 
                                             normalize_y=True)
            model.fit(self.X_train_scaled, np.log(self.y_train))
            self.X_test_scaled = scale.transform(self.X_test)
            mu_z, std_z = model.predict(self.X_test_scaled, return_std=True)
            self.y_pred_test = np.exp(mu_z + std_z**2/2)
            self.y_pred_unc_test = np.sqrt(std_z**2*self.y_pred_test**2)
            mu_z, std_z = model.predict(self.X_train_scaled, return_std=True)
            self.y_pred_train = np.exp(mu_z + std_z**2/2)
            self.y_pred_unc_train = np.sqrt(std_z**2*self.y_pred_train**2)
            self.X_pool_scaled = scale.transform(self.X[self.idx_pool])
            mu_z, std_z = model.predict(self.X_pool_scaled, return_std=True)
            self.y_pred_pool = np.exp(mu_z + std_z**2/2)
            self.y_pred_unc_pool = np.sqrt(std_z**2*self.y_pred_pool**2)
            self.history[i] = {'model': model,
                               'y_pred_test': self.y_pred_test, 
                               'y_pred_unc_test': self.y_pred_unc_test,
                               'y_pred_train': self.y_pred_train, 
                               'y_pred_unc_train': self.y_pred_unc_train,
                               'train_index': self.idx_train,
                               'pool_index': self.idx_pool,
                               'PCC_test': pearsonr(self.y_test, self.y_pred_test)[0],
                               'R2_test': r2_score(self.y_test, self.y_pred_test),
                               'PCC_train': pearsonr(self.y_train, self.y_pred_train)[0],
                               'R2_train': r2_score(self.y_train, self.y_pred_train)}
            if self.verbose:
                self.print_history(self.history[i])
        print("Finished Successfully!")
    
    def get_query_points(self):
        if self.query_strategy == 'random':
            pool_idx = np.random.choice(
                np.arange(len(self.idx_pool)), 
                size=10, 
                replace=False)
            return self.idx_pool[pool_idx], pool_idx
        if self.query_strategy == 'variance_reduction':
            pool_idx = np.argmax(self.y_pred_unc_pool)
            return self.idx_pool[pool_idx], pool_idx
        if self.query_strategy == 'cluster_variance_reduction':
            kmeans = KMeans(n_clusters=self.n_cluster).fit(self.X_pool_scaled)
            q_p, pidx = [], []
            for i in range(self.n_cluster):
                temp = self.y_pred_unc_pool.copy()
                temp[kmeans.labels_ != i] = -1
                q_p.append(self.idx_pool[np.argmax(temp)])
                pidx.append(np.argmax(temp))
            return np.array(q_p), pidx
        
    def print_history(self, history: dict):
        print(f"Number of training data: {len(self.idx_train)}")
        print(f"Number of testing data: {len(self.y_test)}")
        print(f"Number of pooling data: {len(self.idx_pool)}")
        print(f"PCC [train]: {history['PCC_train']}")
        print(f"R2 [train]: {history['R2_train']}")
        print(f"PCC [test]: {history['PCC_test']}")
        print(f"R2 [test]: {history['R2_test']}")
    
            
            
            
            
        