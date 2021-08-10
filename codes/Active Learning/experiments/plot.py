import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import numpy as np
from pathlib import Path


plt.style.use('seaborn-white')
# sns.set()
font = {'family' : 'sans-serif',
        'size'   : 16}
matplotlib.rc('font', **font)
sns.set_palette('mako')

def coverage(y, yL, yH):
    return (100 / y.shape[0] * ((y>yL)&(y<yH)).sum())

def plot_pcc_history(history, plot_path):
    n_data = [len(history[k]['train_index']) for k in history]
    pcc = [history[k]['PCC_test'] for k in history]
    with plt.style.context('seaborn-white'):
        plt.figure(figsize=(8, 8))
        plt.plot(n_data, pcc, 
                    c='r', label='PCC')
        plt.title('PCC over the number of data points')
        plt.legend()
        plt.xlabel("Number of data points")
        plt.ylabel("PCC")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close() 
        
def plot_r2_history(history, plot_path):
    n_data = [len(history[k]['train_index']) for k in history]
    r2 = [history[k]['R2_test'] for k in history]
    with plt.style.context('seaborn-white'):
        plt.figure(figsize=(8, 8))
        plt.plot(n_data, r2, 
                    c='r', label='R2')
        plt.title('R2 over the number of data points')
        plt.legend()
        plt.xlabel("Number of data points")
        plt.ylabel("R2")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        
if __name__ == '__main__':
    
    random = Path(__file__).resolve().parent / 'random'
    for f in random.glob("*.npy"):
        history = np.load(f, allow_pickle=True).item()
        i = str(f).split('.')[0].split('_')[-1]
        plot_pcc_history(history, f.parent / f'pcc_{i}.png')
        plot_r2_history(history, f.parent / f'r2_{i}.png')
        
    var_red = Path(__file__).resolve().parent / 'variance_reduction'
    for f in var_red.glob("*.npy"):
        history = np.load(f, allow_pickle=True).item()
        i = str(f).split('.')[0].split('_')[-1]
        plot_pcc_history(history, f.parent / f'pcc_{i}.png')
        plot_r2_history(history, f.parent / f'r2_{i}.png')
        
    clus_var_red = Path(__file__).resolve().parent / 'cluster_variance_reduction'
    for f in clus_var_red.glob("*.npy"):
        history = np.load(f, allow_pickle=True).item()
        i = str(f).split('.')[0].split('_')[-1]
        plot_pcc_history(history, f.parent / f'pcc_{i}.png')
        plot_r2_history(history, f.parent / f'r2_{i}.png')