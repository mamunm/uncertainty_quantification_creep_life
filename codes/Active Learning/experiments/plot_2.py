import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import numpy as np
from pathlib import Path
import pandas as pd

plt.style.use('seaborn-white')
# sns.set()
font = {'family' : 'sans-serif',
        'size'   : 16}
matplotlib.rc('font', **font)
sns.set_palette('mako')
 
if __name__ == '__main__':
    
    random = Path(__file__).resolve().parent / 'random/active_learning_1.npy'
    random_history = np.load(random, allow_pickle=True).item()    
    n_data = [len(random_history[k]['train_index']) for k in random_history]
    r2 = [random_history[k]['R2_test'] for k in random_history]
    df = pd.DataFrame({'n_data': n_data, 'R2_random': r2})
    clus_var_red = Path(__file__).resolve().parent / 'cluster_variance_reduction/active_learning_2.npy'
    clus_var_red_history = np.load(clus_var_red, allow_pickle=True).item()
    n_data = [len(clus_var_red_history[k]['train_index']) for k in clus_var_red_history]
    r2 = [clus_var_red_history[k]['R2_test'] for k in clus_var_red_history]
    df['R2_al'] = r2
    df = df[df.n_data < 255]
    
    df = pd.melt(df, id_vars=['n_data'], value_vars=['R2_random', 'R2_al'], var_name='method', value_name='R2')
    
    sns.lineplot(data=df, x='n_data', y='R2', hue='method', 
                 style='method', legend=False, palette='mako',
                 linewidth=2.5, markers=['o', 's'], dashes=False)
    plt.xlabel('Number of training data')
    plt.ylabel('$R^2$')
    plt.legend(['Random', 'Active Learning'], loc='lower right')
    plt.show()
    
    
    
    
