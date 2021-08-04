import pandas as pd 
import numpy as np 
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

plt.style.use('seaborn-white')

font = {'family' : 'sans-serif',
        'size'   : 16}

matplotlib.rc('font', **font)

# sns.set()
# colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

data = np.load(Path(__file__).resolve().parents[1] / f"data/chrome_data.npy", 
               allow_pickle=True)[()]

df = pd.DataFrame({'y': data['y'],
                   'log_y': np.log(data['y'])})

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
ax = ax.ravel()
sns.histplot(df['y'], ax=ax[0])
sns.histplot(df['log_y'], ax=ax[1])
ax[0].ticklabel_format(style='sci', scilimits=(-3,4), axis='x')
ax[0].set_xlabel("Rupture Life [hrs]")
ax[1].set_xlabel("log[Rupture Life [hrs]]")
ax[0].set_title("Original Distribution")
ax[1].set_title("Log Transformed Distribution")
ax[0].grid(True, which='major', linestyle='-', linewidth='0.5', color='gray')
ax[0].grid(True, which='minor', linestyle='-', linewidth='0.25', color='gray')
ax[1].grid(True, which='major', linestyle='-', linewidth='0.5', color='gray')
ax[1].grid(True, which='minor', linestyle='-', linewidth='0.25', color='gray')
plt.tight_layout()
plt.savefig(Path(__file__).resolve().parents[0] / f"chrome_data_dist.png")
plt.show()