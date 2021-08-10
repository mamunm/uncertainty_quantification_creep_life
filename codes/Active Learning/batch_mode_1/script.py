from active_learning.active_learning import ActiveLearning
from pathlib import Path
import numpy as np 

data = np.load(Path(__file__).resolve().parents[3] / f"data/chrome_data.npy", 
               allow_pickle=True)[()]


al = ActiveLearning(data, 
    "/Users/osman/deep_learning/uncertainty_quantification_creep_life/codes/Active Learning/batch_mode_1/plots")
al.run(n_iter=20, batch_size=20)
np.save(Path(__file__).resolve().parent / "active_learning_1.npy", 
        al.history)
