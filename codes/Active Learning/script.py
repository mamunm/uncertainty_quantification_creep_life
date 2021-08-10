from src.active_learning import ActiveLearning
from pathlib import Path
import numpy as np 

data = np.load(Path(__file__).resolve().parents[2] / f"data/chrome_data.npy", 
               allow_pickle=True)[()]


# al = ActiveLearning(data = data, 
#                     f_path = "experiments/random",
#                     query_strategy= "random")
# al.run(n_iter=30)
# np.save(Path(__file__).resolve().parent / "experiments/random/active_learning_3.npy", 
#         al.history)

al = ActiveLearning(data = data, 
                    f_path = "experiments/random",
                    query_strategy= "variance_reduction")
al.run(n_iter=300)
np.save(Path(__file__).resolve().parent / "experiments/variance_reduction/active_learning_3.npy", 
        al.history)


# al = ActiveLearning(data = data, 
#                     f_path = "experiments/random",
#                     query_strategy= "cluster_variance_reduction")
# al.run(n_iter=30)
# np.save(Path(__file__).resolve().parent / "experiments/cluster_variance_reduction/active_learning_3.npy", 
#         al.history)

