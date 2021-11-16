import model_selection_breakpoints as msb
import numpy as np
import pandas as pd
loss = np.array([10,9,8,7,6,0], "double")
complexity = np.array([0,1,2,3,4,5], "double")
result = msb.interface(loss, complexity)
n_models = result.pop("n_models")
result_df = pd.DataFrame(result)[:n_models][::-1]
i = result_df["index"]
max_penalty = result_df["penalty"]
print(pd.DataFrame({
    "min_penalty":np.array([0]+list(max_penalty[:-1]), "double"),
    "max_penalty":max_penalty,
    "loss":loss[i],
    "complexity":complexity[i],
}))

