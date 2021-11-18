import model_selection_breakpoints_c as msbc
from model_selection_breakpoints import min_label_error
import numpy as np
import pandas as pd
loss = np.array([10,9,8,7,6,0], "double")
complexity = np.array([0,1,2,3,4,5], "double")
result = msbc.interface(loss, complexity)
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

## PeakSegDisk example data.
model_summary = pd.read_csv("Mono27ac_model_summary.csv")
for iteration in range(1,10):
    iteration_df = model_summary.loc[
        model_summary["iteration"] <= iteration
    ]
    result_dict = min_label_error(iteration_df)
    print(result_dict)
    print("R computed these penalties in next iteration:")
    print(set(model_summary.loc[model_summary.iteration == iteration+1].penalty))
    print("\n")
