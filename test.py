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

## PeakSegDisk example data.
model_summary = pd.read_csv("Mono27ac_model_summary.csv")
iteration = 3
iteration_df = model_summary.loc[
    model_summary["iteration"] <= iteration
].sort_values(by="penalty", ascending=False)
A = {k:iteration_df[k].to_numpy("double") for k in ("total.loss", "peaks")}
result = msb.interface(A["total.loss"], A["peaks"])
n_models = result.pop("n_models")
result_df = pd.DataFrame(result)[:n_models]
i = result_df["index"]
max_penalty = result_df["penalty"]
out_df = iteration_df.iloc[i]
out_df["max_penalty"] = np.array(max_penalty, "double")
out_df["min_penalty"] = np.array(list(max_penalty[1:])+[0], "double")
