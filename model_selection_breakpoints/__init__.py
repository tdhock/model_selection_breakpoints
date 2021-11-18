import pandas as pd
import model_selection_breakpoints_c
def min_label_error(input_df):
    """Compute penalties which will help us find minimum label error.

    input_df should be a pandas.DataFrame with columns peaks,
    total.loss, penalty, fp, possible.fp, fn, possible.fn,

    """
    max_penalty_rows_list = []
    loss = None
    for peaks_value, peaks_df in input_df.groupby("peaks"):
        #pre-processing: for each unique value of peaks, keep only
        #the row with max penalty. Then discard models for which the total
        #loss does not decrease.
        max_pen_row = peaks_df.iloc[peaks_df.penalty.argmax()]
        loss = max_pen_row["total.loss"]
        if loss is None or loss < prev_loss:
            max_penalty_rows_list.append(max_pen_row)
            prev_loss = loss
    processed_df = pd.DataFrame(max_penalty_rows_list)
    A = {k:processed_df[k].to_numpy("double") for k in ("total.loss", "peaks")}
    result = model_selection_breakpoints_c.interface(A["total.loss"], A["peaks"])
    n_models = result.pop("n_models")
    result_df = pd.DataFrame(result)[:n_models]
    i = result_df["index"]
    max_penalty = result_df["penalty"]
    out_df = processed_df.iloc[i]
    out_df["max_penalty"] = np.array(max_penalty, "double")
    out_df["min_penalty"] = np.array(list(max_penalty[1:])+[0], "double")
    for error_type in "fp", "fn":
        possible = out_df["possible."+error_type]
        out_df["w."+error_type] = np.where(
            possible==0, 0, out_df[error_type]/possible)
    for prefix in "", "w.":
        out_df[prefix+"errors"] = out_df[prefix+"fp"] + out_df[prefix+"fn"]
    for m in "min", "max":
        out_df[m+"_log_penalty"] = out_df[m+"_penalty"].transform("log")
    already_computed = out_df["max_penalty"].isin(input_df["penalty"])
    # diff=1 means max_penalty will not result in new model.
    min_boring = out_df["peaks"].diff()==1
    out_df["done"] = min_boring | already_computed
    is_min_errors = out_df["errors"] == out_df["errors"].min()
    min_w_err = out_df["w.errors"][is_min_errors].min()
    is_best = is_min_errors & (out_df["w.errors"]==min_w_err)
    diff_best = np.array(list(is_best[:1])+list(is_best.astype("int").diff()[1:]))
    out_df["best_i"] = (diff_best==1).cumsum()
    out_df["is_other"] = (0 < out_df.fp.diff()) & (out_df.fn.diff() < 0)
    #out_df[["peaks","min_penalty","max_penalty","fp","fn","is_other"]]
    group_info_list = []
    for best_i, group_df in out_df.loc[is_best].groupby("best_i"):
        group_rows = group_df.shape[0]
        if group_rows==1 or 0==group_df.iloc[0].errors:
            mid_penalty = pd.NA
        else:
            for m in "min","max":
                group_df[m] = group_df[m+"_log_penalty"].diff()
            group_df["dist"] = group_df["min"]+group_df["max"]
            some = group_df.loc[(group_df.is_other==False) & (group_df.done==False)]
            mid_penalty = some.iloc[some.dist.argmin()]["max_penalty"] if some.shape[0] else pd.NA
        group_info = {
            "mid_penalty":mid_penalty,
            "peaks_diff":group_df.peaks.iloc[group_rows-1]-group_df.peaks.iloc[0],
        }
        # add min/max(log)penalty
        for m,i in ("min",-1), ("max",0):
            for tr in "_log", "":
                col = m+tr+"_penalty"
                group_info[col] = group_df[col].iloc[i]
        group_info_list.append(group_info)
    group_info_df = pd.DataFrame(group_info_list)
    largest_interval = group_info_df.iloc[group_info_df.peaks_diff.argmax()]
    target_dict = {m:largest_interval[m+"_log_penalty"] for m in ("min","max")}
    stopping_candidates = set([largest_interval[m+"_penalty"] for m in ("min","max")])
    other_candidates = out_df["max_penalty"].loc[ out_df["is_other"] ]
    stopping_candidates.update(other_candidates)
    not_done = out_df["max_penalty"].loc[ out_df["done"]==False ]
    stopping_candidates.intersection_update(not_done)
    if len(stopping_candidates):
        for m in "min","mid","max":
            stopping_candidates.update(group_info_df[m+"_penalty"])
        stopping_candidates.intersection_update(not_done)
    return {
        "new_penalties":stopping_candidates,
        "target_dict":target_dict,
    }
