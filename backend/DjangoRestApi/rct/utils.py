from pathlib import Path

import pandas as pd
import scipy


def batched_decode_preds(
    strong_preds,
    filenames,
    encoder,
    thresholds=[0.5],
    median_filter=7,
    pad_indx=None,
):
    prediction_dfs = {}
    for threshold in thresholds:
        prediction_dfs[threshold] = pd.DataFrame()

    for j in range(strong_preds.shape[0]):  # over batches
        for c_th in thresholds:
            c_preds = strong_preds[j]
            if pad_indx is not None:
                true_len = int(c_preds.shape[-1] * pad_indx[j].item())
                c_preds = c_preds[:true_len]
            pred = c_preds.transpose(0, 1).detach().cpu().numpy()
            pred = pred > c_th
            pred = scipy.ndimage.filters.median_filter(pred, (median_filter, 1))
            pred = encoder.decode_strong(pred)
            pred = pd.DataFrame(pred, columns=["event_label", "onset", "offset"])
            pred["filename"] = Path(filenames[j]).stem + ".wav"
            prediction_dfs[c_th] = prediction_dfs[c_th].append(pred, ignore_index=True)

    return prediction_dfs
