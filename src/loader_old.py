#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PTB-XL data ingestion pipeline.

Created on Thu Apr 23 19:52:06 2026
@author: mehta

The original example from PhysioNet is included as comments so you can compare.
Lines starting with `# SAMPLE:` are the original code.
Lines starting with `# WHY:` explain why we changed it.

Original: https://physionet.org/content/ptb-xl/1.0.3/example_physionet.py
"""
from pathlib import Path   # a modern way to work with file paths
import ast                  # used to turn a string like "{'NORM': 100}" into a real dict
import numpy as np
import pandas as pd
import wfdb                 # the library for reading ECG waveform files
from pipefunc import pipefunc

# ---------------------------------------------------------------------------
# Below is the ENTIRE original example, kept here for reference.
# Our code does the same job, but split into small functions.
# ---------------------------------------------------------------------------
#
# SAMPLE:
# def load_raw_data(df, sampling_rate, path):
#     if sampling_rate == 100:
#         data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
#     else:
#         data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
#     data = np.array([signal for signal, meta in data])
#     return data
#
# path = 'path/to/ptbxl/'
# sampling_rate = 100
#
# Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
# Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
# X = load_raw_data(Y, sampling_rate, path)
#
# agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
# agg_df = agg_df[agg_df.diagnostic == 1]
#
# def aggregate_diagnostic(y_dic):
#     tmp = []
#     for key in y_dic.keys():
#         if key in agg_df.index:
#             tmp.append(agg_df.loc[key].diagnostic_class)
#     return list(set(tmp))
#
# Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)
#
# test_fold = 10
# X_train = X[np.where(Y.strat_fold != test_fold)]
# y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
# X_test = X[np.where(Y.strat_fold == test_fold)]
# y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass
# ---------------------------------------------------------------------------

@pipefunc(output_name="Y_raw")
def load_metadata(path: str | Path) -> pd.DataFrame:
    """Read the big CSV that describes every ECG in the dataset."""

    # WHY: The sample writes `path + 'ptbxl_database.csv'`. If you forget
    # the "/" at the end of path, you end up with "data/ptbxlptbxl_database.csv"
    # and everything breaks. `Path(path) / "file"` always adds the "/" for you,
    # and also works on Windows. Safer and less to remember.
    path = Path(path)
    Y = pd.read_csv(path / "ptbxl_database.csv", index_col="ecg_id")

    # The CSV stores diagnostic codes as text that LOOKS like a dict,
    # e.g. "{'NORM': 100.0}". `ast.literal_eval` turns that text into an
    # actual Python dict so we can use it later.
    #
    # WHY: The sample writes `lambda x: ast.literal_eval(x)`. That's a
    # function that just calls another function. We can pass the inner
    # function directly and skip the wrapper. Same result, cleaner.
    Y.scp_codes = Y.scp_codes.apply(ast.literal_eval)
    return Y


@pipefunc(output_name="X")
def add_diagnostic_labels(Y: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    """Turn the raw diagnostic codes into the 5 big categories we care about.

    PTB-XL has ~70 specific codes (like 'IMI', 'AFIB'). Each one belongs to
    one of 5 broad categories (superclasses): NORM, MI, STTC, CD, HYP.
    We look those up in scp_statements.csv and add them as a new column.
    """
    path = Path(path)

    # scp_statements.csv is a lookup table: each row is a specific code,
    # and one of its columns tells us which big category that code belongs to.
    # We keep only the rows marked `diagnostic == 1` (real diagnoses, not
    # things like "rhythm notes").
    agg_df = pd.read_csv(path / "scp_statements.csv", index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    # For each ECG's dict of codes, look up the big category for each code,
    # and return the unique set of big categories that apply.
    #
    # WHY the short form: The sample uses 5 lines — create an empty list,
    # loop through keys, check if the key exists, append, then remove
    # duplicates with `list(set(...))`. A "set comprehension" does all that
    # in one line. Read it as: "the set of agg_df.loc[k].diagnostic_class
    # for every k in y_dic that is in agg_df.index". Exactly the same work,
    # just more compact.
    def aggregate(y_dic: dict) -> list[str]:
        return list({agg_df.loc[k].diagnostic_class
                     for k in y_dic if k in agg_df.index})

    # WHY `.copy()`: The sample adds the new column directly to Y, which
    # changes the original DataFrame the caller passed in. That can cause
    # confusing bugs later ("why does my unlabeled Y have labels now?").
    # Making a copy means this function doesn't secretly change its input.
    # This is called writing a "pure function" and is generally safer.
    Y = Y.copy()
    Y["diagnostic_superclass"] = Y.scp_codes.apply(aggregate)
    return Y


@pipefunc(output_name="X")
def load_raw_signals(
    Y: pd.DataFrame, path: str | Path, sampling_rate: int = 100
) -> np.ndarray:
    """Read all the ECG waveform files and stack them into one big array.

    The result has shape (number of ECGs, time samples, 12 leads).
    """
    # WHY split this from metadata loading: Loading 21,799 ECG files takes
    # minutes. You usually want to first look at the metadata (Y) — check
    # how many patients, what labels exist, etc. — BEFORE waiting for the
    # signals to load. Separate functions let you do that.
    path = Path(path)

    # Each ECG is stored as two files. The "lr" (low-resolution) version is
    # 100 Hz; the "hr" (high-resolution) version is 500 Hz.
    # We pick which column of filenames to use based on what the user asked for.
    #
    # WHY: The sample has an `if/else` that repeats the `wfdb.rdsamp` loop
    # twice — once for 100 Hz, once for 500 Hz. We pick the column first,
    # then have one loop. Less duplicated code.
    files = Y.filename_lr if sampling_rate == 100 else Y.filename_hr

    # `wfdb.rdsamp` returns a tuple: (signal_array, metadata_dict).
    # We only want the signal, so we grab index [0].
    # Then we stack all signals into one big numpy array.
    return np.array([wfdb.rdsamp(str(path / f))[0] for f in files])



@pipefunc(output_name="splits")
def get_splits(
    X: np.ndarray, Y: pd.DataFrame, val_fold: int = 9, test_fold: int = 10
):
    """Split the data into training, validation, and test sets.

    PTB-XL comes pre-split into 10 folds (the `strat_fold` column, values 1-10).
    The PTB-XL paper recommends: folds 1-8 for training, fold 9 for validation,
    fold 10 for testing. Using this standard split means your numbers can be
    compared to published results.
    """
    # WHY three splits instead of two: The sample only does train and test.
    # But during development you tune things (learning rate, model size, etc.)
    # by checking performance on some held-out data. If you check on the TEST
    # set, you're slowly "cheating" — you end up picking whatever works best
    # ON THE TEST SET, which isn't a fair final score anymore.
    # The fix: hold out a VALIDATION set for tuning, and only touch the test
    # set once at the very end.

    # A "boolean mask" is an array of True/False. `Y[mask]` keeps rows
    # where mask is True.
    #
    # WHY not `np.where` like the sample: `np.where(mask)` converts True/False
    # into a list of row numbers, which then get used to index X. That's two
    # steps where one works fine. `X[mask.values]` is the direct version.
    train_mask = ~Y.strat_fold.isin([val_fold, test_fold])  # NOT in 9 or 10
    val_mask = Y.strat_fold == val_fold
    test_mask = Y.strat_fold == test_fold

    return (
        X[train_mask.values], Y[train_mask].diagnostic_superclass,
        X[val_mask.values],   Y[val_mask].diagnostic_superclass,
        X[test_mask.values],  Y[test_mask].diagnostic_superclass,
    )