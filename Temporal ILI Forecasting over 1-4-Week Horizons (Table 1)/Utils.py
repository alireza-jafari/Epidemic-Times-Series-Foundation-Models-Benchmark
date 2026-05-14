import pandas as pd
import numpy as np
import argparse

def splitter(data, time_series_id, n):
    train_frames = []
    test_frames = []
    for _, group in data.groupby(time_series_id):
        train, test = group[:-n], group[-n:]
        train_frames.append(train)
        test_frames.append(test)
    train_df = pd.concat(train_frames)
    test_df = pd.concat(test_frames)
    train_df.reset_index(inplace=True, drop=True)
    test_df.reset_index(inplace=True, drop=True)
    return train_df, test_df

def drop_last_n_samples(data, time_series_id, n):
    train_frames = []
    for _, group in data.groupby(time_series_id):
        train = group[:-n]
        train_frames.append(train)
    train_df = pd.concat(train_frames)
    train_df.reset_index(inplace=True, drop=True)
    return train_df