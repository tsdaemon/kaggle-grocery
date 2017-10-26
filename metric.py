import numpy as np
import pandas as pd
import math

items = None


def NWRMSLE(y_true, y_pred, weights):
    y_true = np.array(y_true).clip(0, np.max(y_true))
    y_pred = np.array(y_pred).clip(0, np.max(y_pred))
    y_true_log = np.log1p(y_true)
    y_pred_log = np.log1p(y_pred)
    return NWRMSLE_log(y_true_log, y_pred_log, weights)


def NWRMSLE_log(y_true_log, y_pred_log, weights):
    error = math.sqrt(np.sum(weights*np.square(y_true_log-y_pred_log))/np.sum(weights))
    return error


def get_weights(item_nbrs):
    global items
    if items is None:
        items = pd.read_csv('./data/items.csv')
    weights_df = pd.DataFrame({'item_nbr': item_nbrs}).merge(items, on='item_nbr')
    return weights_df['weight']

