from date import *
from mapreduce import *
import numpy as np
items_cols = ['class', 'perishable', 'family']
stores_cols = ['city', 'type', 'cluster', 'state']


def extend_dataset(df, items, stores):
    df_ext, date_cols = add_date_features(df)
    df_ext = df_ext.merge(items, on='item_nbr')
    df_ext = df_ext.merge(stores, on='store_nbr')
    return df_ext, date_cols + items_cols + stores_cols


def fill_empty_sales(df):
    u_dates = df.date.unique()
    u_stores = df.store_nbr.unique()
    u_items = df.item_nbr.unique()
    df.set_index(["date", "store_nbr", "item_nbr"], inplace=True)
    df = df.reindex(
        pd.MultiIndex.from_product(
            (u_dates, u_stores, u_items),
            names=["date", "store_nbr", "item_nbr"]
        )
    )
    df.loc[:, "unit_sales"].fillna(0, inplace=True)
    df.loc[:, "onpromotion"].fillna(False, inplace=True)
    return df.reset_index()


def extract_train_test(df, train_start, train_end, test_start, test_end):
    train_range = get_days_in_range(train_start, train_end)
    test_range = get_days_in_range(test_start, test_end)
    train = df[df.date.isin(train_range)]
    test = df[df.date.isin(test_range)]
    return train, test


def load_data_in_date_range(csv, start, end, position):
    # loading training data
    types = {'id': 'int32',
             'date': 'int16',
             'item_nbr': 'int32',
             'store_nbr': 'int16',
             'unit_sales': 'float32',
             'onpromotion': bool}
    days = get_days_in_range(start, end)
    cols = ['id', 'date', 'store_nbr', 'item_nbr', 'unit_sales', 'onpromotion']
    mapreduce = FilteringMapReduce(lambda df: df[df.date.isin(days)])
    return map_reduce_df(csv, mapreduce, types=types, position=position, cols=cols, verbose=True)


def convert_unit_sales(df):
    df.ix[df.unit_sales <0, 'unit_sales'] = 0
    df['unit_sales'] = np.log1p(df['unit_sales'])
    return df