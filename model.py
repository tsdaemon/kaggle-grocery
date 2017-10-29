from date import *
from mapreduce import *
import numpy as np
import itertools
items_cols = ['class', 'perishable', 'family']
stores_cols = ['city', 'type', 'cluster', 'state']
types = {'id': 'int32',
             'date': 'int16',
             'item_nbr': 'int32',
             'store_nbr': 'int16',
             'unit_sales': 'float32',
             'onpromotion': bool}



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


def extract_by_date(df, train_start, train_end):
    train_range = get_days_in_range(train_start, train_end)
    train = df[df.date.isin(train_range)]
    return train


def load_data_in_date_range(csv, start, end, position):
    # loading training data
    days = get_days_in_range(start, end)
    cols = ['id', 'date', 'store_nbr', 'item_nbr', 'unit_sales', 'onpromotion']
    mapreduce = FilteringMapReduce(lambda df: df[df.date.isin(days)])
    return map_reduce_df(csv, mapreduce, types=types, position=position, cols=cols, verbose=True)


def convert_unit_sales(df):
    df.ix[df.unit_sales <0, 'unit_sales'] = 0
    df['unit_sales'] = np.log1p(df['unit_sales'])
    return df


def fill_mean_encoding(df, df_prev, categorical_features):
    colnames = []
    # for combination in itertools.combinations(categorical_features, 2):
    #     colname = 'mean_unit_sales_by_{}'.format('+'.join(combination))
    #     mean_agg = df_prev.groupby(combination, as_index=False).agg({'unit_sales': 'mean'})
    #     mean_agg.rename(columns={'unit_sales': colname}, inplace=True)
    #     df = df.merge(mean_agg, on=combination, how='left')
    #     colnames.append(colname)

    for combination in itertools.combinations(categorical_features, 1):
        colname = 'mean_unit_sales_by_{}'.format('+'.join(combination))
        mean_agg = df_prev.groupby(combination, as_index=False).agg({'unit_sales': 'mean'})
        mean_agg.rename(columns={'unit_sales': colname}, inplace=True)
        df = df.merge(mean_agg, on=combination, how='left')
        colnames.append(colname)

    combination = ['store_nbr','item_nbr','onpromotion']
    colname = 'mean_unit_sales_by_{}'.format('+'.join(combination))
    mean_agg = df_prev.groupby(combination, as_index=False).agg({'unit_sales': 'mean'})
    mean_agg.rename(columns={'unit_sales': colname}, inplace=True)
    df = df.merge(mean_agg, on=combination, how='left')
    colnames.append(colname)

    return df, colnames


def fill_lagged(df, df_prev, start_lagged, end_lagged):
    df_prev.date += start_lagged
    colnames = []
    while start_lagged <= end_lagged:
        colname = 'unit_sales(t-{})'.format(start_lagged)
        df_prev[colname] = df_prev['unit_sales']
        df = df.merge(df_prev[['item_nbr', 'store_nbr', 'date', colname]], on=['item_nbr', 'store_nbr', 'date'], how='left')
        df_prev.date += 1
        start_lagged += 1
        colnames.append(colname)

    return df


def get_two_week_ranges(num, end_index):
    ranges = []
    for i in range(num):
        week2end = end_index
        week2start = end_index-13
        week1end = end_index-14
        week1start = end_index-35
        ranges.append((list(range(week2start, week2end+1)), list(range(week1start, week1end+1))))
        end_index -= 14
    return ranges


def add_lagged_and_mean_encoding(df, lagged_start, lagged_end, categorical_features):
    # generating two weeks ranges
    ranges = get_two_week_ranges(8, get_date_index_parse('2017-08-15'))

    result = []
    colnames = ['unit_sales_mean']
    # mean encoding
    for week2, week1 in reversed(ranges):
        week2df = df[df.date.isin(week2)]
        week1df = df[df.date.isin(week1)]
        week2df, colnames = fill_mean_encoding(week2df, week1df, categorical_features)
        result.append(week2df)
        del week1df
    df_result = pd.concat(result)

    # lagged features
    df_result, cols = fill_lagged(df_result, df, lagged_start, lagged_end)
    colnames += cols

    df_result.fillna(0.0, inplace=True)
    return df_result, colnames