from dateutil.parser import parse
from datetime import timedelta
import pandas as pd
from mapreduce import *
from model import types

date_start = parse('2013-01-01')


def get_date_index(date):
    delta = date-date_start
    return delta.days


def get_index_date(index):
    delta = timedelta(days=int(index))
    date = date_start+delta
    return date


def get_date_index_parse(date):
    date = parse(date)
    return get_date_index(date)


def get_days_in_range(start, end):
    start = get_date_index_parse(start)
    end = get_date_index_parse(end)

    return list(range(start, end+1))


def add_date_features(df):
    dates = df.date.unique()
    features = list(map(extract_date_features, dates))
    dates_df = pd.DataFrame(features, columns=['date', 'weekday', 'weekend', 'salary'])
    return df.merge(dates_df, on='date'), ['weekday', 'weekend', 'salary']


def add_date_features_one_hot(df):
    dates = df.date.unique()
    features = list(map(extract_date_features, dates))
    dates_df = pd.DataFrame(features, columns=['date', 'weekday', 'weekend', 'salary'])
    dates_weekdays = pd.get_dummies(dates_df[['weekday']], sparse=True).astype(bool)
    del dates_df['weekday']
    dates_df = pd.concat([dates_df, dates_weekdays], axis=1)
    return df.merge(dates_df, on='date'), ['weekend', 'salary'] + dates_weekdays.columns


def extract_date_features(index):
    date = get_index_date(index)
    weekday = date.weekday()
    eofweek = date.weekday() == 5 or date.weekday() == 6
    day = date.day
    salary_day = (14 < day < 17) or day > 29 or day < 2
    return index, weekday, eofweek, salary_day


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