from dateutil.parser import parse
from datetime import timedelta
import pandas as pd

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
    # dates_weekdays = pd.get_dummies(dates_df[['weekday']], sparse=True).astype(bool)
    # del dates_df['weekday']
    # dates_df = pd.concat([dates_df, dates_weekdays], axis=1)
    return df.merge(dates_df, on='date'), ['weekday', 'weekend', 'salary']


def extract_date_features(index):
    date = get_index_date(index)
    #weekdays = ['Mon', 'Tue', 'Wen', 'Thr', 'Fri', 'Sat', 'Sun']
    weekday = date.weekday()
    eofweek = date.weekday() == 5 or date.weekday() == 6
    day = date.day
    salary_day = (14 < day < 17) or day > 29 or day < 2
    return (index, weekday, eofweek, salary_day)