from dateutil.parser import parse
from datetime import timedelta

date_start = parse('2013-01-01')


def get_date_index(date):
    delta = date-date_start
    return delta.days


def get_date_index_parse(date):
    date = parse(date)
    return get_date_index(date)


def get_days_in_range(start, end):
    start = parse(start)
    end = parse(end)

    days = []
    while start <= end:
        days.append(get_date_index(start))
        start = start + timedelta(days=1)
    return days
