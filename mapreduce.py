import pandas as pd


class MapReduceAbstract(object):
    def __init__(self):
        pass

    def map(self, df_batch):
        raise NotImplementedError("Should have implemented this")

    def reduce(self, x, result):
        raise NotImplementedError("Should have implemented this")

    def filter(self, df_batch):
        raise NotImplementedError("Should have implemented this")


class FilteringMapReduce(MapReduceAbstract):
    def __init__(self, filter_function):
        self.filter_function = filter_function

    def map(self, df_batch):
        return df_batch

    def reduce(self, df_batch, df_batch2):
        return pd.concat([df_batch, df_batch2])

    def filter(self, df_batch):
        return self.filter_function(df_batch)


class ItemFamilyAggregationMapReduce(MapReduceAbstract):
    def __init__(self, items):
        self.items = items

    def filter(self, df_batch):
        return df_batch

    def map(self, df_batch):
        df_items = df_batch.merge(self.items[['item_nbr', 'family']], on='item_nbr')
        df_aggregated = df_items.groupby(['family', 'date'], as_index=False).agg({'unit_sales':'sum'})

        # filling empty values
        u_dates = df_batch.date.unique()
        u_families = self.items.family.unique()
        df_aggregated.set_index(["date", "family"], inplace=True)
        df_aggregated = df_aggregated.reindex(
            pd.MultiIndex.from_product(
                (u_dates, u_families),
                names=["date", "family"]
            )
        )
        df_aggregated.loc[:, "unit_sales"].fillna(0, inplace=True)
        return df_aggregated

    def reduce(self, df_aggregated, df_aggregated_prev):
        df_reduce = df_aggregated_prev.merge(df_aggregated, right_index=True, left_index=True, how='outer')
        df_reduce.fillna(0.0, inplace=True)
        df_reduce['unit_sales'] = df_reduce['unit_sales_x'] + df_reduce['unit_sales_y']
        del df_reduce['unit_sales_x']
        del df_reduce['unit_sales_y']
        return df_reduce


def print_if_verbose(s, verbose):
    if verbose:
        print(s)


def map_reduce_df(csv_path, map_reduce_object, types=None, position=0, batch_size=10000000, cols=None, verbose=False):
    result = None
    if position != 0 and cols is None:
        raise ValueError("You should either start from position 0 or specify columns.")

    while True:
        print_if_verbose('Reading batch from position {}, batch size {}...'.format(position, batch_size), verbose)
        try:
            if position == 0:
                df = pd.read_csv(csv_path, dtype=types, nrows=batch_size, skiprows=position)
                cols = df.columns
            else:
                df = pd.read_csv(csv_path, dtype=types, nrows=batch_size, skiprows=position, names=cols, header=None)
        except Exception as e:
            print_if_verbose('End of dataset is found. Exception {}.'.format(e), verbose)
            break

        if len(df) == 0:
            print_if_verbose('End of dataset is found. Dataset is empty.', verbose)
            break

        print_if_verbose('Filtering {}...'.format(len(df)), verbose)
        df_filtered = map_reduce_object.filter(df)
        print_if_verbose('Filtered {}, mapping...'.format(len(df_filtered)), verbose)

        try:
            mapped = map_reduce_object.map(df_filtered)
        except:
            print('Position: {}'.format(position))
            raise

        print_if_verbose('Mapped, reducing...', verbose)

        if result is None:
            result = mapped
        else:
            try:
                result = map_reduce_object.reduce(mapped, result)
            except:
                print('Position: {}'.format(position))
                raise

        print_if_verbose('Batch done.', verbose)

        if len(df) + 1 < batch_size:
            print_if_verbose('End of dataset is found.', verbose)
            break

        position += batch_size

        print_if_verbose('='*80, verbose)

    return result

