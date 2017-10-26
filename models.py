import pandas as pd


def test_without_train_prediction(test):
    items = pd.read_csv('./data/items_encoded.csv')
    stores = pd.read_csv('./data/stores_encoded.csv')

    test_ext = test.merge(items, on='item_nbr')
    test_ext = test_ext.merge(stores, on='store_nbr')

    test_ext['store_item_tuple'] = test_ext[['item_nbr', 'store_nbr']].itertuples(index=False)

    n_train = pd.read_csv('./data/n_train.csv')
    n_train_tuples = n_train.itertuples(index=False)
    test_known = test_ext[test_ext.store_item_tuple.notin(n_train_tuples)]
    test_unknown = test_ext[test_ext.store_item_tuple.isin(n_train_tuples)]