import pandas as pd


class DataSet:

    def __init__(self, data, fetch=None, groupby=None, filter=None):
        if not isinstance(data, pd.DataFrame):
            raise Exception('data can only be a pandas DataFrame')
        self.data = data
        self._fetch = fetch
        self._groupby = groupby
        self._filter = filter

    def get_fetch(self):
        return self._fetch

    def get_groupby(self):
        return self._groupby

    def get_filter(self):
        return self._filter

    def _validate_fetch(self, fetch_cols):
        return fetch_cols

    def _validate_groupby(self, groupby_cols):
        return groupby_cols

    def _validate_filter(self, filter_rules):
        return filter_rules

    def fetch(self, fetch):
        self._fetch = self._validate_fetch(fetch)
        return self

    def groupby(self, groupby):
        self._fetch = self._validate_groupby(groupby)
        return self

    def filter(self, filter):
        self._fetch = self._validate_filter(filter)
        return self

    def run(self):
        return self.data[self.filter].groupby(by=self.groupby)[self.fetch]

