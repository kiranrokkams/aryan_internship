from .base import DataProcessor


class TSProcessor(DataProcessor):

	def __init__(self, data, ts_identifiers=None, *args, **kwargs):
		super().__init__(data, segment_by=ts_identifiers, *args, **kwargs)

