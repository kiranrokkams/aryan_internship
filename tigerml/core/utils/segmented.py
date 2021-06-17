
def get_segment_filter(data, segment_by, segment):
	filter = None
	# segment_by = self.segment_by if isinstance(self.segment_by, list) else [self.segment_by]
	# if len(self.segment_by) > 1:
	assert len(segment_by) == len(segment)
	for idx, segment_col in enumerate(segment_by):
		if filter is None:
			filter = (data[segment_col] == segment[idx])
		else:
			filter &= (data[segment_col] == segment[idx])
	return filter


def get_segment_from_df(df, seg_cols):
	assert all([df[col].nunique() == 1 for col in seg_cols]), 'Passed df has multiple segments'
	return [df[col].unique().tolist()[0] for col in seg_cols]


def calculate_all_segments(data, segment_by):
	# import itertools
	# self.all_segments = [element for element in itertools.product(*[self.data[seg_col].unique().tolist()
	#                                                                 for seg_col in self.segment_by])]
	return data[segment_by].drop_duplicates().values.tolist()

