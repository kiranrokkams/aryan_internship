# from featuretools.primitives import Entropy
from .external import categorical_aggregators as aggregators
import inspect

for agg in aggregators:
	if inspect.isclass(agg):
		exec(f'from {agg.__module__} import {agg.__name__}')
	else:
		exec(f'{agg.__class__.name} = agg')

