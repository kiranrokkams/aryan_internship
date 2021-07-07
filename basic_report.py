import pickle
with open("external_sources/RegressionReport_pickle_dump.pkl", "rb") as pklf:
	x = pickle.load(pklf)
	x.get_report(format=".html")
print("success")
