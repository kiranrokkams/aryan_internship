import json

chart_data = None
with open("chartdata.json") as chartdatafile:
    chart_data = json.load(chartdatafile)
    chart_types = {}

    for script in chart_data:
        for chart_instance in chart_data[script]:
            chart_types[chart_instance["classname"]] = chart_instance["data"]

# all chart classes that show up here can have their data accessed through
# the .dimensions method
for chart_type, dimensions in chart_types.items():
    print(chart_type)
    print(dimensions)
    print()
