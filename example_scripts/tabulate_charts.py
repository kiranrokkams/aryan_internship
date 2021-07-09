"""
This script tabulates the classes of charts and their associated columns in the
data into a JSON format, showing the script, its charts, and each chart's data's
column labels. This data is useful for exploring the variety of chart types one
could encounter when creating standard reports.

```
print("START_CLASS_NAME")
print(type(content))
print("END_CLASS_NAME")
print("START_DIMENSIONS")
print(content.dimensions())
print("END_DIMENSIONS")
```

^must be placed in the logic that processes charts in the create_components
function (the if statement mentioning holoviews, plotly, etc.)
"""

import subprocess
import csv
import os
import json

os.chdir("..")
curr_dir = os.path.dirname(os.path.abspath(__file__))


with open("example_scripts/RESULTS.txt", 'w') as resultfile:
    for scriptname in os.listdir(curr_dir + "/example_scripts"):
        if not scriptname.startswith("model_eval_"):
            continue
        # move each script out of the example_scripts folder to access tigerml
        old_dir = curr_dir + "/example_scripts/" + scriptname
        new_dir = curr_dir + "/" + scriptname

        os.rename(old_dir, new_dir)

        # execute it and store results
        result = subprocess.run(["python", scriptname], stdout=subprocess.PIPE)
        print(scriptname)
        # add markers to show script names among various outputs to stdout
        resultfile.write("START_SCRIPT_NAME" + scriptname + "END_SCRIPT_NAME")
        resultfile.write("\n")
        resultfile.write(result.stdout.decode('utf-8'))
        resultfile.write("\n\n")

        # move it back
        os.rename(new_dir, old_dir)

data = {}
with open("example_scripts/RESULTS.txt", 'r') as resultfile:
    text = resultfile.read()

    # while we can still process more scripts
    while text.find("START_SCRIPT_NAME") != -1:
        sn_start = text.find("START_SCRIPT_NAME") + len("START_SCRIPT_NAME")
        sn_end = text.find("END_SCRIPT_NAME")

        scriptname = text[sn_start:sn_end].strip()
        # show that we have processed this script
        print(scriptname)

        # take a substring to move onto the next section
        text = text[sn_end + len("END_SCRIPT_NAME"):]
        data[scriptname] = []

        # while we have more chart classes in this script
        while (text.find("START_CLASS_NAME") < text.find("START_SCRIPT_NAME")) \
        or (text.find("START_CLASS_NAME") >= 0 and text.find("START_SCRIPT_NAME") == -1):
            cn_start = text.find("START_CLASS_NAME") + len("START_CLASS_NAME")
            cn_end = text.find("END_CLASS_NAME")
            cn = text[cn_start:cn_end].strip()

            body_start = text.find("START_DIMENSIONS") + len("START_DIMENSIONS")
            body_end = text.find("END_DIMENSIONS")
            # take a substring to move onto the next section
            body = text[body_start:body_end].strip()

            data[scriptname].append({"classname": cn, "data": body})

            # take a substring to move onto the next section
            text = text[body_end + len("END_DIMENSIONS"):]

# dump into the resulting json file
with open("example_scripts/chartdata.json", 'w') as jsonresults:
    json.dump(data, jsonresults)
