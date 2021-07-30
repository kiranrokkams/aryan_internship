import os
import subprocess

"""
This script runs each example script and records its output
"""

os.chdir("..")
curr_dir = os.path.dirname(os.path.abspath(__file__))

with open("exploratory_scripts/xlsx_results.txt", "w") as resultfile:
    for scriptname in os.listdir(curr_dir + "/example_scripts"):
        if not scriptname.startswith("model_eval_"):
            continue
        # move each script out of the example_scripts folder to access tigerml
        old_dir = curr_dir + "/example_scripts/" + scriptname
        new_dir = curr_dir + "/" + scriptname

        os.rename(old_dir, new_dir)

        # execute it and store results
        result = subprocess.run(
            ["python", scriptname], stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )

        print("generated output for:", scriptname)
        # add markers to show script names among various outputs to stdout
        resultfile.write("script: " + scriptname + "\n")
        resultfile.write(result.stdout.decode("utf-8"))
        resultfile.write("\n\n")

        # move it back
        os.rename(new_dir, old_dir)
