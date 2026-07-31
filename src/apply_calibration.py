"""Patches the winning calibrated a_arr/alpha_arr (from calibrate_lambda.py's
output) into main_scoring.py's numagents==500/capacity==4/pickupdelay==90
branch. Run only after calibrate_lambda.py has completed successfully.
"""
import json
import sys

RESULT_FILE = '/tmp/calibrate_lambda_final.json'
TARGET_FILE = 'main_scoring.py'

OLD_BLOCK = """        if(args.numagents == 500 and args.capacity == 4 and args.pickupdelay == 90):
            a_arr = [-0.65, -0.45, -0.55, -0.6, -0.55, -0.55, -0.6, -0.55]
            alpha_arr = [7.0, 8.0, 5.0, -10.0, 0.0, -10.0, 10.0, -10.0]"""

with open(RESULT_FILE) as f:
    result = json.load(f)

new_a = result["a_arr"]
new_alpha = result["alpha_arr"]

new_block = (
    "        if(args.numagents == 500 and args.capacity == 4 and args.pickupdelay == 90):\n"
    "            a_arr = {}\n"
    "            alpha_arr = {}"
).format(new_a, new_alpha)

with open(TARGET_FILE) as f:
    content = f.read()

if OLD_BLOCK not in content:
    print("ERROR: expected block not found in {} -- has it already been patched, "
          "or did the file change? Aborting without modifying anything.".format(TARGET_FILE))
    sys.exit(1)

content = content.replace(OLD_BLOCK, new_block)

with open(TARGET_FILE, 'w') as f:
    f.write(content)

print("Patched {} with calibrated constants:".format(TARGET_FILE))
print("  a_arr = {}".format(new_a))
print("  alpha_arr = {}".format(new_alpha))
