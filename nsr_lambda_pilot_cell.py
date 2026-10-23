# Paste this entire cell at the bottom of your initialized FULL P4 notebook.
# Stop any other running cell first. Keep Config, run ID and Drive paths unchanged.
RUN_FULL_TRAINING = False
RUN_FULL_RETENTION = False
RUN_REFERENCE_ABLATION = False
RUN_MECHANISM_DIAGNOSTICS = False
GENERATE_PAPER_RESULTS = False

import urllib.request

_pilot_url = "https://raw.githubusercontent.com/Wellknownbuffoon/ISCAS_CONF/verified/pcm-pipeline-audit/nsr_lambda_training_pilot.py"
_pilot_code = urllib.request.urlopen(_pilot_url, timeout=60).read().decode("utf-8")
# Separate namespace preserves the original notebook's scientific definitions.
_pilot_namespace = {"__name__": "nsr_lambda_pilot", "__source__": _pilot_code}
exec(compile(_pilot_code, "nsr_lambda_training_pilot.py", "exec"), _pilot_namespace)
NSR_PILOT_DIR = _pilot_namespace["run_pilot"](dict(globals()))
