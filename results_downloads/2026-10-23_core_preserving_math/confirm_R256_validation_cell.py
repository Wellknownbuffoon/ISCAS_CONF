# Add to an initialized FULL P4 runtime. Stop any active cell first.
# Keep Config, NSR lambda, run ID and paths unchanged.
# Fold 0 has already been evaluated; this confirms the remaining folds.
RUN_FULL_TRAINING = RUN_FULL_RETENTION = False
RUN_REFERENCE_ABLATION = RUN_MECHANISM_DIAGNOSTICS = False
GENERATE_PAPER_RESULTS = False

import urllib.request

_url = "https://raw.githubusercontent.com/Wellknownbuffoon/ISCAS_CONF/3b830c60a9a2a45f797c611feb84530ca522ea2a/validation_reference_budget_pilot.py"
_code = urllib.request.urlopen(_url, timeout=60).read().decode("utf-8")
_namespace = {"__name__": "core_preserving_reference_validation", "__source__": _code}
exec(compile(_code, "validation_reference_budget_pilot.py", "exec"), _namespace)
CORE_PRESERVING_PILOT_DIR = _namespace["run_pilot"](
    dict(globals()), folds=(1, 2), realizations=3, candidate_budget=256
)
