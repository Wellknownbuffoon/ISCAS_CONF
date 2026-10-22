"""Generate a coordinator notebook with all experiment-execution flags disabled."""
import json
from pathlib import Path
import re

root = Path(__file__).resolve().parent
n = json.loads((root/"DriftSlice_ISCAS_P0_coordinator_FIXED.ipynb").read_text())
n["cells"][0]["source"] = [
    "# P0: paper results without ablation\n",
    "\n",
    "Use existing completed P1–P4 ResNet/CIFAR-10 checkpoints and retention shards. "
    "The last code cell exports figures, tables, source CSVs and a ZIP. "
    "No experiment stages are enabled. Mechanism outputs use existing diagnostics only.\n"]
flags = ("RUN_SMOKE_TEST", "RUN_FULL_TRAINING", "RUN_FULL_RETENTION", "RUN_REFERENCE_ABLATION",
         "RUN_MECHANISM_DIAGNOSTICS", "GENERATE_PAPER_RESULTS")
for cell in n["cells"]:
    if cell["cell_type"] != "code":
        continue
    source = "".join(cell["source"])
    if "# ============================ EXECUTION CONTROL" in source:
        for flag in flags:
            source = re.sub(rf"(?m)^({flag}\s*=\s*)(True|False)", r"\g<1>False", source)
    if "runtime.unassign()" in source:
        source = 'print("Automatic runtime disconnect disabled for the paper export.")\n'
    cell["source"] = source.splitlines(keepends=True)
    cell["outputs"] = []
    cell["execution_count"] = None

module = (root/"paper_no_ablation_export.py").read_text()
source = ("# EXPORT ONLY: combine completed P1-P4; no ablation or new inference.\n"
          "# You may copy this entire last cell into an already initialized P0.\n"
          "# Module functions stay in a separate namespace and do not alter the smoke code hash.\n"
          f"_paper_export_source = {module!r}\n"
          "_paper_export_namespace = {'__name__': 'paper_no_ablation_export', '__source__': _paper_export_source}\n"
          "exec(compile(_paper_export_source, 'paper_no_ablation_export.py', 'exec'), _paper_export_namespace)\n"
          "PAPER_DIR_NO_ABLATION = _paper_export_namespace['run_export'](dict(globals()))\n")
n["cells"].extend([
    dict(cell_type="markdown", metadata={}, id="paper_no_ablation_instructions", source=[
        "## Paper export without ablation\n",
        "Run the following cell after initialization. It requires complete training and retention for "
        "FP32-Direct, HWA, DAR, DriftAttention and NSR, checks their identities and creates a new "
        "`paper_NO_ABLATION/` directory plus a downloadable ZIP. Missing diagnostic data are listed "
        "as omitted outputs; they do not block training/retention figures or tables.\n"]),
    dict(cell_type="code", metadata={}, id="paper_no_ablation_export", execution_count=None,
         outputs=[], source=source.splitlines(keepends=True))])
dest = root/"DriftSlice_ISCAS_P0_coordinator_PAPER_NO_ABLATION.ipynb"
dest.write_text(json.dumps(n, indent=1)+"\n")
print(dest)
