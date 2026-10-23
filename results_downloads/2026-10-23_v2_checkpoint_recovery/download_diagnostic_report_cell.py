# Run only if the recheck cell stopped and you need to share its report.
from google.colab import files
_v2_reports = list((FULL_EXP.run_dir / "checks").glob("checkpoint_recheck_PCM-HWA_fold0_*.json"))
assert _v2_reports, "Run the checkpoint recheck cell first"
_v2_latest = max(_v2_reports, key=lambda p: p.stat().st_mtime)
files.download(str(_v2_latest))
