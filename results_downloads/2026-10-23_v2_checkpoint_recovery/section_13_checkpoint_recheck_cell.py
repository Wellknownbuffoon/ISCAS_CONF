# Add this NEW cell immediately before the full-training runner in Section 13.
# Run in the ORIGINAL uploaded v2 notebook with its definitions/smoke test loaded.
# No training or test evaluation is performed by this cell.
import urllib.request
import hashlib

require_smoke_pass(OUTPUT_DIR)
_v2_recovery_url = 'https://raw.githubusercontent.com/Wellknownbuffoon/ISCAS_CONF/2a40ff3b34406e6b932f00d195cf1ef5795df488/v2_checkpoint_recovery.py'
_v2_recovery_code = urllib.request.urlopen(_v2_recovery_url, timeout=60).read()
assert hashlib.sha256(_v2_recovery_code).hexdigest() == 'fa3dd8ce5bc307bffb4a38c13a988ee59d446dfbc3a4e50799cc1608748c208a', "Recovery source hash mismatch"
_v2_recovery_ns = {"__name__": "v2_checkpoint_recovery"}
exec(compile(_v2_recovery_code, _v2_recovery_url, "exec"), _v2_recovery_ns)
FULL_EXP = globals().get("FULL_EXP") or open_full_experiment()
_v2_recovery_result = _v2_recovery_ns["check_v2_checkpoint"](
    globals(), FULL_EXP, fold=0, method=HWA, repair=True,
    expected_train_prefix="1cb6defff3c3",
)
