"""Paper export integrity tests with synthetic records, without any model execution."""
from dataclasses import replace
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import test_coordinator_transfer as base

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("paper_export", ROOT/"paper_no_ablation_export.py")
export = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = export
spec.loader.exec_module(export)


class PaperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.r = base.load_runtime(ROOT/"DriftSlice_ISCAS_P0_coordinator_FIXED.ipynb", cls.root/"runtime.py")
        notebook = json.loads((ROOT/"DriftSlice_ISCAS_P0_coordinator_FIXED.ipynb").read_text())
        for cell in notebook["cells"]:
            text = "".join(cell["source"])
            if cell["cell_type"] == "code" and ("def ieee_style()" in text or "def generate_paper_results(" in text):
                exec(compile(text, "paper_helpers.py", "exec"), cls.r.__dict__)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def fixture(self):
        r = self.r
        folder = self.root/f"fixture-{self._testMethodName}"
        folder.mkdir()
        methods = (r.HWA, r.DAR, r.ATT, r.NSR)
        cfg = replace(r.make_config("resnet32", False, "fixture"), require_cuda=False,
                      epochs=2, digital_epochs=2, train_repeats=3, realizations=2)
        exp = SimpleNamespace(cfg=cfg, arch="resnet32", methods=methods, models=(r.DIRECT,)+methods,
            train_dir=folder/"training", ret_dir=folder/"retention", run_dir=folder,
            diag_dir=folder/"diagnostics", fp={"train":"fixture-train", "retention":"fixture-ret", "diagnostics":"fixture-diag"},
            device=torch.device("cpu"), data_hashes={"fixture":"synthetic"})
        comps, times = r.compensations_for(cfg), r.exact_retention_times(cfg)
        for fold in range(3):
            fdir = exp.train_dir/f"fold{fold}"
            fdir.mkdir(parents=True)
            state = {"fixture":torch.tensor([100+fold])}
            parent = r.state_dict_hash(state)
            history = [dict(fold=fold, epoch=e, train_accuracy=80+e, val_accuracy=80+e,
                val_ce=1/e, optimizer_steps=e, best_epoch_so_far=e) for e in (1,2)]
            torch.save(dict(kind="digital_best_validation", train_fp="fixture-train", fold=fold,
                model=state, model_identity=parent, digital_epochs_completed=2,
                history=history, optimizer_steps=2, best_epoch=2, best_val_accuracy=82),
                fdir/"digital_best_validation.pt")
            ids = {}
            for j, method in enumerate(exp.models):
                state = {"fixture":torch.tensor([j,fold])}
                ident = r.state_dict_hash(state)
                ids[method] = ident
                direct = method == r.DIRECT
                h = [dict(fold=fold, method=method, epoch=e, train_accuracy=85+e,
                    val_accuracy=84+e+fold/10, val_loss=1/e, best_epoch=e,
                    optimizer_steps=e, optimizer="AnalogAdam", train_extra=0 if method == r.HWA else .01)
                    for e in (1,2)]
                name = f"{method}_analog_checkpoint.pt" if direct else f"{method}_best_validation.pt"
                torch.save(dict(kind="fp32_direct" if direct else "hwa_best_validation",
                    train_fp="fixture-train", fold=fold, method=method, model=state, model_identity=ident,
                    digital_identity=parent, optimizer_steps=0 if direct else 2,
                    analog_epochs_completed=0 if direct else 2, history=h,
                    selected_best_validation_epoch=2, val_accuracy=86+fold/10, transfer_errors=[]), fdir/name)
                for real in range(2):
                    path = exp.ret_dir/f"fold{fold}"/method/f"real{real:02d}.pt"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    rows = []
                    for k, comp in enumerate(comps):
                        for ti, ts in enumerate(times):
                            error = 5+3*j+fold*(1+.05*j)+.1*real+.01*ti+.4*k
                            rows.append(dict(method=method, fold=fold, realization=real, compensation=comp,
                                t_index=ti, t_s=ts, selected_bins=1, split="test", accuracy=100-error,
                                error=error, loss=error/100, gdc_valid=True if comp=="GDC-safe" else np.nan))
                    torch.save(dict(fp="fixture-ret", model_identity=ident, selected_bins=1,
                        results=pd.DataFrame(rows), meta=dict(root_seed=cfg.seed+fold*1_000_000+real*10_000,
                            state_hash_start="fixture-state", state_hash_end="fixture-state")), path)
            p = exp.ret_dir/f"fold{fold}"/"selected_bins.json"
            p.write_text(json.dumps(dict(fold=fold,bins=1,selection_method=r.HWA,
                selection_model_identity=ids[r.HWA], selection="validation_only_5yr")))
        runtime = dict(r.__dict__)
        # Any accidental training / evaluation / ablation call must fail immediately.
        for name in ("run_training_stage", "run_retention_stage", "run_ablation_stage", "run_diagnostics_stage",
                     "collect_shards", "evaluate_model", "train_hwa_resnet"):
            runtime[name] = unittest.mock.Mock(side_effect=AssertionError(f"Unexpected execution: {name}"))
        return runtime, exp

    def test_complete_export_without_ablation_or_diagnostics(self):
        runtime, exp = self.fixture()
        before = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in exp.run_dir.rglob("*") if p.is_file()}
        out = export.run_export(runtime, exp, download=False)
        status = json.loads((out/"export_status.json").read_text())
        self.assertTrue(status["core_training_retention_complete"])
        self.assertFalse(status["ablation_included"])
        self.assertFalse(status["diagnostics"]["included"])
        self.assertEqual(status["retention_records"], 3*5*2*12*4)
        self.assertFalse(any("ablation" in p.name.lower() for p in out.iterdir()))
        table = pd.read_csv(out/"TableI_five_year_test_error.csv")
        row = table[table.method.eq(self.r.HWA) & table.compensation.eq("DS-B1")].iloc[0]
        self.assertAlmostEqual(row.mean_error_pct, 10.01)
        self.assertAlmostEqual(row.sd_across_folds_pct, 1.05)
        self.assertIn("(2 epochs)", (out/"fig1_training_validation_caption.txt").read_text())
        self.assertNotIn("154 epochs", (out/"fig1_training_validation_caption.txt").read_text())
        self.assertTrue((out/"fig3_matched_compensation.pdf").exists())
        self.assertTrue(out.with_suffix(".zip").exists())
        self.assertEqual(before, {p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in before})
        for name in ("run_training_stage", "run_retention_stage", "run_ablation_stage", "run_diagnostics_stage"):
            runtime[name].assert_not_called()

    def test_missing_retention_does_not_generate_partial_paper(self):
        runtime, exp = self.fixture()
        next(exp.ret_dir.glob("fold*/*/real*.pt")).unlink()
        with self.assertRaisesRegex(export.ExportError, "Retention is incomplete"):
            export.run_export(runtime, exp, download=False)
        self.assertFalse((exp.run_dir/"paper_NO_ABLATION").exists())

    def test_corrupted_checkpoint_rejected(self):
        runtime, exp = self.fixture()
        path = exp.train_dir/"fold0"/f"{self.r.NSR}_best_validation.pt"
        p = self.r.torch_load(path)
        p["model"]["fixture"][0] += 1
        torch.save(p,path)
        with self.assertRaisesRegex(export.ExportError, "Content hash mismatch"):
            export.run_export(runtime, exp, download=False)

    def test_wrong_shard_identity_rejected(self):
        runtime, exp = self.fixture()
        path = exp.ret_dir/"fold0"/self.r.NSR/"real00.pt"
        p = self.r.torch_load(path)
        p["model_identity"]="wrong"
        torch.save(p,path)
        with self.assertRaisesRegex(export.ExportError, "Foreign model identity"):
            export.run_export(runtime, exp, download=False)

    def test_invalid_gdc_is_not_silently_averaged(self):
        runtime, exp = self.fixture()
        path = exp.ret_dir/"fold0"/self.r.HWA/"real00.pt"
        p = self.r.torch_load(path)
        mask = p["results"].compensation.eq("GDC-safe") & p["results"].t_index.eq(11)
        p["results"].loc[mask,["accuracy","error","loss"]]=np.nan
        p["results"].loc[mask,"gdc_valid"]=False
        torch.save(p,path)
        out = export.run_export(runtime, exp, download=False)
        table=pd.read_csv(out/"TableI_five_year_test_error.csv")
        row=table[table.method.eq(self.r.HWA)&table.compensation.eq("GDC-safe")].iloc[0]
        self.assertTrue(pd.isna(row.mean_error_pct))
        self.assertEqual(row.invalid_records,1)
        status=json.loads((out/"export_status.json").read_text())
        self.assertTrue(status["core_training_retention_complete"])
        self.assertFalse(status["all_retention_accuracies_valid"])

    def test_existing_diagnostics_included_without_running_them(self):
        runtime, exp = self.fixture()
        r = self.r
        for fold in range(3):
            for method in (r.HWA, r.NSR):
                ident = r.torch_load(exp.train_dir/f"fold{fold}"/f"{method}_best_validation.pt")["model_identity"]
                for real in range(2):
                    source = r.torch_load(exp.ret_dir/f"fold{fold}"/method/f"real{real:02d}.pt")["results"]
                    error = float(source[source.compensation.eq("DS-B1") & source.t_index.eq(11)].error.iloc[0])
                    rows = []
                    for layer in ("conv", "fc"):
                        row = {key: 1.0 for key, *_ in r.TABLE2_ROWS if key != "ref_gain_B1_sd"}
                        row.update(fold=fold,method=method,realization=real,layer=layer,
                                   test_error_B1_5y_pct=error,ref_gain_rel_error_pct=.1)
                        rows.append(row)
                    p = exp.diag_dir/f"fold{fold}"/method/f"real{real:02d}.pt"
                    p.parent.mkdir(parents=True,exist_ok=True)
                    torch.save(dict(fp="fixture-diag",model_identity=ident,selected_bins=1,results=pd.DataFrame(rows)),p)
        out=export.run_export(runtime,exp,download=False)
        self.assertTrue((out/"TableII_mechanism_diagnostics.csv").is_file())
        status=json.loads((out/"export_status.json").read_text())
        self.assertTrue(status["diagnostics"]["included"])
        self.assertTrue(any(x["output"]=="Figure 5 NSR distributions" for x in status["omitted_outputs"]))
        runtime["run_diagnostics_stage"].assert_not_called()

    def test_ready_notebook_preserves_scientific_definitions(self):
        original=json.loads((ROOT/"DriftSlice_ISCAS_P0_coordinator_FIXED.ipynb").read_text())
        ready=json.loads((ROOT/"DriftSlice_ISCAS_P0_coordinator_PAPER_NO_ABLATION.ipynb").read_text())
        def definitions(book):
            result={}
            for cell in book["cells"]:
                if cell["cell_type"]=="code":
                    for node in ast.parse("".join(cell["source"])).body:
                        if isinstance(node,(ast.FunctionDef,ast.ClassDef,ast.AsyncFunctionDef)):
                            result[node.name]=ast.dump(node,include_attributes=False)
            return result
        self.assertEqual(definitions(original),definitions(ready))
        for cell in ready["cells"]:
            if cell["cell_type"]=="code":
                source="".join(cell["source"])
                compile(source,"ready_cell","exec")
                self.assertNotIn("runtime.unassign()",source)
                if "# ============================ EXECUTION CONTROL" in source:
                    assignments={node.targets[0].id:node.value.value for node in ast.parse(source).body
                        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name)
                        and isinstance(node.value,ast.Constant)}
                    for flag in ("RUN_FULL_TRAINING","RUN_FULL_RETENTION","RUN_REFERENCE_ABLATION",
                                 "RUN_MECHANISM_DIAGNOSTICS","GENERATE_PAPER_RESULTS"):
                        self.assertFalse(assignments[flag])


if __name__ == "__main__":
    unittest.main(verbosity=2)
