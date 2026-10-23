if RUN_FULL_RETENTION:
    begin_check_session("full_retention")
    require_smoke_pass(OUTPUT_DIR)
    FULL_EXP = FULL_EXP or open_full_experiment()
    verify_retention_grid(FULL_EXP.cfg, "full")
    verify_pcm_physics(FULL_EXP.cfg, DEVICE, "full")
    verify_driftslice_mechanics(FULL_EXP.cfg, DEVICE, "full")
    estimate_and_report_retention(FULL_EXP, "retention")
    run_retention_stage(FULL_EXP)
    end_stage_checks(FULL_EXP, "full_retention")
else:
    print("RUN_FULL_RETENTION=False")
