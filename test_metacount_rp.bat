@echo off
:: -------------------------------------------------------
:: MetaCount RP test batch – runs the hierarchical CMF
:: with random‑parameters and (optionally) a grouped effect.
::
:: Prerequisites
::   * Conda environment "zigenv" must be on the PATH.
::   * metacountregressor package installed (v1.0.145 + selection‑guards).
::
:: What this does
::   1. Activates the environment.
::   2. Prints the version and checks for RP markers in the generator.
::   3. Runs a quick QLD heavy‑vehicles CMF with --rp-in-search --rp-sweep.
::   4. Optional: add --rdm-terms "log_aadt:normal" to see a grouped random
::      parameter for AADT (the term’s effect varies by site).
:: -------------------------------------------------------

:: 1. Activate conda
call "C:\Users\ahernz\miniconda3\shell\condabin\conda.sh" activate zigenv

:: 2. Version & marker check
python - <<'PYEOF'
import metacountregressor, os, sys
print("metacountregressor version :", metacountregressor.__version__)
F = os.path.join(os.path.dirname(metacountregressor.__file__),
                 "scripts\generate_washington_hierarchical_cmf_assets.py")
txt = open(F, encoding="utf-8", errors="replace").read()
for m in ["_RP_DISTS","_ninsig_rank","Selection BIC","Backward elimination",
          "SparseEA","rdm_terms","random_params"]:
    print(f"{m}: {m in txt}")
PYEOF

:: 3. Run the CMF (QLD data) – you can change the input path for WA
python - <<'PYEOF'
import metacountregressor as mc
import pathlib, sys

inp  = pathlib.Path("data/qld_heavy_vehicles.csv").resolve()
out  = pathlib.Path("results/qld_test_rp").resolve()

cmd = [
    "--input", str(inp),
    "--output-dir", str(out),
    "--y-col", "Headon",
    "--aadt-col", "AADT",
    "--search-iter", "400",
    "--search-method", "ga-adaptive",
    "--families", "both",
    "--candidate-profile", "expanded",
    "--max-upper-terms", "10",
    "--max-lower-terms", "4",
    "--allow-nonmonotonic-fallback",
    "--no-require-benchmark-dominance",
    "--no-require-final-beat-benchmark-both",
    "--seed", "43",
    "--rp-in-search",
    "--rp-sweep",
    "--rp-max-random-terms", "4",
    "--rp-draws", "500",
    "--convergence-early-stop",
    "--conv-patience", "25",
    "--conv-harmony-spread", "2.0",
    "--max-insig-in-search", "2",
]

# Uncomment the next line to test a *grouped* random‑parameter for AADT:
# cmd.extend(["--rdm-terms", "log_aadt:normal"])

# To test a *site‑grouped* random parameter (SD estimated per road segment),
# uncomment the next line and adjust term/distribution as needed.
# cmd.extend(["--rdm-group", "log_aadt:normal"])

# Execute the generator (same entry‑point used by the PBS scripts)
mc.commands.generate_washington_hierarchical_cmf_assets.main(cmd)
print("\n=== DONE ===")
PYEOF

echo.
echo Results (if any) are in results/qld_test_rp/.
echo Look for:
echo   • random_params_summary.json   (fixed + random‑param results)
echo   • coefficients_with_pvalues.csv
echo   • hierarchical_cmf_summary.md
echo.
pause