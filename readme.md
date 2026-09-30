# ICASSP2027 FAST HIERARCHICAL ADMM FOR LARGE-APERTURE DIRECTION-OF-ARRIVAL ESTIMATION

Requires MATLAB with Signal Processing Toolbox. Run from this directory:

```matlab
test_hadmm               % General solver: quick test
test_hadmm('full')       % General solver: full experiment
test_hadmm('paper')      % Standalone fixed-configuration solver: quick test
test_hadmm('paper-full') % Standalone fixed-configuration solver: full experiment
```

Direct call: `angles = hadmm_solver_paper(Y, source_count)` (sensors-by-snapshots input).
The configurable implementations are `hadmm_solver.m` and `hadmm_solver_diagnostic.m`.

RMSE and runtime plots are displayed automatically and saved as
`rmse_runtime.png` and `rmse_runtime.pdf` in the generated `results_*` directory,
alongside `summary.csv` and the MAT files.
