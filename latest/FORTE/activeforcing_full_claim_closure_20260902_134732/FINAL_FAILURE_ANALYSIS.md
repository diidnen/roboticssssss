# Final failure analysis

The taxonomy schema is frozen and populated only with observables present in
the current archive. The 720 summary exposes full-task failure and telemetry
paths, but not an authoritative grip-related transport/place failure code for
every row. Therefore downstream failure candidates are retained in
`DELAYED_FAILURE_CASES.csv` with `grip_related=unknown`; they are not silently
counted as confirmed delayed grip failures. Paired case transitions for the
current Direct/NoPhysical/GT selector are in `MAIN_PAIRED_BENCHMARK_RESULTS.json`.
