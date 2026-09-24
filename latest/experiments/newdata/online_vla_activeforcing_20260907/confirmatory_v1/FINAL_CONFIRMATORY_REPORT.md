# Final confirmatory online-VLA evaluation

Runtime freeze: PASS. Manifest SHA256: `f1441d83aa4eba7a5177815b5c25fde3302b95de548e0878a9e6b88f2a5d5f58`.
New roots: 170044, 170045, 170046, 170047. All 192 confirmatory branches completed; all passed online-VLA provenance and geometry admission. No retry was used. Physics is now stopped.

## New four-root confirmatory table

| Method | Full-task | Lift | Drop | Mean selected (N) | Mean measured bilateral squeeze (N) | Saving vs Fixed-5 measured (N) |
|---|---:|---:|---:|---:|---:|---:|
| FIXED_3 | 21/48 (43.8%) | 91.7% | 39.6% | 3.000 | 1.746 | 2.941 |
| FIXED_4 | 34/48 (70.8%) | 100.0% | 16.7% | 4.000 | 3.483 | 1.204 |
| FIXED_5 | 35/48 (72.9%) | 100.0% | 8.3% | 5.000 | 4.687 | 0.000 |
| ACTIVEFORCING | 37/48 (77.1%) | 100.0% | 10.4% | 3.943 | 3.496 | 1.191 |

## Eight-root pooled descriptive table

| Method | Full-task | Lift | Drop | Mean selected (N) | Mean measured bilateral squeeze (N) | Saving vs Fixed-5 measured (N) |
|---|---:|---:|---:|---:|---:|---:|
| FIXED_3 | 37/96 (38.5%) | 90.6% | 40.6% | 3.000 | 1.762 | 2.927 |
| FIXED_4 | 65/96 (67.7%) | 99.0% | 16.7% | 4.000 | 3.443 | 1.246 |
| FIXED_5 | 71/96 (74.0%) | 99.0% | 7.3% | 5.000 | 4.689 | 0.000 |
| ACTIVEFORCING | 69/96 (71.9%) | 100.0% | 10.4% | 3.987 | 3.519 | 1.170 |

## Paired confirmatory results

- AF vs Fixed-3: both success 19, AF-only 18, baseline-only 2, both fail 9.
- AF vs Fixed-4: both success 30, AF-only 7, baseline-only 4, both fail 7.
- AF vs Fixed-5: both success 33, AF-only 4, baseline-only 2, both fail 9; force saving in both-success contexts = 1.323 N.

## Root consistency

New-root AF measured-force saving versus Fixed-5: 1.343 N, 0.806 N, 1.222 N, 1.393 N; direction is positive in all four roots.
New-root AF–Fixed-5 success differences: -0.083, +0.083, +0.167, +0.000; success direction is mixed, so this is not a replication of a uniform reliability advantage.

## Failure composition

Among pooled AF-fail/Fixed-5-success contexts: 9 total; post-lift geometric=6, under-force=3, VLA execution=0, other=0.
Diagnosis labels: execution variance=6, utility aggressive=3, model error=0.

## Scientific conclusion

The new four roots reproduce the measured-force-saving direction in every root, but the success comparison with Fixed-5 is mixed. In the eight-root pooled descriptive evaluation, ActiveForcing achieves 69/96 full-task successes versus 71/96 for Fixed-5 while using 1.170 N less mean measured bilateral squeeze (24.9%). It improves over Fixed-4 in success count (69/96 versus 65/96) but has a similar measured force. The supported claim is a reliability–force trade-off on top of a frozen online VLA; Fixed-5 reliability is only partially matched, so non-inferiority or superiority should not be claimed.
