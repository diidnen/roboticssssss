# Main 720 paired force-adaptation benchmark

Archive-compatible offline results are summarized from the frozen grouped-root OOF selector artifact. Probe rows use the point friction estimate; GT-Direct is offline oracle; NoProbe-Direct is Query-Ignored and is not zero-query.

| method | scope | n | SR | mean force (N) | under-force | excess (N) | utility |
|---|---|---:|---:|---:|---:|---:|---:|
| GT-Direct | 72 contexts × 2 repeats | 144 | 0.9514 | 4.225 | 0.0347 | 0.404 | 0.1001 |
| GT-Direct | task0 | 36 | 1.0000 | 4.272 | 0.0000 | 0.576 | 0.1455 |
| GT-Direct | task1 | 36 | 0.9444 | 5.120 | 0.0000 | 0.347 | 0.0900 |
| GT-Direct | task5 | 36 | 1.0000 | 4.285 | 0.0000 | 0.588 | 0.1429 |
| GT-Direct | task6 | 36 | 0.8611 | 3.221 | 0.1389 | 0.102 | 0.0221 |
| NoProbe-Direct | 72 contexts × 2 repeats | 144 | 0.9444 | 4.477 | 0.0417 | 0.660 | 0.0487 |
| NoProbe-Direct | task0 | 36 | 1.0000 | 4.375 | 0.0000 | 0.678 | 0.1250 |
| NoProbe-Direct | task1 | 36 | 0.9444 | 5.807 | 0.0000 | 1.075 | -0.0245 |
| NoProbe-Direct | task5 | 36 | 0.9444 | 4.509 | 0.0556 | 0.811 | 0.0365 |
| NoProbe-Direct | task6 | 36 | 0.8889 | 3.216 | 0.1111 | 0.097 | 0.0579 |
| ProbeRich-Direct | 72 contexts × 2 repeats | 144 | 0.9236 | 4.093 | 0.0625 | 0.271 | 0.0932 |
| ProbeRich-Direct | task0 | 36 | 1.0000 | 4.194 | 0.0000 | 0.498 | 0.1611 |
| ProbeRich-Direct | task1 | 36 | 0.9444 | 5.042 | 0.0000 | 0.265 | 0.1030 |
| ProbeRich-Direct | task5 | 36 | 0.8889 | 4.012 | 0.1111 | 0.315 | 0.0623 |
| ProbeRich-Direct | task6 | 36 | 0.8611 | 3.124 | 0.1389 | 0.005 | 0.0463 |
| ProbeScalar-Direct | 72 contexts × 2 repeats | 144 | 0.9375 | 4.202 | 0.0486 | 0.381 | 0.0879 |
| ProbeScalar-Direct | task0 | 36 | 1.0000 | 4.289 | 0.0000 | 0.592 | 0.1422 |
| ProbeScalar-Direct | task1 | 36 | 0.9444 | 5.218 | 0.0000 | 0.451 | 0.0736 |
| ProbeScalar-Direct | task5 | 36 | 0.9444 | 4.136 | 0.0556 | 0.439 | 0.0993 |
| ProbeScalar-Direct | task6 | 36 | 0.8611 | 3.163 | 0.1389 | 0.044 | 0.0365 |

Exact frozen 0.25N grid evaluation remains a data gap because the 720 archive contains continuous context-specific force draws. No nearest-force substitution was used.
