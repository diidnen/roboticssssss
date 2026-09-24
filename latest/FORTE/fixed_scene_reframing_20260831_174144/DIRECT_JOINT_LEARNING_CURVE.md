# Direct vs Joint Fixed-Scene Learning Curve

The four subsets are nested by complete friction context within each scene family. The same 24 held-out-friction DEV contexts are used at every scale; DEV labels are never used for training or checkpoint selection.

| Data | Direct SR | Joint SR |
|---|---:|---:|
| 25% | 0.646 | 0.667 |
| 50% | 0.750 | 0.729 |
| 75% | 0.771 | 0.667 |
| 100% | 0.750 | 0.708 |

Primary metric: mean full-task success rate at the minimum force whose ensemble score is >=0.5. This is an offline fixed-scene held-out-friction diagnostic, not TEST evidence. Training budget and architecture are unchanged across scales.
