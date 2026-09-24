# Mass identification report

Primary split: ROOT-HELDOUT on the formal Mass dataset (TRAIN roots 8100--8103, TEST roots 8200--8201). Only QA-passed formal TRAIN contexts fit the identifier; TEST query outcomes are never used for fitting. The physical identifier is a small query-history regressor. Vision features are compact means/standard deviations from the two retained query RGB views, and are evaluated because all 36 RGB artifacts are present.

Direct downstream decision metrics use the same frozen candidate set and authoritative Expected Utility; they are reported in the force-adaptation table.

| Method | Seeds | MAE (kg) | RMSE (kg) | Median AE | Bias | Spearman | Pairwise rank | Low/Mid/High accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Prior / No Physical Information | 3 | 0.0556 | 0.0624 | 0.0667 | 0.0000 | nan | 0.000 | 0.667/0.333/0.667 |
| Vision Only | 3 | 0.2102 | 0.2355 | 0.1948 | 0.0984 | 0.438 | 0.578 | 0.500/0.667/0.500 |
| Physical History Only | 3 | 0.0229 | 0.0246 | 0.0240 | 0.0117 | 0.956 | 0.800 | 1.000/1.000/1.000 |
| Vision + Physical History | 3 | 0.0221 | 0.0288 | 0.0165 | 0.0024 | 0.956 | 0.800 | 1.000/0.833/0.833 |
| Explicit SysID | 3 | 0.0347 | 0.0465 | 0.0207 | -0.0239 | 0.837 | 0.733 | 0.833/0.833/1.000 |

## Interpretation

The selected ActiveForcing-Mass identifier is Physical History Only. Vision and fused rows are retained as honest comparisons; no visual backbone or image tuning is introduced. The formal benchmark has 12 TRAIN contexts and 6 ROOT-HELDOUT TEST contexts, so uncertainty is represented by three seeds and the small held-out population is stated explicitly.
