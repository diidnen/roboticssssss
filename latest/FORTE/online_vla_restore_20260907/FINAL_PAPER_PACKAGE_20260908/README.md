# Frozen VLA + ActiveForcing final evaluation package

Start with `paper/manuscript.pdf` and `FINAL_RESULTS_EXPLANATION_ZH.md`.

- Main evidence: four fresh reset groups,48contexts,192branches; Fixed3/4/5/AF full-task counts16/31/36/32 of48.
- Development ablations:96branches plus24matched AF controls; kept separate from fresh-test evidence.
- `FINAL_REQUIRED_STATUS.json` records runtime/model/results and explicit conditional/optional dispositions.
- `FINAL_REQUIREMENT_AUDIT.json` maps the25requirements to reviewed evidence; checkpoint and controller audits are separate.
- `paper/` includes editable TeX/BibTeX, PDF, tables, vector figures and supporting evidence. Compile `manuscript.tex` with the Tectonic toolchain identified by `TOOLCHAIN_IDENTITY.json`.
- The full raw rollout arrays and large model checkpoints remain at the absolute source paths in result/manifests. This is a paper/evidence package, not a self-contained simulator distribution.

Scope: online frozen VLA downstream from a common established grasp. AF chooses force once after the probe and uses continuous feedback control. The shared controller includes1.9x conditional feedforward; selected setpoint and measured force are distinct. Mean AF setpoint saving versus Fixed5 is19.375%, accompanied by four fewer full-task successes. No uncertainty-integration superiority over Posterior Mean is established by the small ablation.

The attached historical PDF is retained unchanged. These sources are a results-grounded replacement manuscript. No external submission or publication has been performed. Historic assembly manifests describe their original review-candidate stage; this package manifest and the later PDF review identify the delivered files.
