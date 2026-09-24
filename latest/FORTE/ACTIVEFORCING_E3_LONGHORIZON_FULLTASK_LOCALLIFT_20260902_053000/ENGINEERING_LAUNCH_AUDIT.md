# E3 engineering launch audit

These attempts did not execute an episode and are not scientific task failures.

| UTC | Output directory | Stage reached | Exit cause | GPU residue |
|---|---|---|---|---|
| 05:46 | `/media/volume/newdata/exouser/activeforcing_e3/QUALIFICATION_RUN_20260902_054633/task3` | argument schema construction | detached exec namespace made `typing.Optional` unavailable to Tyro postponed-annotation resolution | none |
| 05:47 | `/media/volume/newdata/exouser/activeforcing_e3/QUALIFICATION_RUN_20260902_054736_retry1/task3` | Isaac extension startup | launcher omitted authoritative B5 Warp import override and loaded incompatible site `warp-lang 1.16.0` (`warp.types.array` absent) | none after exit 05:47:55 |

Remediation is minimal and lineage-preserving:

1. Execute the historical source in the real `__main__` module namespace, so Tyro resolves its own imported annotation names.
2. Restore the exact B5 environment contract: `PYTHONNOUSERSITE=1` and Isaac bundled `omni.warp.core-1.8.2+lx64` first on `PYTHONPATH`, followed by Tabero and the project-local OpenPI client.

Read-only validation imported Warp from the bundled path, reported version 1.8.2, and confirmed `hasattr(warp.types, "array") == True`. No task outcome, force response, or ActiveForcing performance was observed or used to modify the protocol.
