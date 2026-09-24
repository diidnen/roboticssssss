# Worktree recovery layout

Nine FORTE lane worktrees plus one CPU coordinator worktree were created from FORTE commit `7f88d0184c1617ed95e67502da96e60be07b3689`. Each lane has a unique branch and output namespace.

Tabero dedicated recovery checkouts were attempted but failed closed because `/home/exouser` is 100% full (`58G/58G`, about 441MB free). No Tabero worktree was registered by Git and no original Tabero directory was changed. Existing Tabero worktrees remain untouched and are read-only dependencies until disk capacity is restored.

No agent is authorized to bypass this storage gate by deleting or overwriting experiment evidence.
