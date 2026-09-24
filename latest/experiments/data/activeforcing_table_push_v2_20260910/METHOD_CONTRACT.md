# ActiveForcing table-push V2 method contract

V1 remains immutable and valid as a negative Gate-A result. V2 tests only the preregistered causal fresh-chunk direction and larger-but-bounded interface calibration.

At the third consecutive robot/plate contact step, queued actions are discarded and a fresh frozen-pi0 chunk is requested from the live observation. From that chunk, retain the first 15 XY actions with norm at least 0.01, take the coordinate-wise median, and normalize it. The raw chunk, indices, aggregate, direction, and SHA-256 are saved before a residual is applied. No simulator state, plate/goal pose, friction, outcome, or handcrafted world direction enters this calculation.

The selected quantity is horizontal robot-on-plate force projected on this direction. Each contact stores the raw contact-frame wrench, contact frame, sign convention, world force on plate, and summed projection. The residual changes action XY only; z, rotation, and gripper remain exactly nominal.

A2.1 is a five-episode maximum, root-0/mu-1.2 calibration falsifier. Its values are normalized OSC residuals—not Newtons—and are capped at 0.15 with a 0.02 per-step ramp. It must show retained contact, finite telemetry, non-impact-dominated steady response, and a usable signed response over a symmetric grid. Otherwise V2 stops. A2.2 may begin only after its frozen pass criteria are met.
