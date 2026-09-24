# V5 hybrid force/motion OSC

V4 validated an additive torque adapter but falsified its capped additive tracking architecture. V5 locally copies stock OSC torque math only for enabled hybrid mode. It uses stock `J_pos`, mass matrix, lambda decoupling, orientation torque, gravity compensation and nullspace terms. It replaces the scalar projection of stock decoupled translational force along fixed world XY direction `d`; disabled mode calls the original bound controller method exactly.

Every decisive test is a fresh-process replay diagnostic. No V5 online pi0 or task outcome occurs before replay R2H and A3 pass.
