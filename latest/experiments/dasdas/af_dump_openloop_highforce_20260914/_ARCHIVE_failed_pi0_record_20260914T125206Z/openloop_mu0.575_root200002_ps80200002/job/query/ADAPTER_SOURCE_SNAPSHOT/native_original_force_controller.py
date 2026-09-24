"""Unchanged AF arbitration/servo with native gripper-unit and clock bindings."""
import importlib.util
from pathlib import Path
import numpy as np
import qualify_native_interfaces as base

SOURCE=Path('/media/volume/newdata/exouser/online_vla_activeforcing_20260907/final_continuous_friction_generalization_v1/SOURCE_SNAPSHOT/arbitration.py')


class NativeOriginalForceController:
    def __init__(self,env,arm,force,handoff,stock_drives,force_support=(3.,5.)):
        from original_arbitration_binding import load_arbitration
        cls,self.binding_receipt=load_arbitration(force_support)
        self.controller=cls(force,handoff)
        self.servo,_=base.original_servo()
        self.env,self.arm,self.stock_drives=env,arm,stock_drives
        self.original_set_gripper=env.robot.set_gripper
        self.scale=getattr(env.robot,arm+'_gripper_scale')
        self.opened=False
        self.physics_steps=0
        self.next_feedback_step=self.clock_step(1)
        self.feedback_ticks=0
        self.action_receipts=[]
        self.inner=getattr(env,'_af_original_squeeze_inner',None)

    def clock_step(self,tick):
        return int(np.floor(tick*.05/self.env.physics_timestep+.5))

    def before_action(self,action):
        action=np.asarray(action,np.float32)
        if action.shape!=(14,) or not np.isfinite(action).all():raise ValueError('Actual native action required')
        offset=0 if self.arm=='left' else 7
        # Arbitration passes these six native arm numbers through unchanged;
        # they are NOT mislabeled Cartesian motion features. FK is separate.
        proxy=np.zeros(13,np.float32)
        proxy[:6]=action[offset:offset+6]
        proxy[6]=float(self.scale[0]+action[offset+6]*(self.scale[1]-self.scale[0]))
        result,receipt=self.controller.action(proxy)
        if not np.array_equal(result[:6],action[offset:offset+6]):raise RuntimeError('Arm target changed')
        self.opened=receipt['vla_release_intent']
        receipt.update(native_action14=action.tolist(),arm_channel_units='native joint radians, pass-through only',
                       native_vla_gripper_normalized=float(action[offset+6]),
                       servo_clock_period_s=.05)
        self.action_receipts.append(receipt)
        if self.opened or self.inner is not None:
            for joint,stiffness,damping,limit,mode in self.stock_drives:
                joint.set_drive_properties(stiffness,damping,limit,mode)
        else:self.env.robot.set_gripper_force_limit(self.controller.force/2,self.arm)

    def set_gripper(self,value,arm_tag,gripper_eps=.1):
        if str(arm_tag)!=self.arm:
            return self.original_set_gripper(value,arm_tag,gripper_eps=gripper_eps)
        command=(self.inner.last['inner_aperture_m'] if self.inner is not None and self.inner.last
                 else .04 if self.opened else self.controller.command)
        return self.original_set_gripper((command-self.scale[0])/(self.scale[1]-self.scale[0]),
                                         arm_tag,gripper_eps=0)

    def after_physics(self,measured_squeeze):
        self.physics_steps+=1
        if self.physics_steps==self.next_feedback_step:
            measured=(self.inner.last['measured_filtered_N'] if self.inner is not None else measured_squeeze)
            self.controller.feedback(measured,self.opened,self.servo)
            self.feedback_ticks+=1
            self.next_feedback_step=self.clock_step(self.feedback_ticks+1)
            self.set_gripper(0,self.arm)

    def before_physics(self):
        if self.inner is None:return
        measured=base.contacts(self.env,self.arm)['measured_squeeze_n']
        force=0. if self.opened else self.controller.force
        command=.04 if self.opened else self.controller.command
        actual=self.inner.step(measured,force,command)
        self.original_set_gripper((actual-self.scale[0])/(self.scale[1]-self.scale[0]),self.arm,gripper_eps=0)

    def install(self): self.env.robot.set_gripper=self.set_gripper
    def uninstall(self): self.env.robot.set_gripper=self.original_set_gripper
