class C:
 def __init__(self,target):self.t=target;self.ema=None;self.i=0.;self.u=0.;self.valid=False
 def observe(self,force,contact):
  if not contact:self.i=0.;self.u=0.;self.valid=False;return
  self.ema=force if self.ema is None else .5*force+.5*self.ema; e=self.t-self.ema; self.i=max(-4,min(4,self.i+e));want=max(-2,min(2,.5*e+.05*self.i));self.u=max(self.u-.5,min(self.u+.5,want));self.valid=True
 def command(self):return self.u if self.valid else 0.
