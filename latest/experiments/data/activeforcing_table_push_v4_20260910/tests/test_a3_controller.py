def test_first_command_is_zero_and_ramp_bounded():
 from a3_control import C
 c=C(20.5);assert c.command() == 0.;c.observe(18.,True);assert abs(c.command())<=.5
def test_contact_loss_resets():
 from a3_control import C
 c=C(20.5);c.observe(18.,True);c.observe(0.,False);assert c.command()==0.
