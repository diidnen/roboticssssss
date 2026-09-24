"""Original formal collection with the conditionally qualified fixed rim binding."""
import os
import qualify_native_interfaces as base
from collect_original_rootlocal import qualify
from rim20_formal_binding import install

if __name__=='__main__':
    os.environ['AF_QUALIFICATION_BEFORE_SUPPLIED_GRASP']='1'
    install();base.qualify=qualify;base.main()
