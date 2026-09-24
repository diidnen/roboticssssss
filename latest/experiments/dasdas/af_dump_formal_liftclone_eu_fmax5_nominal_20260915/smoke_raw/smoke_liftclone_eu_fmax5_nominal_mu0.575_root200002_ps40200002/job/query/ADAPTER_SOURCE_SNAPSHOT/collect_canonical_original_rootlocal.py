"""Original formal collection plus the prospectively qualified initializer."""
import os
import qualify_native_interfaces as base
from collect_original_rootlocal import qualify
from canonical_formal_binding import install

if __name__=='__main__':
    os.environ['AF_QUALIFICATION_BEFORE_SUPPLIED_GRASP']='1'
    install();base.qualify=qualify;base.main()
