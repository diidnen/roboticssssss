"""Original formal AF inference plus the same qualified initializer as TRAIN."""
import os
import qualify_native_interfaces as base
from infer_original_rootlocal import qualify
from canonical_formal_binding import install

if __name__=='__main__':
    os.environ['AF_QUALIFICATION_BEFORE_SUPPLIED_GRASP']='1'
    install();base.qualify=qualify;base.main()
