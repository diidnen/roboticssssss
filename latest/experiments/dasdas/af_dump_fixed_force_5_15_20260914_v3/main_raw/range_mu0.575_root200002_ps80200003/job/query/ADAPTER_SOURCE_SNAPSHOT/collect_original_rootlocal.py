"""One frozen TRAIN/VAL query group, exact shared native full-task execution."""
import os
from pathlib import Path
import qualify_native_interfaces as base
from qualify_original_online_forks import qualify as run_group
from rootlocal_collection_contract import read


def qualify(env,out):
    context=read(Path(os.environ['AF_COLLECTION_CONTEXT']))
    env._af_original_collection_context=context
    return run_group(env,out)


if __name__=='__main__':
    os.environ['AF_QUALIFICATION_BEFORE_SUPPLIED_GRASP']='1'
    base.qualify=qualify
    base.main()
