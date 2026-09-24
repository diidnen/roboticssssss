"""Original arbitration with an explicit, audited simulator force-range binding.

The user authorized investigating a different force range. Only constructor
support validation may change; utility, release threshold, feedback and action
equations never do. Default support is exactly the original [3,5] N.
"""
import ast
from copy import deepcopy
import hashlib
from pathlib import Path
import numpy as np

SOURCE=Path('/media/volume/newdata/exouser/online_vla_activeforcing_20260907/final_continuous_friction_generalization_v1/SOURCE_SNAPSHOT/arbitration.py')


def load_arbitration(support=(3.,5.)):
    lo,hi=map(float,support)
    if not (0<lo<=hi<=8):raise ValueError('Engineering support must remain positive and at most 8 N')
    tree=ast.parse(SOURCE.read_text());original=ast.dump(tree)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Arbitration')
    ctor=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    guard=ctor.body[0]
    if not isinstance(guard,ast.If) or ast.unparse(guard.test)!='not 3 <= force <= 5':
        raise RuntimeError('Original arbitration force guard drifted')
    old=guard.test
    guard.test=ast.parse('not SUPPORT[0] <= force <= SUPPORT[1]',mode='eval').body
    executable=deepcopy(tree)
    guard.test=old
    if ast.dump(tree)!=original:raise RuntimeError('Undeclared arbitration change')
    ns={'SUPPORT':(lo,hi)}
    exec(compile(ast.fix_missing_locations(executable),str(SOURCE),'exec'),ns)
    return ns['Arbitration'],{'original_source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'force_support_bilateral_N':[lo,hi],'only_force_guard_binding_changed':True,
        'original_AST_recovered_exactly':True,'utility_changed':False,
        'release_and_feedback_rules_changed':False}


if __name__=='__main__':
    import importlib.util
    spec=importlib.util.spec_from_file_location('_reference_arbitration',SOURCE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cls,receipt=load_arbitration()
    rng=np.random.default_rng(130913)
    for _ in range(1000):
        force=rng.uniform(3,5);handoff=rng.uniform(0,.04)
        action=rng.normal(size=13).astype(np.float32);action[6]=rng.uniform(0,.045)
        a,b=module.Arbitration(force,handoff),cls(force,handoff)
        x,rx=a.action(action);y,ry=b.action(action)
        assert np.array_equal(x,y) and rx==ry
    cls,receipt=load_arbitration((3,8))
    action=np.zeros(13,np.float32)
    value,decision=cls(8,.006).action(action)
    assert value[9]==4 and value[12]==4 and not decision['vla_release_intent']
    action[6]=.04
    value,decision=cls(8,.006).action(action)
    assert value[9]==value[12]==0 and decision['vla_release_intent']
    print('ORIGINAL_ARBITRATION_PARITY_1000_AND_8N_UNITS_PASSED',receipt)
