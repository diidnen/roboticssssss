"""Original physical collector with frozen per-context force-grid binding."""
import os
from pathlib import Path
import sys
OLD = Path('/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911/experiments/af_dump_original_restore_20260912')
sys.path.insert(0, str(OLD))
import rootlocal_collection_contract as contract
import qualify_native_interfaces as base
from rim20_formal_binding import install
from collect_original_rootlocal import qualify as original_full_group
from qualify_original_p4_native import qualify as original_query
from admit_original_query import admit
from audit_original_collected_group import audit_query


def qualify(env, out):
    context = contract.read(os.environ['AF_COLLECTION_CONTEXT'])
    contract.verify_runtime(context['runtime_manifest_path'], context['runtime_manifest_sha256'])
    contract.FORCES = list(context['forces_N'])
    contract.validate_context(context, env)
    if context['collection_mode'] == 'FULL_GROUP':
        return original_full_group(env, out)
    if context['collection_mode'] != 'QUERY_ONLY': raise ValueError('Unknown planned collection mode')
    out.mkdir(parents=True, exist_ok=False)
    original_query(env, out/'query')
    import sapien
    body = env.deskbin.actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
    contract.write(out/'query/ACTUAL_OBJECT_MATERIAL.json', {'shape_materials': [
        [float(s.physical_material.static_friction), float(s.physical_material.dynamic_friction)]
        for s in body.collision_shapes]})
    contract.write(out/'QUERY_ADMISSION.json', admit(out/'query', require_material=True))
    audit_query(out/'query')
    contract.write(out/'CONTEXT_LOCK.json', context)
    contract.write(out/'QUERY_ONLY_COMPLETE.json', {'completed': True, 'context_id': context['id'],
                   'downstream_labels': 0, 'query_audit_sha256': contract.sha(out/'query/INDEPENDENT_NATIVE_FORCE_REBUILD.json')})


if __name__ == '__main__':
    os.environ['AF_QUALIFICATION_BEFORE_SUPPLIED_GRASP'] = '1'
    install(); base.qualify = qualify; base.main()
