"""Fixed qualified native rim geometry with the unchanged original fork executor."""
import os
import qualify_native_interfaces as base
import qualify_original_p4_native as query
import qualify_original_online_forks as online
from diagnose_canonical_rim_alignment import AlignedDiagnosticQuery


if __name__=='__main__':
    if os.environ.get('AF_DIAGNOSTIC_RIM_OFFSET_M')!='.02':raise ValueError('Wrong fixed rim offset')
    os.environ['AF_QUALIFICATION_BEFORE_SUPPLIED_GRASP']='1'
    query.NativeP4QueryEnv=AlignedDiagnosticQuery
    online.qualify_query=query.qualify
    base.qualify=online.qualify
    base.main()
