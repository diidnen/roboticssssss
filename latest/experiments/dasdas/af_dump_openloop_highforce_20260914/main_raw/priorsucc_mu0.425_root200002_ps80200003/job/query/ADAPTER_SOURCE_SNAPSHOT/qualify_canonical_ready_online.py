"""Canonical preparation followed by unchanged original query/fork executor."""
import os
import qualify_native_interfaces as base
import qualify_original_p4_native as query
import qualify_original_online_forks as online
from canonical_open_ready_query import CanonicalOpenReadyQuery

if __name__=='__main__':
    os.environ['AF_QUALIFICATION_BEFORE_SUPPLIED_GRASP']='1'
    query.NativeP4QueryEnv=CanonicalOpenReadyQuery
    online.qualify_query=query.qualify
    base.qualify=online.qualify
    base.main()
