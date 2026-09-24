"""Query-only native annotation offset; original P4/control remain untouched."""
import os
import qualify_native_interfaces as base
import qualify_original_p4_native as query
from canonical_open_ready_query import CanonicalOpenReadyQuery
from rootlocal_collection_contract import write,sha


class AlignedDiagnosticQuery(CanonicalOpenReadyQuery):
    def __init__(self,env,out):
        self.annotation_offset=float(os.environ['AF_DIAGNOSTIC_RIM_OFFSET_M'])
        if self.annotation_offset not in (0.,.01,.02):raise ValueError('Unplanned geometry offset')
        super().__init__(env,out)
        write(out/'RIM_ALIGNMENT_DIAGNOSTIC.json',{
            'diagnostic_only':True,'pre_dis_offset_m':self.annotation_offset,
            'original_P4_and_squeeze_unchanged':True,'same_contact_annotation':True,
            'force_or_observation_override':False,'source_sha256':sha(__file__)})

    def canonicalize_open_ready(self,start):
        shifted=start if self.annotation_offset==0. else self.native.get_grasp_pose(
            self.native.deskbin,self.arm_tag,contact_point_id=self.contact_id,
            pre_dis=.12+self.annotation_offset)
        return super().canonicalize_open_ready(shifted)

    def geometry_pose(self,pre_dis):
        return super().geometry_pose(pre_dis+self.annotation_offset)


if __name__=='__main__':
    os.environ['AF_QUALIFICATION_BEFORE_SUPPLIED_GRASP']='1'
    query.NativeP4QueryEnv=AlignedDiagnosticQuery
    base.qualify=query.qualify
    base.main()
