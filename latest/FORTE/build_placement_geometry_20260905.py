"""Offline asset extraction, no Isaac app or physics. Frozen before pilot."""
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from pxr import Usd,UsdGeom,UsdPhysics
from activeforcing_placement_contract import corners

ROOT=Path('/home/exouser/FORTE')
OUT=ROOT/'analysis/results/current_contract_restore_and_matched_pilot_20260905/history_reconstruction/placement_contract_repair'
ASSETS=Path('/home/exouser/Tabero/benchmarks/datasets/libero/USD')
ROBOT=Path('/home/exouser/Tabero/source/tac_manip/tac_manip/assets/data/Robots/Franka_gsmini/physx_rigid_gelpads.usd')


def transformed_mesh(prim,root,cache,scale):
    mesh=UsdGeom.Mesh(prim);points=np.asarray(mesh.GetPointsAttr().Get(),float)
    return (np.c_[points,np.ones(len(points))]@np.asarray(cache.ComputeRelativeTransform(prim,root)[0]))[:,:3]*scale


def raster(vertices,triangles,resolution=.001):
    lo=vertices[:,:2].min(0)-2*resolution;hi=vertices[:,:2].max(0)+2*resolution
    shape=np.ceil((hi-lo)/resolution).astype(int)+1
    field=np.full((shape[1],shape[0]),-np.inf)
    for t in vertices[triangles]:
        a=t[0];b=t[1]-a;c=t[2]-a;den=b[0]*c[1]-b[1]*c[0]
        if abs(den)<1e-15:continue
        lower=np.maximum(0,np.ceil((t[:,:2].min(0)-lo)/resolution).astype(int))
        upper=np.minimum(shape-1,np.floor((t[:,:2].max(0)-lo)/resolution).astype(int))
        if np.any(lower>upper):continue
        xx,yy=np.meshgrid(np.arange(lower[0],upper[0]+1),np.arange(lower[1],upper[1]+1))
        x=lo[0]+xx*resolution-a[0];y=lo[1]+yy*resolution-a[1]
        u=(x*c[1]-y*c[0])/den;v=(b[0]*y-b[1]*x)/den
        hit=(u>=-1e-9)&(v>=-1e-9)&(u+v<=1+1e-9)
        field[yy[hit],xx[hit]]=np.maximum(field[yy[hit],xx[hit]],(a[2]+u*b[2]+v*c[2])[hit])
    field[~np.isfinite(field)]=np.nan
    return field,lo,resolution


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    dest=OUT/'ASSET_GEOMETRY.npz'
    if dest.exists():raise RuntimeError('Do not overwrite frozen geometry')
    basket_path=ASSETS/'basket/basket.usd';obj_path=ASSETS/'alphabet_soup/alphabet_soup.usd'
    stage=Usd.Stage.Open(str(basket_path));cache=UsdGeom.XformCache();root=stage.GetDefaultPrim()
    meshes=[p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    if len(meshes)!=1:raise ValueError('Unexpected basket mesh layout')
    basket=transformed_mesh(meshes[0],root,cache,1.)
    mesh=UsdGeom.Mesh(meshes[0]);counts=np.asarray(mesh.GetFaceVertexCountsAttr().Get())
    if not np.all(counts==3):raise ValueError('Expected triangles')
    triangles=np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3)
    field,origin,res=raster(basket,triangles)
    objstage=Usd.Stage.Open(str(obj_path));objroot=objstage.GetDefaultPrim();objcache=UsdGeom.XformCache()
    colliders=[p for p in objstage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
    if len(colliders)!=1 or not colliders[0].IsA(UsdGeom.Cube):raise ValueError('Unexpected object collision geometry')
    cp=colliders[0];size=float(UsdGeom.Cube(cp).GetSizeAttr().Get());axis=np.linspace(-size/2,size/2,9)
    grid=np.array(list(itertools.product(axis,repeat=3)));grid=grid[np.any(np.isclose(np.abs(grid),size/2),axis=1)]
    obj=(np.c_[grid,np.ones(len(grid))]@np.asarray(objcache.ComputeRelativeTransform(cp,objroot)[0]))[:,:3]*.01
    rs=Usd.Stage.Open(str(ROBOT));rc=UsdGeom.XformCache();groups={}
    for p in Usd.PrimRange(rs.GetPseudoRoot(),Usd.TraverseInstanceProxies()):
        if p.HasAPI(UsdPhysics.CollisionAPI) and any(k in str(p.GetPath()) for k in ('hand','finger','case','gelpad')):
            root=p
            while root and not root.HasAPI(UsdPhysics.RigidBodyAPI):root=root.GetParent()
            if not root or not p.IsA(UsdGeom.Mesh):raise ValueError('Unsupported gripper collider')
            groups.setdefault(root.GetName(),[]).append(transformed_mesh(p,root,rc,1.))
    if len(groups)!=7:raise ValueError('Expected seven gripper bodies')
    names=sorted(groups);bounds=[]
    for name in names:
        points=np.concatenate(groups[name]);bounds.append(corners(points.min(0),points.max(0)))
    np.savez_compressed(dest,basket_vertices=basket,basket_triangles=triangles,
        height_field=field,height_origin=origin,height_resolution=res,
        object_collision_surface=obj,robot_names=np.array(names),robot_bounds=np.array(bounds))
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (basket_path,obj_path,ROBOT,Path(__file__),ROOT/'activeforcing_placement_contract.py')}
    info={'sources_sha256':sources,'geometry_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),
        'spawn_scale':{'basket':1.,'alphabet_soup':.01,'robot':1.},'extra_metersPerUnit_conversion':False,
        'basket_mesh_bounds':[basket.min(0).tolist(),basket.max(0).tolist()],
        'object_collision_bounds':[obj.min(0).tolist(),obj.max(0).tolist()],
        'object_collision_type':'authored Cube; 9x9 samples on each face, not visual cylinder assumption',
        'robot_body_names':names,'height_resolution_m':res,
        'containment_scope':'conservative vertical free-column geometry; no-hit is outside; not only outer AABB',
        'thresholds_selected_before_physics':True}
    (OUT/'ASSET_GEOMETRY_PROVENANCE.json').write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info,indent=2))


if __name__=='__main__':main()
