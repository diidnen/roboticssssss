"""Original depth-camera acquisition on exact physical contact surfaces.

Native visual and collision meshes disagree. This adds ONLY a sensor-visible
render proxy of the actual collision hulls, without adding physical shapes or
changing pi0's view. Raycast depth remains an independent geometric reference.
"""
import numpy as np
import sapien
from qualify_tactile_acquisition import Acquisition


class CollisionSurfaceRenderAcquisition(Acquisition):
    def __init__(self,env,arm):
        super().__init__(env,arm)
        # Native rt/camera.rgen writes outPosition on EVERY primary sample;
        # it therefore contains sample 31, not the pixel-center ray, at spp=32.
        # random.glsl uses Halton(31,2/3). This is shader-derived, not fitted to
        # surface errors. Validate its actual pixel coordinates on every frame.
        def halton(index,base):
            weight,value=1.,0.
            while index:
                weight/=base;value+=weight*(index%base);index//=base
            return value
        self.ray_pixel_offset=(halton(31,2),halton(31,3))
        target=env.deskbin.actor
        physical=target.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
        self.target_visuals=[component for component in target.components
                             if isinstance(component,sapien.render.RenderBodyComponent)]
        self.proxy=sapien.Entity();self.proxy.name='af_sensor_only_exact_contact_surface'
        self.render=sapien.render.RenderBodyComponent()
        material=sapien.render.RenderMaterial(base_color=[.5,.5,.5,1.])
        self.geometry_receipt=[]
        for shape in physical.collision_shapes:
            if not isinstance(shape,sapien.physx.PhysxCollisionShapeConvexMesh):
                raise TypeError('Unimplemented physical sensor surface: '+type(shape).__name__)
            vertices=np.asarray(shape.vertices,np.float32)*np.asarray(shape.scale,np.float32)
            triangles=np.asarray(shape.triangles,np.uint32).reshape(-1,3)
            face_normals=np.cross(vertices[triangles[:,1]]-vertices[triangles[:,0]],
                                  vertices[triangles[:,2]]-vertices[triangles[:,0]])
            normals=np.zeros_like(vertices)
            for i in range(3):np.add.at(normals,triangles[:,i],face_normals)
            normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
            visual=sapien.render.RenderShapeTriangleMesh(vertices,triangles,normals,
                         np.zeros((len(vertices),2),np.float32),material)
            visual.local_pose=shape.local_pose
            self.render.attach(visual)
            self.geometry_receipt.append({'vertices':len(vertices),'triangles':len(triangles),
                'scale':np.asarray(shape.scale).tolist(),
                'local_pose':np.r_[shape.local_pose.p,shape.local_pose.q].tolist()})
        self.proxy.add_component(self.render)
        env.scene.add_entity(self.proxy)
        self.render.visibility=0.

    def validate_pixel_centers(self,xyz):
        yy,xx=np.mgrid[:120,:160]
        z=xyz[...,2]
        valid=(xyz[...,3]<1)&(-z>=.024)&(-z<=.029)
        if not valid.any():return
        u=-xyz[...,0]/np.where(z!=0,z,1)*160+80
        v=xyz[...,1]/np.where(z!=0,z,1)*(120*20/18)+60
        error=max(float(abs((u-xx-self.ray_pixel_offset[0])[valid]).max()),
                  float(abs((v-yy-self.ray_pixel_offset[1])[valid]).max()))
        if error>.01:raise RuntimeError('RT pixel sampling differs from frozen shader/32 spp: '+str(error))

    def capture(self,output,tag):
        saved=[(component,component.visibility) for component in self.target_visuals]
        self.proxy.set_pose(self.env.deskbin.get_pose())
        try:
            for component,_ in saved:component.visibility=0.
            self.render.visibility=1.
            row=super().capture(output,tag)
            row['render_surface']='EXACT_NATIVE_COLLISION_HULLS_SENSOR_ONLY_VISUAL_PROXY'
            row['physics_shapes_added_or_changed']=False
            row['native_policy_visual_geometry_changed']=False
            row['surface_geometry']=self.geometry_receipt
            row['independent_raycast_pixel_offset']=list(self.ray_pixel_offset)
            row['pixel_offset_source']='native rt camera.rgen final spp sample, random.glsl Halton(31,2/3)'
        finally:
            self.render.visibility=0.
            for component,visibility in saved:component.visibility=visibility
            self.env.scene.update_render()
        import json
        (output/(tag+'.json')).write_text(json.dumps(row,indent=2))
        return row
