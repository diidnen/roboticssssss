import ast
import copy
import inspect
import json
import subprocess
from pathlib import Path
import unittest
import numpy as np
from activeforcing_placement_contract import (
    Geometry, CONFIG, PHASE_COUNTS, rotation, transform, evaluate_placement, adapt_placement_runner)
from activeforcing_placement_label_v2 import evaluate as evaluate_v2
import activeforcing_e2e_task0_smoke_20260905 as smoke

HISTORY=Path('/home/exouser/FORTE/analysis/results/current_contract_restore_and_matched_pilot_20260905/history_reconstruction')


class PlacementContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.geometry=Geometry(HISTORY/'placement_contract_repair/ASSET_GEOMETRY.npz')

    def frame(self):
        return {'object_pose_w':[0,0,.07,2**-.5,2**-.5,0,0],
            'basket_pose_w':[0,0,0,1,0,0,0],'object_velocity_w':[0]*6,
            'finger_object_contact_norms_N':[0,0]}

    def evaluate(self,frame=None,phases=None):
        return evaluate_placement([copy.deepcopy(frame or self.frame()) for _ in range(20)],
            PHASE_COUNTS if phases is None else phases,self.geometry)

    def test_contained_unheld_settled_positive(self):
        self.assertEqual(self.evaluate()['place_success'],1)

    def test_still_held_is_not_placed(self):
        f=self.frame();f['finger_object_contact_norms_N']=[1,1]
        self.assertEqual(self.evaluate(f)['place_success'],0)

    def test_moving_airborne_is_not_placed(self):
        f=self.frame();f['object_velocity_w'][2]=-.4
        self.assertEqual(self.evaluate(f)['place_success'],0)

    def test_above_rim_is_not_placed(self):
        f=self.frame();f['object_pose_w'][2]=.22
        self.assertEqual(self.evaluate(f)['place_success'],0)

    def test_outside_wall_is_not_placed(self):
        f=self.frame();f['object_pose_w'][0]=.12
        self.assertEqual(self.evaluate(f)['place_success'],0)

    def test_incomplete_phases_rejected(self):
        self.assertEqual(self.evaluate(phases={'transit':86})['place_success'],0)

    def test_basket_motion_cannot_use_stale_target(self):
        f=self.frame();f['basket_pose_w'][0]=.2
        self.assertEqual(self.evaluate(f)['place_success'],0)

    def test_rotation_translation_invariant(self):
        f=self.frame();q=np.array([2**-.5,0,0,2**-.5]);r=rotation(q);translation=np.array([.9,-.4,.1])
        obj=f['object_pose_w'];rotated_position=r@obj[:3]+translation
        # z90 composed with x90 is [.5,.5,.5,.5].
        f['object_pose_w']=list(rotated_position)+[.5,.5,.5,.5]
        f['basket_pose_w']=list(translation)+list(q)
        self.assertEqual(self.evaluate(f)['place_success'],1)

    def test_three_known_old_outside_cases_rejected(self):
        for band,force in [('high',3),('high',4),('low',5)]:
            p=HISTORY/f'placement_geometry_diagnostic/branches/{band}/F{force}_R0/FINAL_GEOMETRY.json'
            obj=json.loads(p.read_text())['rigid_objects']
            result=self.geometry.containment(obj['alphabet_soup_1']['root_pose_w'][0],obj['basket_1']['root_pose_w'][0])
            self.assertFalse(result['inside'],(band,force))

    def test_v2_also_rejects_all_known_old_outside_cases(self):
        for band,force in [('high',3),('high',4),('low',5)]:
            p=HISTORY/f'placement_geometry_diagnostic/branches/{band}/F{force}_R0/FINAL_GEOMETRY.json'
            obj=json.loads(p.read_text())['rigid_objects']
            frames=[{'episode_step':i,'object_pose_w':obj['alphabet_soup_1']['root_pose_w'][0],
                'basket_pose_w':obj['basket_1']['root_pose_w'][0],'object_velocity_w':[0]*6,
                'finger_object_contact_norms_N':[0,0]} for i in range(21)]
            self.assertEqual(evaluate_v2(frames,PHASE_COUNTS,self.geometry)['place_success'],0)

    def test_geometry_does_not_use_force_or_native_success(self):
        f=self.frame();f['native_success']=False;f['basket_contact_N']=0
        self.assertEqual(self.evaluate(f)['place_success'],1)
        f['object_pose_w'][0]=.12;f['native_success']=True;f['basket_contact_N']=20
        self.assertEqual(self.evaluate(f)['place_success'],0)

    def test_inner_force_execution_loop_unchanged(self):
        module=smoke.load_module('placement_unit_p5',smoke.P5_PATH)
        _,adapt=adapt_placement_runner(module)
        old=ast.parse(inspect.getsource(module.downstream_branch));new=ast.parse(adapt['source'])
        def loop(tree):
            matches=[n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='i']
            self.assertEqual(len(matches),1)
            return ast.dump(matches[0],include_attributes=False)
        self.assertEqual(loop(old),loop(new))
        self.assertEqual(sum(PHASE_COUNTS.values()),350)

    def test_host_isaac_ast_guard_parity_without_simulator(self):
        code=('import activeforcing_e2e_task0_smoke_20260905 as s; '
            'from activeforcing_placement_contract import adapt_placement_runner; '
            'm=s.load_module("cross_version_placement",s.P5_PATH); '
            'print(adapt_placement_runner(m)[1]["adapted_semantic_ast_sha256"])')
        host=subprocess.check_output([str(smoke.HOST_PY),'-c',code],cwd=smoke.ROOT,text=True).strip()
        isaac=subprocess.check_output([str(smoke.ISAAC_PY),'-c',code],cwd=smoke.ROOT,text=True).strip()
        self.assertEqual(host,isaac)


if __name__=='__main__':unittest.main()
