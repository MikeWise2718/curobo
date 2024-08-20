#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#


from torch.fx.experimental.symbolic_shapes import expect_true
import time
import copy
import carb
import numpy as np
from helper import add_robot_to_scene
from enum import Enum
########### OV #################
from omni.isaac.core import World
from omni.isaac.core.objects import cuboid, sphere
from omni.isaac.core.utils.types import ArticulationAction


# CuRobo
# from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
# from curobo.types.robot import JointState
from curobo.types.state import JointState
from curobo.util.logger import log_error
from curobo.util_file import load_yaml
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    PoseCostMetric,
)

from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

from pxr import Gf, Sdf, Usd, UsdGeom
from rotations import euler_angles_to_quat, matrix_to_euler_angles, rot_matrix_to_quat, gf_rotation_to_np_array
from omni.isaac.core.utils.stage import get_current_stage
from senut import apply_material_to_prim_and_children, apply_matdict_to_prim_and_children, build_material_dict, get_link_paths

############################################################
import torch


# ########## OV #################;;;;;
import typing

from mgrut import get_args, get_vek, print_mat, list4to_quatd, quatd_to_list4, get_sphere_entry

args = get_args()


class RocuTranMan:
    def __init__(self, robid, usealt=True):
        self.rob_pos = Gf.Vec3d(0, 0, 0)
        self.rob_ori_quat = Gf.Quatd(1, 0, 0, 0)
        self.rob_ori_euler = Gf.Vec3d(0, 0, 0)
        self.rob_ori_sel = "0,0,0"
        # self.memstage: Usd.Stage = Usd.Stage.CreateInMemory()
        self.stage: Usd.Stage = get_current_stage()
        # self.default_prim: Usd.Prim = UsdGeom.Xform.Define(self.memstage, Sdf.Path("/World")).GetPrim()
        # self.memstage.SetDefaultPrim(self.default_prim)
        # self.xformpath = self.default_prim.GetPath().AppendPath(f"Xform_{robid}")

        self.xformw_full_prim: Usd.Prim = UsdGeom.Xform.Define(self.stage, f"/World/XformFull_{robid}")
        self.xformw_robot_proxy_prim: Usd.Prim = UsdGeom.Xform.Define(self.stage, f"/World/XformRobProxy_{robid}")

        self.xform_full_pre_rot_op = self.xformw_full_prim.AddRotateXYZOp(opSuffix='prerot')
        self.xform_full_tran_op = self.xformw_full_prim.AddTranslateOp()
        self.xform_full_rot_op = self.xformw_full_prim.AddRotateXYZOp()

        self.xform_robopt_proxy_tran_op = self.xformw_robot_proxy_prim.AddTranslateOp()
        self.xform_robopt_proxy_orient_op = self.xformw_robot_proxy_prim.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
        self.xform_robopt_proxy_scale_op = self.xformw_robot_proxy_prim.AddScaleOp()

        self.usealt = usealt
        print("RobDeco created usealt:", usealt)

    def get_world_xform(self, prim: Usd.Prim, dump=False) -> typing.Tuple[Gf.Vec3d, Gf.Rotation, Gf.Vec3d, Gf.Matrix3d]:
        xform = UsdGeom.Xformable(prim)
        time = Usd.TimeCode.Default()  # The time at which we compute the bounding box
        world_transform: Gf.Matrix4d = xform.ComputeLocalToWorldTransform(time)
        if dump:
            print("world_transform:", world_transform)
        translation: Gf.Vec3d = world_transform.ExtractTranslation()
        rotation: Gf.Rotation = world_transform.ExtractRotation()
        scale: Gf.Vec3d = Gf.Vec3d(*(v.GetLength() for v in world_transform.ExtractRotationMatrix()))
        return translation, rotation, scale, world_transform

    def find_rcc_to_wc_tansform(self, prerot_euler, pos, ori_euler):
        self.xform_full_pre_rot_op.Set(value=prerot_euler)
        self.xform_full_tran_op.Set(value=pos)
        self.xform_full_rot_op.Set(value=ori_euler)
        t, r, s, m = self.get_world_xform(self.xformw_full_prim)
        return t, r, s, m

    def set_robot_proxy_tran(self, tranvek, oriquat):
        self.xform_robopt_proxy_tran_op.Set(tranvek)
        qlist = oriquat.tolist()
        q = Gf.Quatd(qlist[0], qlist[1:])
        self.xform_robopt_proxy_orient_op.Set(value=q)
        self.xform_robopt_proxy_scale_op.Set(Gf.Vec3d(1, 1, 1))
        t, r, s, m = self.get_world_xform(self.xformw_robot_proxy_prim, dump=False)
        return t, r, s, m

    def set_transform(self, prerot, pos, ori):

        self.rob_prerot_euler = self.to_gfvec(prerot)
        self.prerot = prerot
        self.rob_pos = Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2]))
        self.rob_ori_euler = self.to_gfvec(ori)
        self.rob_ori_quat_nparray = euler_angles_to_quat(self.rob_ori_euler, degrees=True)
        self.rob_ori_quat = list4to_quatd(self.rob_ori_quat_nparray)

        (t, r, _, w) = self.find_rcc_to_wc_tansform(self.rob_prerot_euler, self.rob_pos, self.rob_ori_euler)
        (self.tran, self.rotmat3d_gfrot, self.rcc_to_wc_transform) = (t, r, w)
        self.rotmat3d = Gf.Matrix3d(self.rotmat3d_gfrot)
        self.inv_rotmat3d = self.rotmat3d.GetTranspose()

        self.rotmat3d_eulers = matrix_to_euler_angles(self.rotmat3d, degrees=True)
        self.rotmat3d_quat_nparray = euler_angles_to_quat(self.rotmat3d_eulers, degrees=True)
        npmat3x3 = self.to_npmat3x3(self.inv_rotmat3d)
        self.rotmat3d_quat_nparray_inv = rot_matrix_to_quat(npmat3x3)
        self.rotmat3d_eulers_inv = matrix_to_euler_angles(npmat3x3, degrees=True)
        # (t, r, _, w) = self.set_robot_proxy_tran(self.tran, self.rotmat3d_quat_nparray)
        (t, r, _, w) = self.set_robot_proxy_tran(self.rob_pos, self.rotmat3d_quat_nparray)
        (self.robproxy_tran, self.robproxy_rot, self.robproxy_world_tran) = (t, r, w)

        print("----- input values -----")
        print("rob_pos:", self.rob_pos)
        print("rob_ori_euler:", self.rob_ori_euler)
        print("rob_prerot_euler:", self.rob_prerot_euler)
        print("rob_ori_quat:", self.rob_ori_quat)
        print("----- cacluated values -----")
        print_mat("rcc_to_wc_transform:", 4, 4, self.rcc_to_wc_transform)
        print_mat("rotmat3d", 3, 3, self.rotmat3d)
        print_mat("inv_rotmat3d", 3, 3, self.inv_rotmat3d)
        print("rob_ori_sel:", self.rob_ori_sel)
        print("rotmat3d_eulers    :", self.rotmat3d_eulers)
        print("rotmat3d_eulers_inv:", self.rotmat3d_eulers_inv)
        print("tran:", self.tran)
        print("rob_pos:", self.rob_pos)
        print("rotmat3d_quat_nparray    :", self.rotmat3d_quat_nparray)
        print("rotmat3d_quat_nparray_inv:", self.rotmat3d_quat_nparray_inv)
        print_mat("robproxy_world_tran:", 4, 4, self.robproxy_world_tran)

    def get_robot_base(self):
        return self.rob_pos, self.rotmat3d_quat_nparray

    def to_gfvec(self, vek):
        x = float(vek[0])
        y = float(vek[1])
        z = float(vek[2])
        return Gf.Vec3d(x, y, z)

    def to_npmat3x3(self, gfmat):
        npmat3x3 = np.zeros((3, 3), dtype=np.float32)
        for i in range(3):
            for j in range(3):
                npmat3x3[i, j] = gfmat[i, j]
        return npmat3x3

    def quat_apply(self, q1, q2):
        q1inv = q1.GetInverse()
        q2q = Gf.Quatd(float(q2[0]), float(q2[1]), float(q2[2]), float(q2[3]))
        rv = q1inv * q2q * q1
        return rv

    def rcc_to_wc(self, pos, ori):
        pos_gfv = self.to_gfvec(pos)
        # pos_new = self.tran + pos_gfv * self.inv_rotmat3d
        pos_new = self.rob_pos + pos_gfv * self.inv_rotmat3d
        return pos_new, ori

    def wc_to_rcc(self, pos, ori):
        pos_gfv = self.to_gfvec(pos)
        # pos_new = (pos_gfv - self.tran) * self.rotmat3d
        pos_new = (pos_gfv - self.rob_pos) * self.rotmat3d
        return pos_new, ori

    def dump_robot_transforms(self, robpathname):
        robpath = Sdf.Path(robpathname)
        robprim = self.stage.GetPrimAtPath(robpath)
        (t, r, s, m) = self.get_world_xform(robprim)
        print("world_tranlate:", t)
        print("world_rotate:", r)
        print_mat("world_transform:", 4, 4, m)
        print_mat("rotmat3d:", 3, 3, self.rotmat3d)
        print_mat("inv_rotmat3d:", 3, 3, self.inv_rotmat3d)


def dst(p1, p2):
    rv = np.linalg.norm(p1 - p2)
    return rv

class RocuMoveMode(Enum):
    FollowTargetWithMoGen = 1
    FollowTargetWithInvKin = 2
    ReachabilityWithInvKin = 3

class RocuConfiguator:
    def __init__(self):
        self.robot_config_path = "robot"
        self.external_asset_path = "assetpath"
        self.external_robot_configs_path = "config"
        self.my_world = None
        self.world_cfg = None
        self.matman = None

RocuConfig = RocuConfiguator()

class RocuWrapper:

    def __init__(self, robid):
        self.robid = robid
        self.Initialize()

    def Initialize(self):

        self.rocuTranman = RocuTranMan(self.robid)
        self.move_mode = None
        self.robot_config_path = RocuConfig.robot_config_path
        self.external_asset_path = RocuConfig.external_asset_path
        self.external_robot_configs_path = RocuConfig.external_robot_configs_path

        self.my_world = RocuConfig.my_world
        self.matman = RocuConfig.matman

        self.robot_cfg = None
        self.robot_prim_path = None
        self.robot = None
        self.articulation_controller = None
        self.motion_gen_config = None
        self.motion_gen = None
        self.plan_config = None
        self.max_attempts = 0
        self.tensor_args = TensorDeviceType()
        self.world_cfg = None
        self.stage = self.my_world.stage
        self.cu_js = None

        self.spheres = None
        self.spherenames = None
        self.spheres_visible = False

        self.past_pose = None
        self.past_orientation = None
        self.target_pose = None
        self.target_orientation = None
        self.trigger = False

        self.vizi_spheres = False
        self.reactive = None
        self.reach_partial_pose = None
        self.hold_partial_pose = None
        self.constrain_grasp_approach = None

        self.cur_cmd_plan = None
        self.pose_metric = None
        self.num_targets = 0

        self.cmd_plan_queue = []

        self.ik_result = None

        self.upper_dof_lim = None
        self.lower_dof_lim = None

        # robot_cfg = load_yaml(robot_cfg_path)["robot_cfg"]
        self.LoadRobotCfg(self.robot_config_path)

        if self.external_asset_path is not None:
            self.robot_cfg["kinematics"]["external_asset_path"] = self.external_asset_path
        if self.external_robot_configs_path is not None:
            self.robot_cfg["kinematics"]["external_robot_configs_path"] = self.external_robot_configs_path
        self.j_names = self.robot_cfg["kinematics"]["cspace"]["joint_names"]
        self.default_config = self.robot_cfg["kinematics"]["cspace"]["retract_config"]

    def SetRobotMoveMode(self, mode: RocuMoveMode, reactive, reach_pp, hold_pp, con_grasp, n_obstacle_cuboids, n_obstacle_mesh):
        self.move_mode = mode
        match mode:
            case RocuMoveMode.FollowTargetWithMoGen:
                self.InitMotionGen(n_obstacle_cuboids, n_obstacle_mesh)
                self.SetMoGenOptions(reactive=reactive,
                                     reach_partial_pose=hold_pp,
                                     hold_partial_pose=reach_pp,
                                     constrain_grasp_approach=con_grasp
                                     )
                self.Warmup()
                self.SetupMoGenPlanConfig()
                self.CreateTarget()
            case RocuMoveMode.FollowTargetWithInvKin:
                self.InitInvKin(n_obstacle_cuboids, n_obstacle_mesh)
                self.SetMoGenOptions(reactive=reactive,
                                     reach_partial_pose=hold_pp,
                                     hold_partial_pose=reach_pp,
                                     constrain_grasp_approach=con_grasp
                                     )
                self.Warmup()
                self.CreateTarget()
            case _:
                carb.log_warn(f"Move Mode {mode} not implemented yet.")

    def wc_to_rcc(self, pos, ori):
        return self.rocuTranman.wc_to_rcc(pos, ori)

    def rcc_to_wc(self, pos, ori):
        return self.rocuTranman.rcc_to_wc(pos, ori)

    def dump_robot_transforms(self):
        robpathname = self.robot_prim_path
        self.rocuTranman.dump_robot_transforms(robpathname)

    def get_start_pose(self):
        """Returns the start pose of the robot."""
        match self.move_mode:
            case RocuMoveMode.FollowTargetWithMoGen:
                rollout_fn = self.motion_gen.rollout_fn
            case RocuMoveMode.FollowTargetWithInvKin:
                rollout_fn = self.ik_solver.solver.rollout_fn

        start_state = JointState.from_position(
            rollout_fn.dynamics_model.retract_config.view(1, -1).clone(),
            joint_names=rollout_fn.joint_names,
        )
        state = rollout_fn.compute_kinematics(start_state)
        sp = state.ee_pos_seq.cpu()[0]
        sq = state.ee_quat_seq.cpu()[0]
        return sp, sq
        # return self.motion_gen.get_start_pose()

    def get_cur_pose(self):
        cur_pose = self.cu_js.position.view(1,-1)
        # return self.motion_gen.get_cur_pose(joint_state)
        return cur_pose

    def get_cur_pose_old(self, joint_state):
        cur_pose =  self.motion_gen.get_cur_pose(joint_state)
        return cur_pose


    def update_world(self, obstacles):
        match self.move_mode:
            case RocuMoveMode.FollowTargetWithMoGen:
                rv = self.motion_gen.update_world(obstacles)
            case RocuMoveMode.FollowTargetWithInvKin:
                rv = self.ik_solver.update_world(obstacles)
            case _:
                carb.log_warn(f"Move Mode {self.move_mode} not implemented yet.")
                rv = None
        return rv

    def LoadAndPositionRobot(self, prerot, pos, ori, subroot=""):
        self.rocuTranman.set_transform(prerot=prerot, pos=pos, ori=ori)
        rp, ro = self.rocuTranman.get_robot_base()
        self.robot_name = f"robot_{self.robid}"
        self.robot, self.robot_prim_path = add_robot_to_scene(self.robot_cfg, self.my_world, position=rp, orient=ro, subroot=subroot, robot_name=self.robot_name)
        self.articulation_controller = self.robot.get_articulation_controller()
        self.init_alarm_skin()

    def SetMoGenOptions(self, reactive=None, reach_partial_pose=None, hold_partial_pose=None,
                        constrain_grasp_approach=None, vizi_spheres=None):
        if reactive is not None:
            self.reactive = reactive
        if reach_partial_pose is not None:
            self.reach_partial_pose = reach_partial_pose
        if hold_partial_pose is not None:
            self.hold_partial_pose = hold_partial_pose
        if constrain_grasp_approach is not None:
            self.constrain_grasp_approach = constrain_grasp_approach
        if vizi_spheres is not None:
            self.vizi_spheres = vizi_spheres

    def LoadRobotCfg(self, robot_pathname):
        self.robot_cfg = load_yaml(robot_pathname)["robot_cfg"]

    def InitMotionGen(self, n_obstacle_cuboids, n_obstacle_mesh):
        trajopt_dt = None
        optimize_dt = True
        trajopt_tsteps = 16
        trim_steps = None
        self.max_attempts = 4
        self.world_cfg = RocuConfig.world_cfg
        interpolation_dt = 0.05
        if self.reactive:
            trajopt_tsteps = 40
            trajopt_dt = 0.04
            optimize_dt = False
            self.max_attempts = 1
            trim_steps = [1, None]
            interpolation_dt = trajopt_dt
        self.motion_gen_config = MotionGenConfig.load_from_robot_config(
            self.robot_cfg,
            self.world_cfg,
            self.tensor_args,
            collision_checker_type=CollisionCheckerType.MESH,
            num_trajopt_seeds=12,
            num_graph_seeds=12,
            interpolation_dt=interpolation_dt,
            collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
            optimize_dt=optimize_dt,
            trajopt_dt=trajopt_dt,
            trajopt_tsteps=trajopt_tsteps,
            trim_steps=trim_steps,
        )
        self.motion_gen = MotionGen(self.motion_gen_config)

       #  self.robot._articulation_view.initialize() # don't do this - causes an exceptino - can't create phyics sim  view

    def InitInvKin(self, n_obstacle_cuboids, n_obstacle_mesh):
        self.world_cfg = RocuConfig.world_cfg
        self.ik_config = IKSolverConfig.load_from_robot_config(
            self.robot_cfg,
            self.world_cfg,
            rotation_threshold=0.05,
            position_threshold=0.005,
            num_seeds=20,
            self_collision_check=True,
            self_collision_opt=True,
            tensor_args=self.tensor_args,
            use_cuda_graph=True,
            collision_checker_type=CollisionCheckerType.MESH,
            collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
            # use_fixed_samples=True,
        )
        self.ik_solver = IKSolver(self.ik_config)

    def StartStep(self, step_index):
        if step_index == 1:
            self.Reset()

        if self.circle_target:
            self.UpdateCirclingTarget()

    def EndStep(self)->bool:
        requestPause = False
        match self.move_mode:
            case RocuMoveMode.FollowTargetWithMoGen:
                self.UpdateJointState()
                self.realize_joint_alarms()
                self.ProcessCollisionSpheres()
                self.HandleTargetProcessing()
                requestPause = self.ExecuteMoGenCmdPlan()
            case RocuMoveMode.FollowTargetWithInvKin:
                self.UpdateJointState()
                self.realize_joint_alarms()
                self.ProcessCollisionSpheres()
                self.HandleTargetProcessing()
                requestPause = self.ExecuteInvKinCmdPlan()
            case _:
                carb.log_warn(f"Move Mode {self.move_mode} not implemented yet.")

        return requestPause

    def Reset(self):
        self.idx_list = [self.robot.get_dof_index(x) for x in self.j_names]
        self.robot.set_joint_positions(self.default_config, self.idx_list)

        self.robot._articulation_view.set_max_efforts(
            values=np.array([5000 for i in range(len(self.idx_list))]), joint_indices=self.idx_list
        )

    def Warmup(self):
        match self.move_mode:
            case RocuMoveMode.FollowTargetWithMoGen:
                start_warmup = time.time()
                print("Warming up MoGen...")
                self.motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False, parallel_finetune=True)
                elap = time.time() - start_warmup
                print(f"Curobo MoGen is Ready and Warmed-up - took:{elap:.2f} secs")
            case RocuMoveMode.FollowTargetWithInvKin:
                print("Warming up InvKin...")
                start_warmup = time.time()
                elap = time.time() - start_warmup
                print(f"Curobo InvKin is Ready and Warmed-up - took:{elap:.2f} secs")
                pass
            case _:
                carb.log_warn(f"Move Mode {self.move_mode} not implemented yet.")

    def CreateTarget(self, target_pos=None, target_ori=None):

        if target_pos is None or target_ori is None:
            sp_rcc, sq_rcc = self.get_start_pose()
            sp_wc, sq_wc = self.rocuTranman.rcc_to_wc(sp_rcc, sq_rcc)
            if type(sq_wc) is Gf.Quatd:
                sq_wc = quatd_to_list4(sq_wc)

        if target_pos is None:
            target_pos = sp_wc

        if target_ori is None:
            target_ori = sq_wc

        self.target = cuboid.VisualCuboid(
            f"/World/target_{self.robid}",
            position=target_pos,
            orientation=target_ori,
            color=np.array([1.0, 0, 0]),
            size=0.05,
        )
        self.cube_position = target_pos
        self.cube_orientation = target_ori

        self.curcen, self.curori = self.target.get_world_pose()
        self.curvel = 0.02
        self.curang = 0
        self.currad = 0.1
        self.circle_target = False

        return self.target

    def UpdateCirclingTarget(self):
        self.curang += self.curvel
        newpos = np.zeros(3)
        newpos[0] = self.curcen[0] + self.currad * np.cos(self.curang)
        newpos[1] = self.curcen[1] + self.currad * np.sin(self.curang)
        newpos[2] = self.curcen[2]
        self.target.set_world_pose(
            position=newpos,
            orientation=self.curori
        )

    def InitCirclingTargetValues(self):
        self.curcen, self.curori = self.target.get_world_pose()
        self.curvel = 0.02
        self.curang = 0
        self.currad = 0.1

    def ToggleCirclingTarget(self):
        self.circle_target = not self.circle_target
        if self.circle_target:
            self.InitCirclingTargetValues()

    def SetupMoGenPlanConfig(self):
        self.plan_config = MotionGenPlanConfig(
            enable_graph=False,
            enable_graph_attempt=2,
            max_attempts=self.max_attempts,
            enable_finetune_trajopt=True,
            parallel_finetune=True,
        )

    def UpdateJointState(self):
        self.sim_js = self.robot.get_joints_state()
        self.sim_js_names = self.robot.dof_names
        if self.upper_dof_lim is None:
            self.upper_dof_lim = self.robot.dof_properties["upper"]
            self.lower_dof_lim = self.robot.dof_properties["lower"]
        # rcfg.dof_paths = art._prim_view._dof_paths[0] # why is this a list while the following ones are not?
        # rcfg.dof_types = art._prim_view._dof_types
        # rcfg.dof_names = art._prim_view._dof_names
        # rcfg.dof_properties = art._prim_view._dof_properties
            dofpaths = self.robot._articulation_view._dof_paths[0]
            self.link_paths = get_link_paths(dofpaths)


        if np.any(np.isnan(self.sim_js.positions)):
            print("isaac sim has returned NAN joint position values.")
            log_error("isaac sim has returned NAN joint position values.")
        self.cu_js = JointState(
            position=self.tensor_args.to_device(self.sim_js.positions),
            velocity=self.tensor_args.to_device(self.sim_js.velocities),  # * 0.0,
            acceleration=self.tensor_args.to_device(self.sim_js.velocities) * 0.0,
            jerk=self.tensor_args.to_device(self.sim_js.velocities) * 0.0,
            joint_names=self.sim_js_names,
        )

        if not self.reactive:
            self.cu_js.velocity *= 0.0
            self.cu_js.acceleration *= 0.0

        if self.reactive and self.past_cmd is not None:
            self.cu_js.position[:] = self.past_cmd.position
            self.cu_js.velocity[:] = self.past_cmd.velocity
            self.cu_js.acceleration[:] = self.past_cmd.acceleration
        # self.cu_js = self.cu_js.get_ordered_joint_state(self.motion_gen.kinematics.joint_names)
        self.cu_js = self.cu_js.get_ordered_joint_state(self.j_names)

    def CreateCollisionSpheres(self):
        # get a fresh list of spheres:
        self.sph_list = self.motion_gen.kinematics.get_robot_as_spheres(self.cu_js.position)
        if self.spheres is None:
            config_spheres = self.robot_cfg["kinematics"]["collision_spheres"]
            self.spheres = []
            self.spherenames = []
            # create spheres:
            ncreated = 0
            for sidx, sph in enumerate(self.sph_list[0]):
                sphentry = get_sphere_entry(config_spheres, sidx)
                sname = "/curobo/robot_sphere_" + str(sidx)
                clr = np.array([0, 0.8, 0.2])
                if sphentry is not None:
                    keyname = sphentry["keyname"]
                    keyidx = sphentry["keyidx"]
                    linkname = f"/curobo/{keyname}"
                    if not self.stage.GetPrimAtPath(linkname):
                        _ = self.stage.DefinePrim(linkname, "Xform")
                    sname = f"{linkname}/sphere_{keyidx}"
                    if "scolor" in sphentry:
                        clr = np.array(sphentry["scolor"])
                s_ori = np.array([1, 0, 0, 0])
                sp_wc, _ = self.rocuTranman.rcc_to_wc(sph.position, s_ori)
                sp = sphere.VisualSphere(
                    prim_path=sname,
                    position=np.ravel(sp_wc),
                    radius=float(sph.radius),
                    color=clr,
                )
                self.spheres.append(sp)
                self.spherenames.append(sname)
                ncreated += 1
            print(f"Created {ncreated} Spheres")

    def UpdateCollistionSpherePositions(self):
        self.sph_list = self.motion_gen.kinematics.get_robot_as_spheres(self.cu_js.position)
        s_ori = np.array([1, 0, 0, 0])
        for sidx, sph in enumerate(self.sph_list[0]):
            if not np.isnan(sph.position[0]):
                sp_wc, _ = self.rocuTranman.rcc_to_wc(sph.position, s_ori)
                self.spheres[sidx].set_world_pose(position=np.ravel(sp_wc))
                self.spheres[sidx].set_radius(float(sph.radius))

    def DeleteCollisionSpheres(self):
        if self.spherenames is not None:
            for sn in self.spherenames:
                self.stage.RemovePrim(sn)
        self.spheres = None
        self.spherenames = None
        self.spheres_visible = False

    def ProcessCollisionSpheres(self):
        if self.vizi_spheres:
            if self.spheres is None:
                self.CreateCollisionSpheres()
            else:
                self.UpdateCollistionSpherePositions()
            self.spheres_visible = True

        if not self.vizi_spheres and self.spheres_visible:
            self.DeleteCollisionSpheres()

    def HandleTargetProcessing(self):

        self.cube_position, self.cube_orientation = self.target.get_world_pose()
        match self.move_mode:
            case RocuMoveMode.FollowTargetWithMoGen:
                triggerMoGen = self.CalcMoGenTargetTrigger()
                if triggerMoGen:
                    print("Triggering MoGen")

                    self.DoMoGenToTarget()
            case RocuMoveMode.FollowTargetWithInvKin:
                triggerInvKin = self.CalcMoGenTargetTrigger()
                if triggerInvKin:
                    print("Triggering InvKin")

                    self.DoInvKinToTarget()

        self.past_pose = self.cube_position
        self.past_orientation = self.cube_orientation

    def GetTargetPose(self):
        return self.cube_position, self.cube_orientation

    def SetTargetPose(self, pos, ori):
        if type(ori) is Gf.Quatd:
            ori = quatd_to_list4(ori)
        self.target.set_world_pose(position=pos, orientation=ori)
        self.cube_position = pos
        self.cube_orientation = ori
        return

    def CalcMoGenTargetTrigger(self):
        if self.past_pose is None:
            self.past_pose = self.cube_position
        if self.target_pose is None:
            self.target_pose = self.cube_position
        if self.target_orientation is None:
            self.target_orientation = self.cube_orientation
        if self.past_orientation is None:
            self.past_orientation = self.cube_orientation

        self.static_robo = True
        # robot_static = True
        # velmag = np.max(np.abs(self.sim_js.velocities))
        # if (velmag < 0.2) or reactive:
        #     static_robo = True
        #     pass
        cube_pos, cube_ori = self.cube_position, self.cube_orientation
        pretrig = (dst(cube_pos, self.target_pose) > 1e-3 or
                   dst(cube_ori, self.target_orientation) > 1e-3)
        self.trigger = (pretrig and
                        dst(self.past_pose, cube_pos) == 0.0 and
                        dst(self.past_orientation, cube_ori) == 0.0 and
                        self.static_robo)
        # print(f"trigger:{trigger} pretrig:{pretrig} velmag:{velmag:.2f} static_robo:{static_robo}")
        # print("pretrig:", pretrig, "  trigger:", self.trigger, "  static_robo:", self.static_robo)

        if self.circle_target:
            self.trigger = self.cur_cmd_plan is None

        return self.trigger

    def ApplyAction(self, art_action):
        self.articulation_controller.apply_action(art_action)

    def DoMoGenToTarget(self):
        rv = self.DoMoGenToPosOri(self.cube_position, self.cube_orientation)
        return rv

    def DoMoGenToPosOri(self, pos, ori):
        # Set EE teleop goals, use cube for simple non-vr init:
        ee_pos_rcc, ee_ori_rcc = self.rocuTranman.wc_to_rcc(pos, ori)
        if type(ee_ori_rcc) is Gf.Quatd:
            ee_ori_rcc = quatd_to_list4(ee_ori_rcc)

        print("ik_goal-p:", ee_pos_rcc, " q:", ee_ori_rcc)
        # compute curobo solution:
        ik_goal = Pose(
            # position=tensor_args.to_device(ee_translation_goal),
            # quaternion=tensor_args.to_device(ee_orientation_teleop_goal),
            position=self.tensor_args.to_device(ee_pos_rcc),
            quaternion=self.tensor_args.to_device(ee_ori_rcc),
        )
        self.plan_config.pose_cost_metric = self.pose_metric
        try:
            result = self.motion_gen.plan_single(self.cu_js.unsqueeze(0), ik_goal, self.plan_config)
        except Exception as e:
            print(f"Exception in motion_gen.plan_single e:{e}")
            result = self.motion_gen.plan_single(self.cu_js.unsqueeze(0), ik_goal, self.plan_config)
            return

        print("motion_gen.plan_single success:", result.success)
        # ik_result = ik_solver.solve_single(ik_goal, cu_js.position.view(1,-1), cu_js.position.view(1,1,-1))

        successfull = result.success.item()  # ik_result.success.item()
        if self.num_targets == 1:
            if self.constrain_grasp_approach:
                # print("2: Creating grasp approach metric - cga --------- ")
                self.pose_metric = PoseCostMetric.create_grasp_approach_metric()
            if self.reach_partial_pose is not None:
                # print("2: Creating grasp approach metric - rpp --------- ")
                reach_vec = self.motion_gen.tensor_args.to_device(self.reach_partial_pose)
                self.pose_metric = PoseCostMetric(
                    reach_partial_pose=True, reach_vec_weight=reach_vec
                )
            if self.hold_partial_pose is not None:
                # print("2: Creating grasp approach metric - hpp --------- ")
                hold_vec = self.motion_gen.tensor_args.to_device(self.hold_partial_pose)
                self.pose_metric = PoseCostMetric(hold_partial_pose=True, hold_vec_weight=hold_vec)
        if successfull:
            self.num_targets += 1
            cmd_plan = result.get_interpolated_plan()
            cmd_plan = self.motion_gen.get_full_js(cmd_plan)
            print(f"Plan Success with {len(cmd_plan.position)} steps")
            # get only joint names that are in both:
            idx_list = []
            common_js_names = []
            for x in self.sim_js_names:
                if x in cmd_plan.joint_names:
                    idx_list.append(self.robot.get_dof_index(x))
                    common_js_names.append(x)
            # idx_list = [robot.get_dof_index(x) for x in sim_js_names]

            cmd_plan = cmd_plan.get_ordered_joint_state(common_js_names)
            self.QueCmdPlan(cmd_plan)

            self.cmd_idx = 0

        else:
            msg = f"Plan did not converge to a solution. Status:{result.status}. No action is being taken."
            print(msg)
            carb.log_warn(msg)

        self.target_pose = self.cube_position
        self.target_orientation = self.cube_orientation

    def DoInvKinToTarget(self):
        rv = self.DoInvKinToPosOri(self.cube_position, self.cube_orientation)
        return rv

    def DoInvKinToPosOri(self, pos, ori):

        ee_pos_rcc, ee_ori_rcc = self.rocuTranman.wc_to_rcc(pos, ori)
        if type(ee_ori_rcc) is Gf.Quatd:
            ee_ori_rcc = quatd_to_list4(ee_ori_rcc)

        # compute curobo solution:
        print("ik_goal-p:", ee_pos_rcc, " q:", ee_ori_rcc)
        ik_goal = Pose(
            position=self.tensor_args.to_device(ee_pos_rcc),
            quaternion=self.tensor_args.to_device(ee_ori_rcc),
        )
        st_time = time.time()
        # ik_result = self.ik_solver.solve_single(ik_goal)
        ik_result = self.ik_solver.solve_single(ik_goal, self.cu_js.position.view(1,-1), self.cu_js.position.view(1,1,-1))
        total_time = (time.time() - st_time)
        print(
            "Success, Solve Time(s), Total Time(s)",
            torch.count_nonzero(ik_result.success).item(),
            ik_result.solve_time,
            total_time,
            1.0 / total_time,
            torch.mean(ik_result.position_error) * 100.0,
            torch.mean(ik_result.rotation_error) * 100.0,
        )
        self.ik_result = ik_result

    def QueCmdPlan(self, cmd_plan):
        # should do a plausiblity check on cmd_plan
        self.cmd_plan_queue.append(cmd_plan)

    def AssignCurCmdPlan(self):
        if self.cur_cmd_plan is None:
            if len(self.cmd_plan_queue) > 0:
                len_bef = len(self.cmd_plan_queue)
                self.cur_cmd_plan = self.cmd_plan_queue.pop(0)
                len_aft = len(self.cmd_plan_queue)
                print(f"AssignCurCmdPlan len_bef:{len_bef} len_aft:{len_aft}")

    def ExecuteMoGenCmdPlan(self):
        requestPause = False
        self.AssignCurCmdPlan()
        if self.cur_cmd_plan is not None:
            print(f"Executing plan step {self.cmd_idx}/{len(self.cur_cmd_plan.position)}")
            cmd_state = self.cur_cmd_plan[self.cmd_idx]
            self.past_cmd = cmd_state.clone()
            # get full dof state
            art_action = ArticulationAction(
                cmd_state.position.cpu().numpy(),
                cmd_state.velocity.cpu().numpy(),
                joint_indices=self.idx_list,
            )
            # set desired joint angles obtained from IK:
            # print(f"Applying action: {art_action}")
            #  articulation_controller.apply_action(art_action)
            self.ApplyAction(art_action)
            print("Applied Action - ArtAction:", art_action)
            self.cmd_idx += 1
            for _ in range(2):
                self.my_world.step(render=False)
            requestPause = True
            if self.cmd_idx >= len(self.cur_cmd_plan.position):
                self.cmd_idx = 0
                self.cur_cmd_plan = None
                self.past_cmd = None
            return requestPause

    def ExecuteInvKinCmdPlan(self):
        requestPause = False
        if self.ik_result is not None:
            if torch.count_nonzero(self.ik_result.success) > 0:
                joint_positions = self.ik_result.js_solution.position[0]
                jp_numpy = joint_positions.cpu().numpy()
                art_action = ArticulationAction(
                    jp_numpy,
                    None,
                    joint_indices=self.idx_list,
                )
                self.ApplyAction(art_action)
                print("Applied Action - ArtAction:", art_action)
                # for _ in range(2):
                #     self.my_world.step(render=False)
                requestPause = True
            self.ik_result = None
            return requestPause

    def init_alarm_skin(self):
        self.robmatskin = "default"
        self.alarmskin = "Red_Glass"
        self.show_joints_close_to_limits = False
        # self.realize_joint_alarms(force=True)
        self.ensure_orimat()

    def check_alarm_status(self):
        self.jchk_str = ""
        self.jchk_str_val = ""
        for j,jn in enumerate(self.sim_js_names):
            jpos = self.sim_js.positions[j]
            llim = self.lower_dof_lim[j]
            ulim = self.upper_dof_lim[j]
            denom = ulim - llim
            if denom == 0:
                denom = 1
            pct = 100*(jpos - llim)/denom
            self.jchk_str_val += f"{pct:.0f} "
            if pct < 10:
                self.jchk_str += "L"
            elif pct > 90:
                self.jchk_str += "U"
            else:
                self.jchk_str += "."
        print("joint_check:", self.jchk_str, " ", self.jchk_str_val)

    def change_material(self, matname):
        apply_material_to_prim_and_children(self.stage, self.matman, matname, self.robot_prim_path)

    def ensure_orimat(self):
        if not hasattr(self, "orimat"):
            self.orimat = build_material_dict(self.stage, self.robot_prim_path)

    def toggle_material(self):
        if self.robmatskin == "default":
            self.robmatskin = "Red_Glass"
            apply_material_to_prim_and_children(self.stage, self.matman, self.robmatskin, self.robot_prim_path)
        else:
            self.robmatskin = "default"
            apply_matdict_to_prim_and_children(self.stage, self.orimat, self.robot_prim_path)
        self.assign_alarm_skin()

    def toggle_show_joints_close_to_limits(self, notoggle=False):
        if not notoggle:
            self.show_joints_close_to_limits = not self.show_joints_close_to_limits
        # print(f"toggle_show_joints_close_to_limits on {rcfg.robot_name} {rcfg.robot_id} - {rcfg.show_joints_close_to_limits}")
        if self.show_joints_close_to_limits:
            self.assign_alarm_skin()
            self.check_alarm_status()
            self.realize_joint_alarms(force=True)
        else:
            if self.robmatskin == "default":
                self.ensure_orimat()
                # print("Reverting to original materials (default)")
                apply_matdict_to_prim_and_children(self._stage, self.orimat, self.robot_prim_path)
            else:
                 #print(f"Reverting to {rcfg.robmatskin}")
                apply_material_to_prim_and_children(self._stage, self._matman, self.robmatskin, self.robot_prim_path)
        # print("toggle_show_joints_close_to_limits done")
        return self.show_joints_close_to_limits

    def assign_alarm_skin(self):
        if self.robmatskin == "Red_Glass":
            self.alarmskin = "Blue_Glass"
        else:
            self.alarmskin = "Red_Glass"

    def realize_joint_alarms(self, force=False):
        # print(f"realize_joint_alarms force:{force}")
        if self.show_joints_close_to_limits:
            self.check_alarm_status()
            for j, jstate in enumerate(self.jchk_str):
                if force or (jstate != self.last_jchk_str[j]):
                    link_path = self.link_paths[j]
                    joint_in_alarm = jstate != "."
                    if joint_in_alarm:
                        # print(f"   changing {link_path} to {rcfg.alarmskin} - inalarm:{joint_in_alarm}")
                        # print(f"Joint {jn} is close to limit for {rcfg.robot_name} {rcfg.robot_id} link_path:{link_path}")
                        apply_material_to_prim_and_children(self.stage, self.matman, self.alarmskin, link_path)
                    else:
                        # print(f"Joint {jn} is not close to limit for {rcfg.robot_name} {rcfg.robot_id} link_path:{link_path}")
                        if self.robmatskin == "default":
                            self.ensure_orimat()
                            # print(f"   changing {link_path} to rcfg.orimat - inalarm:{joint_in_alarm}")
                            apply_matdict_to_prim_and_children(self.stage, self.orimat, link_path)
                        else:
                            # print(f"   changing {link_path} to {rcfg.robmatskin} - inalarm:{joint_in_alarm}")
                            apply_material_to_prim_and_children(self.stage, self.matman, self.robmatskin, link_path)
            self.last_jchk_str = copy.deepcopy(self.jchk_str)
