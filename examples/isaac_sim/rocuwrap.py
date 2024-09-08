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
from omni.isaac.core.utils.stage import get_current_stage
from senut import apply_material_to_prim_and_children, apply_matdict_to_prim_and_children, build_material_dict, get_link_paths

############################################################
import torch

from rocu_components import RocuConfiguator, RocuMoveMode, RocuTranMan
from mgr_ut import get_args, get_vek, print_mat, list4to_quatd, quatd_to_list4, get_sphere_entry
from rocu_reachability import ReachGridMan

from rocu_reachability import GridRenderFilter, GridRenderStyle


args = get_args()

RocuConfig = RocuConfiguator()

class RocuWrapper:

    def __init__(self, robid):
        self.robid = robid
        self.name = "robot "+robid
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
        self.sim_js_names = []

        self.coli_spheres = None
        self.coli_spherenames = None
        self.coli_spheres_visible = False

        self.past_pose = None
        self.past_orientation = None
        self.target_pose = None
        self.target_orientation = None
        self.trigger = False
        self.step_index = -1

        self.vizi_spheres = False
        self.reactive = None
        self.reach_partial_pose = None
        self.hold_partial_pose = None
        self.constrain_grasp_approach = None

        self.cur_cmd_plan = None
        self.grid_closest = None
        self.grid_closeest_dist = 1e6

        self.pose_metric = None
        self.num_targets = 0

        self.cmd_plan_queue = []

        self.ik_result = None

        self.upper_dof_lim = None
        self.lower_dof_lim = None
        self.count_unique_solutions = True

        self.ik_solver = None
        self.motion_gen = None

        self.n_x, self.n_y, self.n_z = 9, 9, 9
        self.max_x, self.max_y, self.max_z = 0.5, 0.5, 0.5

        self.grid_succ_rad = 40
        self.grid_fail_rad = 20

        self.grid_timer_tick = 20
        self.rgm: ReachGridMan = ReachGridMan(self)

        self.jchk_str = ""
        self.last_jchk_str = ""
        self.jchk_str_val = ""

        self.nseed = 20

        # robot_cfg = load_yaml(robot_cfg_path)["robot_cfg"]
        self.LoadRobotCfg(self.robot_config_path)

        if self.external_asset_path is not None:
            self.robot_cfg["kinematics"]["external_asset_path"] = self.external_asset_path
        if self.external_robot_configs_path is not None:
            self.robot_cfg["kinematics"]["external_robot_configs_path"] = self.external_robot_configs_path
        self.j_names = self.robot_cfg["kinematics"]["cspace"]["joint_names"]
        self.default_config = self.robot_cfg["kinematics"]["cspace"]["retract_config"]

# --------------- Initialization ------------------

    def LoadAndPositionRobot(self, prerot, pos, ori, subroot=""):
        self.rocuTranman.set_transform(prerot=prerot, pos=pos, ori=ori)
        rp, ro = self.rocuTranman.get_robot_base()
        self.robot_name = f"robot_{self.robid}"
        self.robot, self.robot_prim_path = add_robot_to_scene(self.robot_cfg, self.my_world, position=rp, orient=ro, subroot=subroot, robot_name=self.robot_name)
        self.articulation_controller = self.robot.get_articulation_controller()
        self.init_alarm_skin()

    def LoadRobotCfg(self, robot_pathname):
        self.robot_cfg = load_yaml(robot_pathname)["robot_cfg"]

    def SetGridSize(self, n_x=9, n_y=9, n_z=9):
        self.n_x = n_x
        self.n_y = n_y
        self.n_z = n_z
        self.rgm.SetGridSize(n_x, n_y, n_z)
        self.rgm.InitPositionGridOffset()

    def SetGridSpan(self, max_x=0.5, max_y=0.5, max_z=0.5):
        self.max_x = max_x
        self.max_y = max_y
        self.max_z = max_z
        self.rgm.SetGridSpan(max_x, max_y, max_z)
        self.rgm.InitPositionGridOffset()

    def ChangeGridSpan(self, fak = 1.5):
        if self.rgm is not None:
            self.rgm.ChangeGridSpan(fak)
        else:
            print("In ChangeGridSpan rgm is None")

    def ChangeGridSize(self, fak = 1.5):
        if self.rgm is not None:
            self.rgm.ChangeGridSize(fak)
        else:
            print("In ChangeGridSize rgm is None")

    def SetGridTimererTick(self, timer_tick):
        self.grid_timer_tick = timer_tick

    def ChangeNumSeeds(self, fak = 2):
        if self.rgm is not None:
            return self.rgm.ChangeNumSeeds(fak)
        else:
            print("In ChangeNumSeeds rgm is None")
            return -1

    def ChangeGridSlicerValue(self, slicer: int):
        if self.rgm is not None:
            return self.rgm.ChangeGridSlicerValue(slicer)
        else:
            print("In ChangeGridSlicer rgm is None")
            return -1

    def ChangeGridSlicerMode(self):
        if self.rgm is not None:
            return self.rgm.ChangeGridSlicerMode()
        else:
            print("In ChangeGridSlicerMode rgm is None")
            return -1

    def ChangeGridRenderFilter(self, filter: GridRenderFilter=None):
        if self.rgm is not None:
            return self.rgm.ChangeGridRenderFilter(filter)
        else:
            print("In ChangeGridRenderFilter rgm is None")
            return "None"

    def ChangeGridRenderStyle(self, style: GridRenderStyle=None):
        if self.rgm is not None:
            return self.rgm.ChangeGridRenderStyle(style)
        else:
            print("In ChangeGridRenderStyle rgm is None")
            return "None"

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

# ------------------ Utilities-------------------

    def dst(self, p1, p2):
        rv = np.linalg.norm(p1 - p2)
        return rv

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
            case RocuMoveMode.FollowTargetWithInvKin | RocuMoveMode.ReachabilityWithInvKin:
                rollout_fn = self.ik_solver.solver.rollout_fn

        start_state = JointState.from_position(
            rollout_fn.dynamics_model.retract_config.view(1, -1).clone(),
            joint_names=rollout_fn.joint_names,
        )
        state = rollout_fn.compute_kinematics(start_state)
        sp = state.ee_pos_seq.cpu()[0]
        sq = state.ee_quat_seq.cpu()[0]
        return sp, sq

    def get_cur_eepose(self):
        match self.move_mode:
            case RocuMoveMode.FollowTargetWithMoGen:
                rollout_fn = self.motion_gen.rollout_fn
            case RocuMoveMode.FollowTargetWithInvKin | RocuMoveMode.ReachabilityWithInvKin:
                rollout_fn = self.ik_solver.solver.rollout_fn
        state = rollout_fn.compute_kinematics(self.cu_js)
        sp = state.ee_pos_seq.cpu()[0]
        sq = state.ee_quat_seq.cpu()[0]
        return sp, sq

# ------------------ Motion Generation Intialization -------------------

    def SetupMoGenPlanConfig(self):
        self.plan_config = MotionGenPlanConfig(
            enable_graph=False,
            enable_graph_attempt=2,
            max_attempts=self.max_attempts,
            enable_finetune_trajopt=True,
            parallel_finetune=True,
        )

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
            case RocuMoveMode.FollowTargetWithInvKin | RocuMoveMode.ReachabilityWithInvKin:
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
        self.rgm.InitSolver(n_obstacle_cuboids, n_obstacle_mesh)
        self.InitInvKinGrid(n_obstacle_cuboids, n_obstacle_mesh)

    def InitInvKinGrid(self, n_obstacle_cuboids, n_obstacle_mesh):
        self.rgm.InitSolver(n_obstacle_cuboids, n_obstacle_mesh)
        self.rgm.InitPositionGridOffset()

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
        )
        self.ik_solver = IKSolver(self.ik_config)
        self.InitInvKinGrid(n_obstacle_cuboids, n_obstacle_mesh)

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
                print(f"Warming up MoGen for {self.robot_name}...")
                self.motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False, parallel_finetune=True)
                elap = time.time() - start_warmup
                print(f"Curobo MoGen is Ready and Warmed-up - took:{elap:.2f} secs")
            case RocuMoveMode.FollowTargetWithInvKin | RocuMoveMode.ReachabilityWithInvKin:
                print(f"Warming up InvKin for {self.robot_name}...")
                start_warmup = time.time()
                elap = time.time() - start_warmup
                print(f"Curobo InvKin is Ready and Warmed-up - took:{elap:.2f} secs")
                pass
            case _:
                carb.log_warn(f"Move Mode {self.move_mode} not implemented yet.")

# ------------------- Motion Generation Exection -------------------

    def UpdateWorldObsticles(self, obstacles):
        match self.move_mode:
            case RocuMoveMode.FollowTargetWithMoGen:
                rv = self.motion_gen.update_world(obstacles)
            case RocuMoveMode.FollowTargetWithInvKin | RocuMoveMode.ReachabilityWithInvKin:
                rv = self.ik_solver.update_world(obstacles)
            case _:
                carb.log_warn(f"Move Mode {self.move_mode} not implemented yet.")
                rv = None
        self.rgm.UpdateWorldObsticles(obstacles)
        return rv

    def StartStep(self, step_index):
        self.step_index = step_index
        if step_index == 1:
            self.Reset()

        if self.circle_target:
            self.UpdateCirclingTarget()

    def EndStep(self) -> bool:
        requestPause = False
        match self.move_mode:
            case RocuMoveMode.FollowTargetWithMoGen:
                self.UpdateJointState()
                self.RealizeJointAlarms()
                self.ProcessCollisionSpheres()
                self.ProcessTarget()
                requestPause = self.ExecuteMoGenCmdPlan()
            case RocuMoveMode.FollowTargetWithInvKin:
                self.UpdateJointState()
                self.RealizeJointAlarms()
                self.ProcessCollisionSpheres()
                self.ProcessTarget()
                requestPause = self.ExecuteInvKinCmdPlan()
            case RocuMoveMode.ReachabilityWithInvKin:
                # TODO - implement reachability with InvKin
                self.UpdateJointState()
                self.RealizeJointAlarms()
                self.ProcessCollisionSpheres()
                self.ProcessTarget()
                requestPause = self.ExecuteGridInvKinCmdPlan()
            case _:
                carb.log_warn(f"Move Mode {self.move_mode} not implemented yet.")

        return requestPause

    def UpdateJointState(self):
        self.sim_js = self.robot.get_joints_state()
        self.sim_js_names = self.robot.dof_names
        if self.upper_dof_lim is None:
            self.upper_dof_lim = self.robot.dof_properties["upper"]
            self.lower_dof_lim = self.robot.dof_properties["lower"]
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

    def GetKinematics(self):
        kine = None
        match self.move_mode:
            case RocuMoveMode.FollowTargetWithMoGen:
                kine = self.motion_gen.kinematics
            case RocuMoveMode.FollowTargetWithInvKin | RocuMoveMode.ReachabilityWithInvKin:
                kine = self.ik_solver.kinematics
        return kine


# -----------------  Collision Sphere Visulaization Code ---------------------

    def ToggleCollisionSphereVisiblity(self):
        self.vizi_spheres = not self.vizi_spheres

    def CreateCollisionSpheres(self):
        # get a fresh list of spheres:
        kine = self.GetKinematics()
        if kine is None:
            carb.log_warn("Can't find kinematics - can't do collision sphers")
            return
        self.coli_sph_list = kine.get_robot_as_spheres(self.cu_js.position)
        if self.coli_spheres is None:
            config_spheres = self.robot_cfg["kinematics"]["collision_spheres"]
            self.coli_spheres = []
            self.coli_spherenames = []
            # create spheres:
            ncreated = 0
            for sidx, sph in enumerate(self.coli_sph_list[0]):
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
                self.coli_spheres.append(sp)
                self.coli_spherenames.append(sname)
                ncreated += 1
            print(f"Created {ncreated} Spheres")

    def UpdateCollistionSpherePositions(self):
        kine = self.GetKinematics()
        if kine is None:
            # carb.log_warn("Can't find kinematics - can't do collision sphers")
            return
        self.coli_sph_list = kine.get_robot_as_spheres(self.cu_js.position)
        s_ori = np.array([1, 0, 0, 0])
        for sidx, sph in enumerate(self.coli_sph_list[0]):
            if not np.isnan(sph.position[0]):
                sp_wc, _ = self.rocuTranman.rcc_to_wc(sph.position, s_ori)
                self.coli_spheres[sidx].set_world_pose(position=np.ravel(sp_wc))
                self.coli_spheres[sidx].set_radius(float(sph.radius))

    def DeleteCollisionSpheres(self):
        if self.coli_spherenames is not None:
            for sn in self.coli_spherenames:
                self.stage.RemovePrim(sn)
        self.coli_spheres = None
        self.coli_spherenames = None
        self.coli_spheres_visible = False

    def ProcessCollisionSpheres(self):
        if self.vizi_spheres:
            if self.coli_spheres is None:
                self.CreateCollisionSpheres()
            else:
                self.UpdateCollistionSpherePositions()
            self.coli_spheres_visible = True

        if not self.vizi_spheres and self.coli_spheres_visible:
            self.DeleteCollisionSpheres()

# ----------------------- Target Processing Code --------------------
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

    def ProcessTarget(self):
        self.cube_position, self.cube_orientation = self.target.get_world_pose()
        match self.move_mode:
            case RocuMoveMode.FollowTargetWithMoGen:
                triggerMoGen = self.CalcTargetTrigger()
                if triggerMoGen:
                    print("Triggering MoGen")

                    self.DoMoGenToTarget()
            case RocuMoveMode.FollowTargetWithInvKin:
                triggerInvKin = self.CalcTargetTrigger()
                if triggerInvKin:
                    print("Triggering InvKin")

                    self.DoInvKinToTarget()

            case RocuMoveMode.ReachabilityWithInvKin:
                triggerGridInvKin = self.CalcTargetTrigger()
                if triggerGridInvKin:
                    print("Triggering GridInvKin")
                    # self.CalcReachabilityToTarget()
                    self.DoGridInvKinToTarget()

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

    def MoveTargetToEepose(self):
        sp_rcc, sq_rcc = self.get_cur_eepose()
        sp_wc, sq_wc = self.rcc_to_wc(sp_rcc, sq_rcc)
        self.SetTargetPose(sp_wc, sq_wc)

    def CalcTargetTrigger(self):
        if self.past_pose is None:
            self.past_pose = self.cube_position
        if self.target_pose is None:
            self.target_pose = self.cube_position
        if self.target_orientation is None:
            self.target_orientation = self.cube_orientation
        if self.past_orientation is None:
            self.past_orientation = self.cube_orientation

        self.static_robo = True
        cube_pos, cube_ori = self.cube_position, self.cube_orientation
        cubeDiffFromTarget = (self.dst(cube_pos, self.target_pose) > 1e-3
                             or self.dst(cube_ori, self.target_orientation) > 1e-3)
        cubeStill = self.dst(self.past_pose, cube_pos) == 0.0 and self.dst(self.past_orientation, cube_ori) == 0.0

        self.trigger = (cubeDiffFromTarget and cubeStill and self.static_robo)
        # print(f"trigger:{self.trigger} cubeDiffFromTarget:{cubeDiffFromTarget} cubeStill:{cubeStill}")

        if self.circle_target:
            self.trigger = self.cur_cmd_plan is None

        return self.trigger

# ----------------------- Arm Movement and Command Plan Processing Code --------------------

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

        # print("ik_goal-p:", ee_pos_rcc, " q:", ee_ori_rcc)
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

            cmd_plan = cmd_plan.get_ordered_joint_state(common_js_names)
            nsteps = len(cmd_plan.position)
            msg = f" interpolated plan with {nsteps} steps"
            self.QueCmdPlan(cmd_plan, msg)

            self.cmd_idx = 0

        else:
            msg = f"Plan did not converge to a solution. Status:{result.status}. No action is being taken."
            print(msg)
            carb.log_warn(msg)

        self.target_pose = self.cube_position
        self.target_orientation = self.cube_orientation

    def InitPositionGridOffsetElems(self, ee_pose):
        self.rgm.InitGridSize(self.n_x, self.n_y, self.n_z,
                              self.max_x, self.max_y, self.max_z,
                              self.grid_succ_rad, self.grid_fail_rad)

    def DoGridInvKinToTarget(self):
        rv = self.DoGridInvKinToPosOri(self.cube_position, self.cube_orientation)
        return rv

    def DoGridInvKinToPosOri(self, pos, ori):
        ik_result = self.rgm.CalcReachabilityAroundPosOri(pos, ori)
        self.rgm.BuildReachabilityGrid(ik_result, self.rgm.goal_pose)

        successful = torch.any(ik_result.success)
        nsucess = torch.sum(ik_result.success).item()

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

        if successful:
            self.num_targets += 1
            cmd_plan = ik_result.js_solution[ik_result.success]
            # get only joint names that are in both:
            idx_list = []
            common_js_names = []
            for x in self.sim_js_names:
                if x in cmd_plan.joint_names:
                    idx_list.append(self.robot.get_dof_index(x))
                    common_js_names.append(x)
            # idx_list = [robot.get_dof_index(x) for x in sim_js_names]

            cmd_plan = cmd_plan.get_ordered_joint_state(common_js_names)
            msg = f" - Grid with {nsucess} reachable solutions"
            self.QueCmdPlan(cmd_plan, msg)

            self.cmd_idx = 0

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
        # print("ik_goal-p:", ee_pos_rcc, " q:", ee_ori_rcc)
        ik_goal = Pose(
            position=self.tensor_args.to_device(ee_pos_rcc),
            quaternion=self.tensor_args.to_device(ee_ori_rcc),
        )
        st_time = time.time()
        retract_pos = self.cu_js.position.view(1, -1)
        seed_pos = self.cu_js.position.view(1, 1, -1)
        ik_result = self.ik_solver.solve_single(ik_goal, retract_pos, seed_pos)
        total_time = (time.time() - st_time)
        solve_time = ik_result.solve_time
        succount = torch.count_nonzero(ik_result.success).item(),
        pos_err = torch.mean(ik_result.position_error) * 100.0
        rot_err = torch.mean(ik_result.rotation_error) * 100.0
        msg = f"Finished - success:{succount} total_time {total_time:.3f} solve_time:{solve_time:.3f} pos_err:{pos_err:.3f} rot_err:{rot_err:.3f}"
        print(msg)
        # print(
        #     "Finished, Solve Time(s), Total Time(s)",
        #     torch.count_nonzero(ik_result.success).item(),
        #     ik_result.solve_time,
        #     total_time,
        #     1.0 / total_time,
        #     torch.mean(ik_result.position_error) * 100.0,
        #     torch.mean(ik_result.rotation_error) * 100.0,
        # )
        self.ik_result = ik_result
        self.target_pose = self.cube_position
        self.target_orientation = self.cube_orientation

    def QueCmdPlan(self, cmd_plan, comment=""):
        # should do a plausiblity check on cmd_plan
        if comment == "":
            comment = f"CmdPlan-{len(self.cmd_plan_queue)}"
        self.cmd_plan_queue.append((cmd_plan, comment))

    def ShuffleCmdPlanQueue(self):
        if self.cur_cmd_plan is None:
            if len(self.cmd_plan_queue) > 0:
                len_bef = len(self.cmd_plan_queue)
                (self.cur_cmd_plan, self.cur_cmd_plan_cmt) = self.cmd_plan_queue.pop(0)
                len_aft = len(self.cmd_plan_queue)
                print(f"ShuffleCmdPlanQueue len_bef:{len_bef} len_aft:{len_aft}")

    def ExecuteMoGenCmdPlan(self):
        requestPause = False
        self.ShuffleCmdPlanQueue()
        if self.cur_cmd_plan is not None:
            print(f"Executing plan step {self.cmd_idx}/{len(self.cur_cmd_plan.position)} - {self.cur_cmd_plan_cmt}")
            cmd_state = self.cur_cmd_plan[self.cmd_idx]
            self.past_cmd = cmd_state.clone()
            # get full dof state
            if cmd_state.velocity is None:
                art_action = ArticulationAction(
                    cmd_state.position.cpu().numpy(),
                    None,
                    joint_indices=self.idx_list,
                )
            else:
                art_action = ArticulationAction(
                    cmd_state.position.cpu().numpy(),
                    cmd_state.velocity.cpu().numpy(),
                    joint_indices=self.idx_list,
                )
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

    def ExecuteGridInvKinCmdPlan(self):
        requestPause = False
        self.ShuffleCmdPlanQueue()

        if self.cur_cmd_plan is not None and self.step_index % self.grid_timer_tick == 0 and True:
            print(f"Executing plan step {self.cmd_idx}/{len(self.cur_cmd_plan.position)} - {self.cur_cmd_plan_cmt}")
            requestPause = True
            cmd_state = self.cur_cmd_plan[self.cmd_idx]

            pos = cmd_state.position.view(1, -1)
            self.robot.set_joint_positions(pos.cpu().numpy(), self.idx_list)
            sp_wc_np, sq_wc_np = self.rgm.CalcPoseGrid(pos, needAsNumpy=True)
            dist = self.dst(sp_wc_np, self.cube_position) + self.dst(sq_wc_np, self.cube_orientation)
            # print(f"Dist:{dist} - {sp_wc_np} - {self.cube_position} ")
            if dist < self.grid_closeest_dist:
                self.grid_closest = cmd_state
                self.grid_closeest_dist = dist
                # print(f"Setting closest to target idx:{self.cmd_idx} dist:{dist}")
                # pos = self.grid_closest.position
                # print(" moving to joint pos:", pos, " idx_list:", self.idx_list)

            # set desired joint angles obtained from IK:
            # articulation_controller.apply_action(art_action)
            self.cmd_idx += 1
            if self.cmd_idx >= len(self.cur_cmd_plan.position):
                print("Plan Completed")
                self.cmd_idx = 0
                self.cur_cmd_plan = None
                if self.grid_closest is not None:
                    print("Moving closest to target closest_dist:", self.grid_closeest_dist)
                    pos = self.grid_closest.position.view(1, -1)
                    self.robot.set_joint_positions(pos.cpu().numpy(), self.idx_list)

                self.grid_closest = None      # this initialization should be done somewhere else in the next iteration
                self.grid_closeest_dist = 1e6

        return requestPause

    def ExecuteInvKinCmdPlan(self):
        # print("ExecuteInvKinCmdPlan")
        requestPause = False
        if self.ik_result is not None:
            succount = torch.count_nonzero(self.ik_result.success)
            print("ExecuteInvKinCmdPlan success count:", succount)
            if succount > 0:
                joint_positions = self.ik_result.js_solution.position[0]
                jp_numpy = joint_positions.cpu().numpy()
                use_apply_action = False
                if use_apply_action:
                    art_action = ArticulationAction(
                        jp_numpy,
                        None,
                        joint_indices=self.idx_list,
                    )
                    self.ApplyAction(art_action)
                    print("Applied Action - ArtAction:", art_action)
                else:
                    self.robot.set_joint_positions(jp_numpy, self.idx_list)
                    print("Set Joint Pos:", jp_numpy)
                requestPause = True
            self.ik_result = None
            return requestPause

    # def ShowReachabilityGridOld(self, clear=True):
    #     print(f"ShowReachability {self.robid}")
    #     res = self.rgm.CalcReachabilityAroundPosOri(self.cube_position, self.cube_orientation)
    #     if self.count_unique_solutions:
    #         unique = res.get_batch_unique_solution()
    #     else:
    #         unique = None
    #     self.rgm._show_reachability_grid(res.solution, self.rgm.goal_pose, res.success, unique=unique, clear=clear)

    def ShowReachabilityGrid(self, clear=True):
        print(f"ShowReachability {self.robid}")
        self.rgm.ShowReachabilityGridToPosOri(self.cube_position, self.cube_orientation)


    def ClearReachabilityGrid(self):
        self.rgm.ClearReachabilityGrid()

    def ToggleDebugSphereMode(self):
        self.rgm.ToggleDebugSphereMode()

    def RotateGridFilter(self):
        self.rgm.RotateGridFilter()



# ----------------------- Joint Alarm Code ---------------------------
    def init_alarm_skin(self):
        self.robmatskin = "default"
        self.alarmskin = "Red_Glass"
        self.show_joints_close_to_limits = True
        self.ensure_orimat()

    def check_alarm_status(self):
        self.jchk_str = ""
        self.jchk_str_val = ""
        for j, jn in enumerate(self.sim_js_names):
            jpos = self.sim_js.positions[j]
            llim = self.lower_dof_lim[j]
            ulim = self.upper_dof_lim[j]
            denom = ulim - llim
            if denom == 0:
                denom = 1
            pct = 100 * (jpos - llim) / denom
            self.jchk_str_val += f"{pct:.0f} "
            if pct < 10:
                self.jchk_str += "L"
            elif pct > 90:
                self.jchk_str += "U"
            else:
                self.jchk_str += "."
        # print("joint_check:", self.jchk_str, " ", self.jchk_str_val)

    def change_material(self, matname):
        apply_material_to_prim_and_children(self.stage, self.matman, matname, self.robot_prim_path)

    def ensure_orimat(self):
        if not hasattr(self, "orimat"):
            self.orimat = build_material_dict(self.stage, self.robot_prim_path)

    def toggle_material(self):
        if self.robmatskin == "default":
            self.robmatskin = "Blue_Glass"
            apply_material_to_prim_and_children(self.stage, self.matman, self.robmatskin, self.robot_prim_path)
        else:
            self.robmatskin = "default"
            apply_matdict_to_prim_and_children(self.stage, self.orimat, self.robot_prim_path)
        self.assign_alarm_skin()

    def toggle_show_joints_close_to_limits(self, notoggle=False):
        if not notoggle:
            self.show_joints_close_to_limits = not self.show_joints_close_to_limits
        if self.show_joints_close_to_limits:
            self.assign_alarm_skin()
            self.check_alarm_status()
            self.RealizeJointAlarms(force=True)
        else:
            if self.robmatskin == "default":
                self.ensure_orimat()
                apply_matdict_to_prim_and_children(self.stage, self.orimat, self.robot_prim_path)
            else:
                apply_material_to_prim_and_children(self.stage, self.matman, self.robmatskin, self.robot_prim_path)
        return self.show_joints_close_to_limits

    def assign_alarm_skin(self):
        if self.robmatskin == "Red_Glass":
            self.alarmskin = "Blue_Glass"
        else:
            self.alarmskin = "Red_Glass"

    def RealizeJointAlarms(self, force=False):
        # print(f"realize_joint_alarms force:{force}")
        if self.show_joints_close_to_limits:
            self.check_alarm_status()
            if self.last_jchk_str=="":  # first time
                self.last_jchk_str = copy.deepcopy(self.jchk_str)
                return
            for j, jstate in enumerate(self.jchk_str):
                if force or (jstate != self.last_jchk_str[j]):
                    link_path = self.link_paths[j]
                    joint_in_alarm = jstate != "."
                    if joint_in_alarm:
                        apply_material_to_prim_and_children(self.stage, self.matman, self.alarmskin, link_path)
                    else:
                        if self.robmatskin == "default":
                            self.ensure_orimat()
                            apply_matdict_to_prim_and_children(self.stage, self.orimat, link_path)
                        else:
                            apply_material_to_prim_and_children(self.stage, self.matman, self.robmatskin, link_path)
            self.last_jchk_str = copy.deepcopy(self.jchk_str)
