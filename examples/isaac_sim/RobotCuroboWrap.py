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


# Third Party
from torch.fx.experimental.symbolic_shapes import expect_true
import time






# Third Party
import carb
import numpy as np
from helper import add_extensions, add_robot_to_scene
from omni.isaac.core import World
from omni.isaac.core.objects import cuboid, sphere

########### OV #################
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.viewports import set_camera_view


# CuRobo
# from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.types.state import JointState
from curobo.util.logger import log_error, setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import (
    get_assets_path,
    get_filename,
    get_path_of_dir,
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    PoseCostMetric,
)
from pxr import Gf, Sdf, Usd, UsdGeom
from rotations import euler_angles_to_quat, matrix_to_euler_angles, rot_matrix_to_quat, gf_rotation_to_np_array
from omni.isaac.core.utils.stage import get_current_stage

############################################################


########### OV #################;;;;;
import typing

from mgrut import get_args, get_vek, print_mat, list4to_quatd, quatd_to_list4, get_sphere_entry

args = get_args()


class TranMan:
    def __init__(self, usealt=True):
        self.rob_pos = Gf.Vec3d(0,0,0)
        self.rob_ori_quat = Gf.Quatd(1,0,0,0)
        self.rob_ori_euler = Gf.Vec3d(0,0,0)
        self.rob_ori_sel = "0,0,0"
        # self.memstage: Usd.Stage = Usd.Stage.CreateInMemory()
        self.memstage: Usd.Stage = get_current_stage()
        self.default_prim: Usd.Prim = UsdGeom.Xform.Define(self.memstage, Sdf.Path("/World")).GetPrim()
        self.xformpath = self.default_prim.GetPath().AppendPath("Xform")
        self.memstage.SetDefaultPrim(self.default_prim)

        self.xformw_full_prim: Usd.Prim = UsdGeom.Xform.Define(self.memstage, "/World/XformFull")
        self.xformw_robot_proxy_prim: Usd.Prim = UsdGeom.Xform.Define(self.memstage, "/World/XformRobProxy")

        self.xform_full_pre_rot_op = self.xformw_full_prim.AddRotateXYZOp(opSuffix='prerot')
        self.xform_full_tran_op = self.xformw_full_prim.AddTranslateOp()
        self.xform_full_rot_op = self.xformw_full_prim.AddRotateXYZOp()

        self.xform_robopt_proxy_tran_op = self.xformw_robot_proxy_prim.AddTranslateOp()
        self.xform_robopt_proxy_orient_op = self.xformw_robot_proxy_prim.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
        self.xform_robopt_proxy_scale_op = self.xformw_robot_proxy_prim.AddScaleOp()

        self.usealt = usealt
        print("RobDeco created usealt:", usealt)

    def get_world_transform_xform_full(self, prim: Usd.Prim, dump=False) -> typing.Tuple[Gf.Vec3d, Gf.Rotation, Gf.Vec3d, Gf.Matrix3d]:
        xform = UsdGeom.Xformable(prim)
        time = Usd.TimeCode.Default() # The time at which we compute the bounding box
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
        t, r, s, m = self.get_world_transform_xform_full(self.xformw_full_prim)
        return t, r, s, m

    def set_robot_proxy_tran(self, tranvek, oriquat):
        self.xform_robopt_proxy_tran_op.Set(tranvek)
        qlist = oriquat.tolist()
        q = Gf.Quatd(qlist[0], qlist[1:])
        self.xform_robopt_proxy_orient_op.Set(value=q)
        self.xform_robopt_proxy_scale_op.Set(Gf.Vec3d(1,1,1))
        t, r, s, m = self.get_world_transform_xform_full(self.xformw_robot_proxy_prim, dump=False)
        return t, r, s, m

    def set_transform(self, prerot, pos, ori):

        self.rob_prerot_euler = self.to_gfvec(prerot)
        self.prerot = prerot
        self.rob_pos = Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2]))
        self.rob_ori_euler = self.to_gfvec(ori)
        self.rob_ori_quat_nparray = euler_angles_to_quat(self.rob_ori_euler, degrees=True)
        self.rob_ori_quat = list4to_quatd(self.rob_ori_quat_nparray)

        (self.tran, self.rotmat3d_gfrot, _, self.rcc_to_wc_transform) = self.find_rcc_to_wc_tansform(self.rob_prerot_euler, self.rob_pos, self.rob_ori_euler)
        self.rotmat3d = Gf.Matrix3d(self.rotmat3d_gfrot)
        self.inv_rotmat3d = self.rotmat3d.GetTranspose()

        self.rotmat3d_eulers = matrix_to_euler_angles(self.rotmat3d, degrees=True)
        self.rotmat3d_quat_nparray = euler_angles_to_quat(self.rotmat3d_eulers, degrees=True)
        npmat3x3 = self.to_npmat3x3(self.inv_rotmat3d)
        self.rotmat3d_quat_nparray_inv = rot_matrix_to_quat(npmat3x3)
        self.rotmat3d_eulers_inv = matrix_to_euler_angles(npmat3x3, degrees=True)
        (self.robproxy_tran, self.robproxy_rot, _, self.robproxy_world_tran) = self.set_robot_proxy_tran(self.tran, self.rotmat3d_quat_nparray)

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
        npmat3x3 = np.zeros((3,3), dtype=np.float32)
        for i in range(3):
            for j in range(3):
                npmat3x3[i,j] = gfmat[i,j]
        return npmat3x3

    def quat_apply(self, q1, q2):
        q1inv = q1.GetInverse()
        q2q = Gf.Quatd(float(q2[0]), float(q2[1]), float(q2[2]), float(q2[3]))
        rv = q1inv * q2q * q1
        return rv

    def rcc_to_wc(self, pos, ori):
        pos_gfv = self.to_gfvec(pos)
        pos_new = self.tran + pos_gfv*self.inv_rotmat3d
        return pos_new, ori

    def wc_to_rcc(self, pos, ori):
        pos_gfv = self.to_gfvec(pos)
        pos_new = (pos_gfv - self.tran)*self.rotmat3d
        return pos_new, ori

    def dump_robot_transforms(self, robpathname):
        robpath = Sdf.Path(robpathname)
        robprim = self.memstage.GetPrimAtPath(robpath)
        (t, r, s, m) = self.get_world_transform_xform_full(robprim)
        print("world_tranlate:", t)
        print("world_rotate:", r)
        print_mat("world_transform:", 4, 4, m)
        print_mat("rotmat3d:", 3, 3, self.rotmat3d)
        print_mat("inv_rotmat3d:", 3, 3, self.inv_rotmat3d)


class RobotCuroboWrapper:

    def __init__(self):
        self.tranman = TranMan()

    def Initialize(self, robot_config_path, external_asset_path, external_robot_configs_path, my_world):
        self.robot_config_path = robot_config_path
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
        self.robot_cfg_path = robot_config_path
        self.external_asset_path = external_asset_path
        self.external_robot_configs_path = external_robot_configs_path
        self.my_world = my_world
        self.stage = my_world.stage
        self.cu_js = None

        self.spheres = None
        self.spherenames = None
        self.spheres_visable = False

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

        self.cmd_plan = None
        self.pose_metric = None
        self.num_targets = 0

        # robot_cfg = load_yaml(robot_cfg_path)["robot_cfg"]
        self.LoadRobotCfg(self.robot_cfg_path)

        if external_asset_path is not None:
            self.robot_cfg["kinematics"]["external_asset_path"] = external_asset_path
        if external_robot_configs_path is not None:
            self.robot_cfg["kinematics"]["external_robot_configs_path"] = external_robot_configs_path
        self.j_names = self.robot_cfg["kinematics"]["cspace"]["joint_names"]
        self.default_config = self.robot_cfg["kinematics"]["cspace"]["retract_config"]

        # self.tranman.set_transform(prerot=get_vek(args.robprerot), pos=get_vek(args.robpos), ori=get_vek(args.robori))
        # rp, ro = self.tranman.get_robot_base()

        # robot, robot_prim_path = add_robot_to_scene(self.robot_cfg, my_world, position=rp, orient=ro)

    def PositionRobot(self, prerot, pos, ori):
        self.tranman.set_transform(prerot=prerot, pos=pos, ori=ori)
        rp, ro = self.tranman.get_robot_base()
        self.robot, self.robot_prim_path = add_robot_to_scene(self.robot_cfg, self.my_world, position=rp, orient=ro)
        self.articulation_controller = self.robot.get_articulation_controller()

    def SetMoGenOptions(self, reactive=None, reach_partial_pose=None, hold_partial_pose=None, constrain_grasp_approach=None, vizi_spheres=None):
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

    def InitMotionGen(self, n_obstacle_cuboids, n_obstacle_mesh, world_cfg):
        trajopt_dt = None
        optimize_dt = True
        trajopt_tsteps = 16
        trim_steps = None
        self.max_attempts = 4
        self.world_cfg = world_cfg
        interpolation_dt = 0.05
        if self.reactive:
            trajopt_tsteps = 40
            trajopt_dt = 0.04
            optimize_dt = False
            max_attempts = 1
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

    def Reset(self):
        self.robot._articulation_view.initialize()
        self.idx_list = [self.robot.get_dof_index(x) for x in self.j_names]
        self.robot.set_joint_positions(self.default_config, self.idx_list)

        self.robot._articulation_view.set_max_efforts(
            values=np.array([5000 for i in range(len(self.idx_list))]), joint_indices=self.idx_list
        )

    def Warmup(self):
        print("warming up...")
        start_warmup = time.time()

        self.motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False, parallel_finetune=True)

        elap = time.time() - start_warmup

        print(f"Curobo is Ready and Warmed-up - took:{elap:.2f} secs")

    def CreateTarget(self, target_pos=None, target_ori=None):

        if target_pos is None or target_ori is None:
            sp_rcc, sq_rcc = self.motion_gen.get_start_pose()
            sp_wc, sq_wc = self.tranman.rcc_to_wc(sp_rcc, sq_rcc)
            if type(sq_wc) is Gf.Quatd:
                sq_wc = quatd_to_list4(sq_wc)

        if target_pos is None:
            target_pos = sp_wc

        if target_ori is None:
            target_ori = sq_wc

        self.target = cuboid.VisualCuboid(
            "/World/target",
            position=target_pos,
            orientation=target_ori,
            color=np.array([1.0, 0, 0]),
            size=0.05,
        )

        self.curcen, self.curori = self.target.get_world_pose()
        self.curvel = 0.02
        self.curang = 0
        self.currad = 0.1
        self.circle_target = False

        return self.target

    def CircleTarget(self):
        if self.circle_target:
            self.curang += self.curvel
            newpos = np.zeros(3)
            newpos[0] = self.curcen[0]  + self.currad*np.cos(self.curang)
            newpos[1] = self.curcen[1]  + self.currad*np.sin(self.curang)
            newpos[2] = self.curcen[2]
            self.target.set_world_pose(
                position=newpos,
                orientation=self.curori
            )
        pass


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
        # if (step_index % 100) == 0:
            # print(f"   sim_js_names: {sim_js_names}")
            # print(f"   sim_js.positions: {sim_js.positions}")
            # print(f"   sim_js.velocities: {sim_js.velocities}")

        if not self.reactive:
            self.cu_js.velocity *= 0.0
            self.cu_js.acceleration *= 0.0

        if self.reactive and self.past_cmd is not None:
            self.cu_js.position[:] = self.past_cmd.position
            self.cu_js.velocity[:] = self.past_cmd.velocity
            self.cu_js.acceleration[:] = self.past_cmd.acceleration
        self.cu_js = self.cu_js.get_ordered_joint_state(self.motion_gen.kinematics.joint_names)

    def HandleCollisionSpheres(self):
        if self.vizi_spheres:
            # get a fresh list of spheres:
            sph_list = self.motion_gen.kinematics.get_robot_as_spheres(self.cu_js.position)
            if self.spheres is None:
                config_spheres = self.robot_cfg["kinematics"]["collision_spheres"]
                self.spheres = []
                self.spherenames = []
                # create spheres:
                ncreated = 0
                for sidx, sph in enumerate(sph_list[0]):
                    sphentry = get_sphere_entry(config_spheres, sidx)
                    sname ="/curobo/robot_sphere_" + str(sidx)
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
                    sp_wc, _ = self.tranman.rcc_to_wc(sph.position, s_ori)
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
            else:
                s_ori = np.array([1, 0, 0, 0])
                for sidx, sph in enumerate(sph_list[0]):
                    if not np.isnan(sph.position[0]):
                        sp_wc, _ = self.tranman.rcc_to_wc(sph.position, s_ori)
                        self.spheres[sidx].set_world_pose(position=np.ravel(sp_wc))
                        self.spheres[sidx].set_radius(float(sph.radius))
            self.spheres_visable = True

        if not self.vizi_spheres and self.spheres_visable:
            if self.spherenames is not None:
                for sn in self.spherenames:
                    self.stage.RemovePrim(sn)
            self.spheres = None
            self.spherenames = None
            self.spheres_visable = False

    def CalcMoGenTrigger(self, cube_pos, cube_ori):
        self.cube_position = cube_pos
        self.cube_orientation = cube_ori
        if self.past_pose is None:
            self.past_pose = cube_pos
        if self.target_pose is None:
            self.target_pose = cube_pos
        if self.target_orientation is None:
            self.target_orientation = cube_ori
        if self.past_orientation is None:
            self.past_orientation = cube_ori

        self.static_robo = True
        # robot_static = True
        velmag = np.max(np.abs(self.sim_js.velocities))
        # if (velmag < 0.2) or reactive:
        #     static_robo = True
        #     pass
        pretrig = np.linalg.norm(cube_pos - self.target_pose) > 1e-3 or np.linalg.norm(cube_ori - self.target_orientation) > 1e-3
        self.trigger =  pretrig and np.linalg.norm(self.past_pose - cube_pos) == 0.0 and np.linalg.norm(self.past_orientation - cube_ori) == 0.0 and self.static_robo
        # print(f"trigger:{trigger} pretrig:{pretrig} velmag:{velmag:.2f} static_robo:{static_robo}")


        # print("pretrig:", pretrig, "  trigger:", self.trigger, "  static_robo:", self.static_robo,"  circle_target:", circle_target)
        if self.circle_target:
            self.trigger = self.cmd_plan is None

        return self.trigger

    def ApplyAction(self, art_action):
        self.articulation_controller.apply_action(art_action)

    def DoMoGen(self):
        # Set EE teleop goals, use cube for simple non-vr init:
        ee_pos_rcc, ee_ori_rcc = self.tranman.wc_to_rcc(self.cube_position, self.cube_orientation)
        if type(ee_ori_rcc) is Gf.Quatd:
            ee_ori_rcc = quatd_to_list4(ee_ori_rcc)

        # compute curobo solution:
        ik_goal = Pose(
            # position=tensor_args.to_device(ee_translation_goal),
            # quaternion=tensor_args.to_device(ee_orientation_teleop_goal),
            position=self.tensor_args.to_device(ee_pos_rcc),
            quaternion=self.tensor_args.to_device(ee_ori_rcc),
        )
        self.plan_config.pose_cost_metric = self.pose_metric
        # print("2: num_targets:", self.num_targets,
        #       "  cga:", self.constrain_grasp_approach,
        #       "  rpp:", self.reach_partial_pose,
        #       "  hpp:", self.hold_partial_pose
        #       )
        # print("2: pose_metric:", self.pose_metric)
        try:
            result = self.motion_gen.plan_single(self.cu_js.unsqueeze(0), ik_goal, self.plan_config)
        except Exception as e:
            print(f"Exception in motion_gen.plan_single e:{e}")
            result = self.motion_gen.plan_single(self.cu_js.unsqueeze(0), ik_goal, self.plan_config)
            return

        print("motion_gen.plan_single success:", result.success)
        # ik_result = ik_solver.solve_single(ik_goal, cu_js.position.view(1,-1), cu_js.position.view(1,1,-1))

        succ = result.success.item()  # ik_result.success.item()
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
        if succ:
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
            self.AssignCmdPlan(cmd_plan)

            self.cmd_idx = 0

        else:
            msg =  f"Plan did not converge to a solution. Status:{result.status}. No action is being taken."
            print(msg)
            carb.log_warn(msg)

        self.target_pose = self.cube_position
        self.target_orientation = self.cube_orientation

    def AssignCmdPlan(self, cmd_plan):
        print("assigned cmdplan len:", len(cmd_plan.position))
        self.cmd_plan = cmd_plan
        self.cmd_idx = 0

    def ExecuteMoGenCmdPlan(self):
        if self.cmd_plan is not None:
            print(f"Executing plan step {self.cmd_idx}/{len(self.cmd_plan.position)}")
            cmd_state = self.cmd_plan[self.cmd_idx]
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
            self.cmd_idx += 1
            for _ in range(2):
                self.my_world.step(render=False)
            if self.cmd_idx >= len(self.cmd_plan.position):
                self.cmd_idx = 0
                self.cmd_plan = None
                self.past_cmd = None
