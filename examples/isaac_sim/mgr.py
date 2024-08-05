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
import numpy.linalg.lapack_lite
import torch
import time
import os
import keyboard


a = torch.zeros(4, device="cuda:0")

# Third Party
from omni.isaac.kit import SimulationApp

# many omnivrse things can't be imported until after SimulationApp is created

simulation_app = SimulationApp(
    {
        "headless": False,
        "width": "1920",
        "height": "1080",
    }
)


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
        (t,r,s,m) = self.get_world_transform_xform_full( robprim )
        print("world_tranlate:", t)
        print("world_rotate:", r)
        print_mat("world_transform:", 4, 4, m)
        print_mat("rotmat3d:", 3, 3, self.rotmat3d)
        print_mat("inv_rotmat3d:", 3, 3, self.inv_rotmat3d)


class RoboDeco:

    def __init__(self):
        self.tranman = TranMan()

    def AssignRobot(self, robot, robot_usd_prim_path):
        self.robot = robot
        self.robot_prim_path = robot_usd_prim_path
        self.robot.deco = self
        self.articulation_controller = robot.get_articulation_controller()

    def load_robot_cfg(self, robot_pathname):
        self.robot_cfg = load_yaml(robot_pathname)["robot_cfg"]


def main():
    # create a curobo motion gen instance:
    num_targets = 0
    # assuming obstacles are in objects_path:
    my_world = World(stage_units_in_meters=0.05)
    stage = my_world.stage

    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")
    # my_world.stage.SetDefaultPrim(my_world.stage.GetPrimAtPath("/World"))
    stage = my_world.stage
    # stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

    setup_curobo_logger("warn")
    past_pose = None
    n_obstacle_cuboids = 30
    n_obstacle_mesh = 100

    # warmup curobo instance
    usd_help = UsdHelper()
    target_pose = None

    deco : RoboDeco = RoboDeco()

    tensor_args = TensorDeviceType()

    # convoluted way to get the robot config path
    robot_cfg_path = get_robot_configs_path()
    if args.external_robot_configs_path is not None:
        robot_cfg_path = args.external_robot_configs_path
    if os.path.isfile(args.robot):
        # robot_cfg = load_yaml(args.robot)["robot_cfg"]
        robot_cfg_path = args.robot
    else:
        robot_cfg_path = join_path(robot_cfg_path, args.robot)
        # robot_cfg = load_yaml(join_path(robot_cfg_path, args.robot))["robot_cfg"]

    # robot_cfg = load_yaml(robot_cfg_path)["robot_cfg"]
    deco.load_robot_cfg(robot_cfg_path)

    if args.external_asset_path is not None:
        deco.robot_cfg["kinematics"]["external_asset_path"] = args.external_asset_path
    if args.external_robot_configs_path is not None:
        deco.robot_cfg["kinematics"]["external_robot_configs_path"] = args.external_robot_configs_path
    j_names = deco.robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = deco.robot_cfg["kinematics"]["cspace"]["retract_config"]


    # deco = RoboDeco()
    deco.tranman.set_transform(prerot=get_vek(args.robprerot), pos=get_vek(args.robpos), ori=get_vek(args.robori))
    rp, ro = deco.tranman.get_robot_base()

    robot, robot_prim_path = add_robot_to_scene(deco.robot_cfg, my_world, position=rp, orient=ro)
    deco.AssignRobot(robot, robot_prim_path)

    world_cfg_table = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    )
    world_cfg_table.cuboid[0].pose[2] -= 0.02
    world_cfg1 = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    ).get_mesh_world()
    world_cfg1.mesh[0].name += "_mesh"
    world_cfg1.mesh[0].pose[2] = -10.5

    world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid, mesh=world_cfg1.mesh)

    trajopt_dt = None
    optimize_dt = True
    trajopt_tsteps = 16
    trim_steps = None
    max_attempts = 4
    interpolation_dt = 0.05
    if args.reactive:
        trajopt_tsteps = 40
        trajopt_dt = 0.04
        optimize_dt = False
        max_attempts = 1
        trim_steps = [1, None]
        interpolation_dt = trajopt_dt
    robot.deco.motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot.deco.robot_cfg,
        world_cfg,
        tensor_args,
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
    robot.deco.motion_gen = MotionGen(robot.deco.motion_gen_config)
    print("warming up...")

    sp_rcc, sq_rcc = robot.deco.motion_gen.get_start_pose()
    sp_wc, sq_wc = robot.deco.tranman.rcc_to_wc(sp_rcc, sq_rcc)
    if type(sq_wc) is Gf.Quatd:
        sq_wc = quatd_to_list4(sq_wc)

    # Make a target to follow

    target = cuboid.VisualCuboid(
        "/World/target",
        position=sp_wc,
        orientation=sq_wc,
        color=np.array([1.0, 0, 0]),
        size=0.05,
    )

    robot.deco.motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False, parallel_finetune=True)

    print("Curobo is Ready and Warmed-up")

    add_extensions(simulation_app, args.headless_mode)

    robot.deco.plan_config = MotionGenPlanConfig(
        enable_graph=False,
        enable_graph_attempt=2,
        max_attempts=max_attempts,
        enable_finetune_trajopt=True,
        parallel_finetune=True,
    )

    usd_help.load_stage(my_world.stage)
    usd_help.add_world_to_stage(world_cfg, base_frame="/World")

    cmd_plan = None
    cmd_idx = 0
    my_world.scene.add_default_ground_plane()
    i = 0
    spheres = None
    past_cmd = None
    target_orientation = None
    past_orientation = None
    pose_metric = None
    step_index = 0
    velmag = 0.0
    start = time.time()
    static_robo = True
    cube_position, cube_orientation = target.get_world_pose()
    articulation_controller = robot.get_articulation_controller()
    sim_js = robot.get_joints_state()
    sim_js_names = robot.dof_names
    cu_js = None
    vizi_spheres = args.visualize_spheres
    spheres_visable = False
    spherenames = None
    spheres = None
    sph_list = None
    circle_target = False
    curcen, curori = target.get_world_pose()
    curvel = 0.1
    curang = 0
    currad = 0.02

    # Front view
    set_camera_view(eye=[0.0, 2.5, 1.0], target=[0,0,0], camera_prim_path="/OmniverseKit_Persp")
    # Overhead view
    # set_camera_view(eye=[0.0, 0, 4.0], target=[0,0,0], camera_prim_path="/OmniverseKit_Persp")

    while simulation_app.is_running():

        if keyboard.is_pressed("a"):
            k = keyboard.read_key()
            circle_target = not circle_target
            if circle_target:
                curcen, curori = target.get_world_pose()
                curvel = 0.02
                curang = 0
                currad = 0.1
            print(f"You pressed ‘a’. circle_target:{circle_target} curcen:{curcen} curvel:{curvel}")

        elif keyboard.is_pressed("b"):
            k = keyboard.read_key()
            print("You pressed ‘b’.")

        elif keyboard.is_pressed("*"):
            k = keyboard.read_key()
            curvel *= 1.5
            print("You pressed ‘*’. curvel:{curvel}")

        elif keyboard.is_pressed("/"):
            k = keyboard.read_key()
            curvel /= 1.5
            print("You pressed ‘/’. curvel:{curvel}")

        elif keyboard.is_pressed("c"):
            k = keyboard.read_key()
            print("You pressed ‘c’ - will reset object to start pose.")
            sp_rcc, sq_rcc = robot.deco.motion_gen.get_start_pose() # this is the robots starting position in rcc
            if robot.deco is not None:
                sp_wc, sq_wc = robot.deco.tranman.rcc_to_wc(sp_rcc, sq_rcc)
                if type(sq_wc) is Gf.Quatd:
                    sq_wc = quatd_to_list4(sq_wc)
            target.set_world_pose(position=sp_wc, orientation=sq_wc)

        elif keyboard.is_pressed("d"):
            k = keyboard.read_key()
            print("You pressed ‘d’ - will move to robot's current end-effector pose.")
            if cu_js is not None:
                sp_rcc, sq_rcc = robot.deco.motion_gen.get_cur_pose(cu_js)
                sp_wc, sq_wc = robot.deco.tranman.rcc_to_wc(sp_rcc, sq_rcc)
                target.set_world_pose(position=sp_wc, orientation=sq_wc)

        elif keyboard.is_pressed("e"):
            k = keyboard.read_key()
            robot.deco.dump_robot_transforms(robot.deco.robot_prim_path)

        elif keyboard.is_pressed("q"):
            k = keyboard.read_key()
            break

        elif keyboard.is_pressed("s"):
            k = keyboard.read_key()
            print("You pressed ‘s’.")

        elif keyboard.is_pressed("v"):
            k = keyboard.read_key()
            vizi_spheres = not vizi_spheres

        my_world.step(render=True)
        if not my_world.is_playing():
            if i % 500 == 0:
                print(f"**** Click Play to start simulation ***** si:{step_index}")
            i += 1
            continue

        if circle_target:
            dang = curvel
            curang += dang
            newpos = np.zeros(3)
            newpos[0] = curcen[0]  + currad*np.cos(curang)
            newpos[1] = curcen[1]  + currad*np.sin(curang)
            newpos[2] = curcen[2]
            target.set_world_pose(
                position=newpos,
                orientation=curori
            )

        step_index = my_world.current_time_step_index
        if (step_index % 100) == 0:
            elap = time.time() - start
            cp = cube_position
            co = cube_orientation
            print(f"si:{step_index} time:{elap:.2f} velmag:{velmag:.2f} static:{static_robo} cp:{cp} co:{co}")
        if step_index < 2:
            print(f"resetting world step:{step_index}")
            my_world.reset()
            robot._articulation_view.initialize()
            idx_list = [robot.get_dof_index(x) for x in j_names]
            robot.set_joint_positions(default_config, idx_list)

            robot._articulation_view.set_max_efforts(
                values=np.array([5000 for i in range(len(idx_list))]), joint_indices=idx_list
            )

        if step_index == 50 or step_index % 1000 == 0.0:
            print("Updating world, reading w.r.t.", robot.deco.robot_prim_path)
            obstacles = usd_help.get_obstacles_from_stage(
                # only_paths=[obstacles_path],
                reference_prim_path=robot.deco.robot_prim_path,
                ignore_substring=[
                    robot.deco.robot_prim_path,
                    "/World/target",
                    "/World/defaultGroundPlane",
                    "/curobo",
                ],
            ).get_collision_check_world()
            print(len(obstacles.objects))

            robot.deco.motion_gen.update_world(obstacles)
            print("Updated World")
            carb.log_info("Synced CuRobo world from stage.")

        # position and orientation of target virtual cube:
        cube_position, cube_orientation = target.get_world_pose()

        if past_pose is None:
            past_pose = cube_position
        if target_pose is None:
            target_pose = cube_position
        if target_orientation is None:
            target_orientation = cube_orientation
        if past_orientation is None:
            past_orientation = cube_orientation

        sim_js = robot.get_joints_state()
        sim_js_names = robot.dof_names
        if np.any(np.isnan(sim_js.positions)):
            print("isaac sim has returned NAN joint position values.")
            log_error("isaac sim has returned NAN joint position values.")
        cu_js = JointState(
            position=tensor_args.to_device(sim_js.positions),
            velocity=tensor_args.to_device(sim_js.velocities),  # * 0.0,
            acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=sim_js_names,
        )
        # if (step_index % 100) == 0:
            # print(f"   sim_js_names: {sim_js_names}")
            # print(f"   sim_js.positions: {sim_js.positions}")
            # print(f"   sim_js.velocities: {sim_js.velocities}")

        if not args.reactive:
            cu_js.velocity *= 0.0
            cu_js.acceleration *= 0.0

        if args.reactive and past_cmd is not None:
            cu_js.position[:] = past_cmd.position
            cu_js.velocity[:] = past_cmd.velocity
            cu_js.acceleration[:] = past_cmd.acceleration
        cu_js = cu_js.get_ordered_joint_state(robot.deco.motion_gen.kinematics.joint_names)

        # if args. avisualize_spheresnd step_index % 2 == 0:
        if vizi_spheres and step_index % 2 == 0:
            sph_list = robot.deco.motion_gen.kinematics.get_robot_as_spheres(cu_js.position)

            config_spheres = robot.deco.robot_cfg["kinematics"]["collision_spheres"]

            if spheres is None:
                spheres = []
                spherenames = []
                # create spheres:
                ncreated = 0
                for sidx, s in enumerate(sph_list[0]):
                    sphentry = get_sphere_entry(config_spheres, sidx)
                    sname ="/curobo/robot_sphere_" + str(sidx)
                    clr = np.array([0, 0.8, 0.2])
                    if sphentry is not None:
                        keyname = sphentry["keyname"]
                        keyidx = sphentry["keyidx"]
                        linkname = f"/curobo/{keyname}"
                        if not stage.GetPrimAtPath(linkname):
                            _ = stage.DefinePrim(linkname, "Xform")
                        sname = f"{linkname}/sphere_{keyidx}"
                        if "scolor" in sphentry:
                            clr = np.array(sphentry["scolor"])
                    s_ori = np.array([1, 0, 0, 0])
                    sp_wc, _ = robot.deco.tranman.rcc_to_wc(s.position, s_ori)
                    sp = sphere.VisualSphere(
                        prim_path=sname,
                        position=np.ravel(sp_wc),
                        radius=float(s.radius),
                        color=clr,
                    )
                    spheres.append(sp)
                    spherenames.append(sname)
                    ncreated += 1
                print(f"Created {ncreated} Spheres")
            else:
                for sidx, s in enumerate(sph_list[0]):
                    if not np.isnan(s.position[0]):
                        s_ori = np.array([1, 0, 0, 0])
                        sp_wc, _ = robot.deco.tranman.rcc_to_wc(s.position, s_ori)
                        spheres[sidx].set_world_pose(position=np.ravel(sp_wc))
                        spheres[sidx].set_radius(float(s.radius))
            spheres_visable = True

        if not vizi_spheres and spheres_visable:
            if spherenames is not None:
                for sn in spherenames:
                    stage.RemovePrim(sn)
            spheres = None
            spherenames = None
            spheres_visable = False
            sph_list = None

        static_robo = True
        # robot_static = True
        velmag = np.max(np.abs(sim_js.velocities))
        if (velmag < 0.2) or args.reactive:
            # robot_static = True
            pass
        pretrig = np.linalg.norm(cube_position - target_pose) > 1e-3 or np.linalg.norm(cube_orientation - target_orientation) > 1e-3
        trigger =  pretrig and np.linalg.norm(past_pose - cube_position) == 0.0 and np.linalg.norm(past_orientation - cube_orientation) == 0.0 and static_robo
        # print(f"trigger:{trigger} pretrig:{pretrig} velmag:{velmag:.2f} static_robo:{static_robo}")

        if circle_target:
            trigger = cmd_plan is None

        if trigger:
            print("cube moved")
            # Set EE teleop goals, use cube for simple non-vr init:
            ee_pos_rcc, ee_ori_rcc = robot.deco.tranman.wc_to_rcc(cube_position, cube_orientation)
            if type(ee_ori_rcc) is Gf.Quatd:
                ee_ori_rcc = quatd_to_list4(ee_ori_rcc)

            # compute curobo solution:
            ik_goal = Pose(
                # position=tensor_args.to_device(ee_translation_goal),
                # quaternion=tensor_args.to_device(ee_orientation_teleop_goal),
                position=tensor_args.to_device(ee_pos_rcc),
                quaternion=tensor_args.to_device(ee_ori_rcc),
            )
            robot.deco.plan_config.pose_cost_metric = pose_metric
            try:
                result = robot.deco.motion_gen.plan_single(cu_js.unsqueeze(0), ik_goal, robot.deco.plan_config)
            except Exception as e:
                print(f"Exception in motion_gen.plan_single e:{e}")

            print("motion_gen.plan_single success:", result.success)
            # ik_result = ik_solver.solve_single(ik_goal, cu_js.position.view(1,-1), cu_js.position.view(1,1,-1))

            succ = result.success.item()  # ik_result.success.item()
            if num_targets == 1:
                if args.constrain_grasp_approach:
                    pose_metric = PoseCostMetric.create_grasp_approach_metric()
                if args.reach_partial_pose is not None:
                    reach_vec = robot.deco.motion_gen.tensor_args.to_device(args.reach_partial_pose)
                    pose_metric = PoseCostMetric(
                        reach_partial_pose=True, reach_vec_weight=reach_vec
                    )
                if args.hold_partial_pose is not None:
                    hold_vec = robot.deco.motion_gen.tensor_args.to_device(args.hold_partial_pose)
                    pose_metric = PoseCostMetric(hold_partial_pose=True, hold_vec_weight=hold_vec)
            if succ:
                num_targets += 1
                cmd_plan = result.get_interpolated_plan()
                cmd_plan = robot.deco.motion_gen.get_full_js(cmd_plan)
                print(f"Plan Success with {len(cmd_plan.position)} steps")
                # get only joint names that are in both:
                idx_list = []
                common_js_names = []
                for x in sim_js_names:
                    if x in cmd_plan.joint_names:
                        idx_list.append(robot.get_dof_index(x))
                        common_js_names.append(x)
                # idx_list = [robot.get_dof_index(x) for x in sim_js_names]

                cmd_plan = cmd_plan.get_ordered_joint_state(common_js_names)

                cmd_idx = 0

            else:
                msg =  f"Plan did not converge to a solution. Status:{result.status}. No action is being taken."
                print(msg)
                carb.log_warn(msg)
            target_pose = cube_position
            target_orientation = cube_orientation
        past_pose = cube_position
        past_orientation = cube_orientation
        if cmd_plan is not None:
            print(f"Executing plan step {cmd_idx}/{len(cmd_plan.position)}")
            cmd_state = cmd_plan[cmd_idx]
            past_cmd = cmd_state.clone()
            # get full dof state
            art_action = ArticulationAction(
                cmd_state.position.cpu().numpy(),
                cmd_state.velocity.cpu().numpy(),
                joint_indices=idx_list,
            )
            # set desired joint angles obtained from IK:
            # print(f"Applying action: {art_action}")
            articulation_controller.apply_action(art_action)
            cmd_idx += 1
            for _ in range(2):
                my_world.step(render=False)
            if cmd_idx >= len(cmd_plan.position):
                cmd_idx = 0
                cmd_plan = None
                past_cmd = None
    simulation_app.close()

if __name__ == "__main__":
    main()
