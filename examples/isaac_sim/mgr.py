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

# from pynput.keyboard import Key, Listener

# def on_press(key):
#     print('{0} pressed'.format(
#         key))

# def on_release(key):
#     print('{0} release'.format(
#         key))
#     if key == Key.esc:
#         # Stop listener
#         return False

# # Collect events until released
# with Listener(
#         on_press=on_press,
#         on_release=on_release) as listener:
#     listener.join()


a = torch.zeros(4, device="cuda:0")

# Standard Library
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--headless_mode",
    type=str,
    default=None,
    help="To run headless, use one of [native, websocket], webrtc might not work.",
)
parser.add_argument("--robot", type=str, default="franka.yml", help="robot configuration to load")
parser.add_argument(
    "--external_asset_path",
    type=str,
    default=None,
    help="Path to external assets when loading an externally located robot",
)
parser.add_argument(
    "--external_robot_configs_path",
    type=str,
    default=None,
    help="Path to external robot config when loading an external robot",
)
parser.add_argument(
    "-rbp",
    "--robpos",
    type=str,
    default=None,
    help="Robot Position - default = [0,0,0]",
)
parser.add_argument(
    "-rbo",
    "--robori",
    type=str,
    default=None,
    help="Robot Orientation - default = [1,0,0,0]",
)

parser.add_argument(
    "-vzs",
    "--visualize_spheres",
    action="store_true",
    help="When True, visualizes robot spheres",
    default=False,
)
parser.add_argument(
    "--reactive",
    action="store_true",
    help="When True, runs in reactive mode",
    default=False,
)

parser.add_argument(
    "--constrain_grasp_approach",
    action="store_true",
    help="When True, approaches grasp with fixed orientation and motion only along z axis.",
    default=False,
)

parser.add_argument(
    "--reach_partial_pose",
    nargs=6,
    metavar=("qx", "qy", "qz", "x", "y", "z"),
    help="Reach partial pose",
    type=float,
    default=None,
)
parser.add_argument(
    "--hold_partial_pose",
    nargs=6,
    metavar=("qx", "qy", "qz", "x", "y", "z"),
    help="Hold partial pose while moving to goal",
    type=float,
    default=None,
)


args = parser.parse_args()

############################################################
import sys
import os
exe_path = sys.executable
sys_path = sys.path
envv = ""
print("==== exepath ====")
print(sys.executable)
print("==== syspath ====")
for ppth in sys.path:
    print(ppth)
print("==== envh ====")
for name, value in os.environ.items():
    print("{0}: {1}".format(name, value))
    envv += f"{name}: {value}\n"

# Third Party
pass
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": args.headless_mode is not None,
        "width": "1920",
        "height": "1080",
    }
)

# from pynput.keyboard import Key, Listener

# def on_press(key):
#     print('{0} pressed'.format(
#         key))

# def on_release(key):
#     print('{0} release'.format(
#         key))
#     if key == Key.esc:
#         # Stop listener
#         return False

# # Collect events until released
# with Listener(
#         on_press=on_press,
#         on_release=on_release) as listener:
#     listener.join()

# Standard Library
from typing import Dict

# Third Party
import carb
import numpy as np
from helper import add_extensions, add_robot_to_scene
from omni.isaac.core import World
from omni.isaac.core.objects import cuboid, sphere

########### OV #################
from omni.isaac.core.utils.types import ArticulationAction

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

############################################################


########### OV #################;;;;;
def get_vek(s:str, default = [0,0,0]):
    if s is None:
        return default
    if s=="":
        return default
    if s[0]=="[":
        s = s[1:]
    if s[-1]=="]":
        s = s[:-1]
    sar = s.split(",")
    far = [float(x) for x in sar]
    return far


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

    tensor_args = TensorDeviceType()
    robot_cfg_path = get_robot_configs_path()
    if args.external_robot_configs_path is not None:
        robot_cfg_path = args.external_robot_configs_path
    print(f"robot_cfg_path: {robot_cfg_path}")
    print(f"args.robot: {args.robot}")
    if os.path.isfile(args.robot):
        robot_cfg = load_yaml(args.robot)["robot_cfg"]
    else:
        robot_cfg = load_yaml(join_path(robot_cfg_path, args.robot))["robot_cfg"]

    if args.external_asset_path is not None:
        robot_cfg["kinematics"]["external_asset_path"] = args.external_asset_path
    if args.external_robot_configs_path is not None:
        robot_cfg["kinematics"]["external_robot_configs_path"] = args.external_robot_configs_path
    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]

    robpos = np.array( get_vek(args.robpos) )
    print(f"robpos: {robpos}")
    robori1 = get_vek(args.robori)
    print(f"robori1: {robori1}")

    # robpos = np.array([0, 0, 0.1])
    robori = np.array([1, 0, 0, 0])

    robot, robot_prim_path = add_robot_to_scene(robot_cfg, my_world, position=robpos, orient = robori)

    articulation_controller = None

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
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
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
    motion_gen = MotionGen(motion_gen_config)
    print("warming up...")

    sp1, sq1 = motion_gen.get_start_pose()
    sp1 += robpos

    # Make a target to follow
    target = cuboid.VisualCuboid(
        "/World/target",
        position=np.array(sp1),
        orientation=np.array(sq1),
        color=np.array([1.0, 0, 0]),
        size=0.05,
    )
    # target = cuboid.VisualCuboid(
    #     "/World/target",
    #     position=np.array([-0.1492, -0.0395,  0.8978]),
    #     orientation=np.array([0.4906, 0.3388, 0.4896, 0.6363]),
    #     color=np.array([1.0, 0, 0]),
    #     size=0.05,
    # )
    # target = cuboid.VisualCuboid(
    #     "/World/target",
    #     position=np.array([-0.5, 0, 0.5]),
    #     orientation=np.array([1, 0, 0, 0]),
    #     color=np.array([1.0, 0, 0]),
    #     size=0.05,
    # )

    motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False, parallel_finetune=True)

    print("Curobo is Ready")

    add_extensions(simulation_app, args.headless_mode)

    plan_config = MotionGenPlanConfig(
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
    robot_static = True
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

    while simulation_app.is_running():

        if keyboard.is_pressed("a"):
            k = keyboard.read_key()
            print("You pressed ‘a’.")

        elif keyboard.is_pressed("b"):
            k = keyboard.read_key()
            print("You pressed ‘b’.")

        elif keyboard.is_pressed("c"):
            k = keyboard.read_key()
            print("You pressed ‘c’.")
            sp1, sq1 = motion_gen.get_start_pose()
            sp1 += robpos
            target.set_world_pose(position=sp1, orientation=sq1)

        elif keyboard.is_pressed("d"):
            k = keyboard.read_key()
            print("You pressed ‘d’.")
            # cu_js = JointState(
            #     position=tensor_args.to_device(sim_js.positions),
            #     velocity=tensor_args.to_device(sim_js.velocities),  # * 0.0,
            #     acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
            #     jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
            #     joint_names=sim_js_names,
            # )
            sp1, sq1 = motion_gen.get_cur_pose(cu_js)
            sp1 += robpos
            target.set_world_pose(position=sp1, orientation=sq1)

        elif keyboard.is_pressed("v"):
            k = keyboard.read_key()
            vizi_spheres = not vizi_spheres

        my_world.step(render=True)
        if not my_world.is_playing():
            if i % 100 == 0:
                print(f"**** Click Play to start simulation ***** si:{step_index}")
            i += 1
            # if step_index == 0:
            #    my_world.play()
            continue

        step_index = my_world.current_time_step_index
        if (step_index % 100) == 0:
            elap = time.time() - start
            cp = cube_position
            co = cube_orientation
            print(f"si:{step_index} time:{elap:.2f} velmag:{velmag:.2f} static:{robot_static} cp:{cp} co:{co}")
        if articulation_controller is None:
            # robot.initialize()
            articulation_controller = robot.get_articulation_controller()
        if step_index < 2:
            print(f"resetting world step:{step_index}")
            my_world.reset()
            robot._articulation_view.initialize()
            idx_list = [robot.get_dof_index(x) for x in j_names]
            robot.set_joint_positions(default_config, idx_list)

            robot._articulation_view.set_max_efforts(
                values=np.array([5000 for i in range(len(idx_list))]), joint_indices=idx_list
            )
        if step_index < 20:
            continue

        if step_index == 50 or step_index % 1000 == 0.0:
            print("Updating world, reading w.r.t.", robot_prim_path)
            obstacles = usd_help.get_obstacles_from_stage(
                # only_paths=[obstacles_path],
                reference_prim_path=robot_prim_path,
                ignore_substring=[
                    robot_prim_path,
                    "/World/target",
                    "/World/defaultGroundPlane",
                    "/curobo",
                ],
            ).get_collision_check_world()
            print(len(obstacles.objects))

            motion_gen.update_world(obstacles)
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
        if (step_index % 100) == 0:
            # print(f"   sim_js_names: {sim_js_names}")
            print(f"   sim_js.positions: {sim_js.positions}")
            # print(f"   sim_js.velocities: {sim_js.velocities}")


        if not args.reactive:
            cu_js.velocity *= 0.0
            cu_js.acceleration *= 0.0

        if args.reactive and past_cmd is not None:
            cu_js.position[:] = past_cmd.position
            cu_js.velocity[:] = past_cmd.velocity
            cu_js.acceleration[:] = past_cmd.acceleration
        cu_js = cu_js.get_ordered_joint_state(motion_gen.kinematics.joint_names)

        # if args. avisualize_spheresnd step_index % 2 == 0:
        if vizi_spheres and step_index % 2 == 0:
            sph_list = motion_gen.kinematics.get_robot_as_spheres(cu_js.position)

            config_spheres = robot_cfg["kinematics"]["collision_spheres"]
            keynames = list(config_spheres.keys())

            if spheres is None:
                spheres = []
                spherenames = []
                # create spheres:
                ncreated = 0
                for si, s in enumerate(sph_list[0]):
                    sname ="/curobo/robot_sphere_" + str(si)
                    clr = np.array([0, 0.8, 0.2])
                    if si < len(keynames):
                        key = keynames[si]
                        if key in config_spheres:
                            attdict = config_spheres[key][0]
                            if "scolor" in attdict:
                                clr = np.array(attdict["scolor"])
                    sp = sphere.VisualSphere(
                        prim_path=sname,
                        position=np.ravel(s.position),
                        radius=float(s.radius),
                        color=clr,
                    )
                    spheres.append(sp)
                    spherenames.append(sname)
                    ncreated += 1
                print(f"Created {ncreated} Spheres")
            else:
                for si, s in enumerate(sph_list[0]):
                    if not np.isnan(s.position[0]):
                        spheres[si].set_world_pose(position=np.ravel(s.position))
                        spheres[si].set_radius(float(s.radius))
            spheres_visable = True

        if not vizi_spheres and spheres_visable:
            if spherenames is not None:
                for sn in spherenames:
                    stage.RemovePrim(sn)
            spheres = None
            spherenames = None
            spheres_visable = False
            sph_list = None

        robot_static = True
        # robot_static = True
        velmag = np.max(np.abs(sim_js.velocities))
        if (velmag < 0.2) or args.reactive:
            # robot_static = True
            pass
        if (
            (
                np.linalg.norm(cube_position - target_pose) > 1e-3
                or np.linalg.norm(cube_orientation - target_orientation) > 1e-3
            )
            and np.linalg.norm(past_pose - cube_position) == 0.0
            and np.linalg.norm(past_orientation - cube_orientation) == 0.0
            and robot_static
        ):
            print("cube moved")
            # Set EE teleop goals, use cube for simple non-vr init:
            ee_translation_goal = cube_position - robpos
            ee_orientation_teleop_goal = cube_orientation

            # compute curobo solution:
            ik_goal = Pose(
                position=tensor_args.to_device(ee_translation_goal),
                quaternion=tensor_args.to_device(ee_orientation_teleop_goal),
            )
            plan_config.pose_cost_metric = pose_metric
            try:
                result = motion_gen.plan_single(cu_js.unsqueeze(0), ik_goal, plan_config)
            except Exception as e:
                print(f"Exception in motion_gen.plan_single e:{e}")

            print("motion_gen.plan_single success:", result.success)
            # ik_result = ik_solver.solve_single(ik_goal, cu_js.position.view(1,-1), cu_js.position.view(1,1,-1))

            succ = result.success.item()  # ik_result.success.item()
            if num_targets == 1:
                if args.constrain_grasp_approach:
                    pose_metric = PoseCostMetric.create_grasp_approach_metric()
                if args.reach_partial_pose is not None:
                    reach_vec = motion_gen.tensor_args.to_device(args.reach_partial_pose)
                    pose_metric = PoseCostMetric(
                        reach_partial_pose=True, reach_vec_weight=reach_vec
                    )
                if args.hold_partial_pose is not None:
                    hold_vec = motion_gen.tensor_args.to_device(args.hold_partial_pose)
                    pose_metric = PoseCostMetric(hold_partial_pose=True, hold_vec_weight=hold_vec)
            if succ:
                num_targets += 1
                cmd_plan = result.get_interpolated_plan()
                cmd_plan = motion_gen.get_full_js(cmd_plan)
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
