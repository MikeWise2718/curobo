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
from helper import add_extensions
from omni.isaac.core import World

########### OV #################
from omni.isaac.core.utils.viewports import set_camera_view

# CuRobo
# from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.geom.types import WorldConfig
from curobo.util.logger import log_error, setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import (
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
)


from mgrut import get_args, get_vek
from rocuwrap import RocuWrapper


def main():
    # ---------------------------------
    #    Misc Initialization
    # ---------------------------------
    args = get_args()

    add_extensions(simulation_app, args.headless_mode)

    # assuming obstacles are in objects_path:
    my_world = World(stage_units_in_meters=0.05)
    stage = my_world.stage

    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")
    stage = my_world.stage

    setup_curobo_logger("warn")
    n_obstacle_cuboids = 30
    n_obstacle_mesh = 100

    usd_help = UsdHelper()

    # ---------------------------------
    #    World Initialization
    # ---------------------------------
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

    # ---------------------------------
    #    Robot Initialization
    # ---------------------------------
    # convoluted way to get the robot config path
    robot_cfg_path = get_robot_configs_path()
    if args.external_robot_configs_path is not None:
        robot_cfg_path = args.external_robot_configs_path
    if os.path.isfile(args.robot):
        robot_cfg_path = args.robot
    else:
        robot_cfg_path = join_path(robot_cfg_path, args.robot)
    assetpath = args.external_asset_path
    configpath = args.external_robot_configs_path

    prerot1 = get_vek(args.robprerot)
    pos1 = get_vek(args.robpos)
    ori1 = get_vek(args.robori)
    numrobs = 1

    match args.jakas:
        case "R":
            prerot1, pos1, ori1 = [0, 0, 60], [-0.05, 0, 1], [0, 150, 180]
            numrobs = 1
        case "L":
            prerot1, pos1, ori1 = [0, 0, -90], [+0.05, 0, 1], [0, -150, 180]
            numrobs = 1
        case "LR":
            prerot1, pos1, ori1 = [0, 0, -90], [+0.05, 0, 1], [0, -150, 180]
            prerot2, pos2, ori2 = [0, 0, +60], [-0.05, 0, 1], [0, +150, 180]
            numrobs = 2
        case "RL":
            prerot1, pos1, ori1 = [0, 0, +60], [-0.05, 0, 1], [0, +150, 180]
            prerot2, pos2, ori2 = [0, 0, -90], [+0.05, 0, 1], [0, -150, 180]
            numrobs = 2
        case _:
            print("Bad Jaka specification")

    def defineRobot(prerot, pos, ori, robid="") -> RocuWrapper:
        rw: RocuWrapper = RocuWrapper(robid)
        rw.Initialize(robot_cfg_path, assetpath, configpath, my_world)
        rw.SetMoGenOptions(reactive=args.reactive,
                           reach_partial_pose=args.reach_partial_pose,
                           hold_partial_pose=args.hold_partial_pose,
                           constrain_grasp_approach=args.constrain_grasp_approach,
                           vizi_spheres=args.visualize_spheres)
        rw.LoadAndPositionRobot(prerot, pos, ori, subroot=robid)
        rw.InitMotionGen(n_obstacle_cuboids, n_obstacle_mesh, world_cfg)
        rw.Warmup()
        rw.SetupMoGenPlanConfig()
        rw.CreateTarget()
        return rw

    rocuWrap1: RocuWrapper = defineRobot(prerot1, pos1, ori1, "1")

    if numrobs == 2:
        rocuWrap2: RocuWrapper = defineRobot(prerot2, pos2, ori2, "2")
    else:
        rocuWrap2 = None

    usd_help.load_stage(my_world.stage)
    usd_help.add_world_to_stage(world_cfg, base_frame="/World")

    my_world.scene.add_default_ground_plane()
    step_index = 0

    last_play_time = 0
    loop_start = time.time()

    # Front view
    set_camera_view(eye=[0.0, 2.5, 1.0], target=[0, 0, 0], camera_prim_path="/OmniverseKit_Persp")
    # Overhead view
    # set_camera_view(eye=[0.0, 0, 4.0], target=[0, 0, 0], camera_prim_path="/OmniverseKit_Persp")

    # ---------------------------------
    #    LOOP
    # ---------------------------------
    while simulation_app.is_running():

        # ---------------------------------
        #    Keyboard Input Processing
        # ---------------------------------

        if keyboard.is_pressed("a"):
            k = keyboard.read_key()
            rocuWrap1.circle_target = not rocuWrap1.circle_target
            print(f"You pressed ‘a’. circle_target is now:{rocuWrap1.circle_target}")

        elif keyboard.is_pressed("b"):
            k = keyboard.read_key()
            print("You pressed ‘b’.")

        elif keyboard.is_pressed("*"):
            k = keyboard.read_key()
            rocuWrap1.curvel *= 1.5
            print(f"You pressed ‘*’. curvel:{rocuWrap1.curvel}")

        elif keyboard.is_pressed("/"):
            k = keyboard.read_key()
            rocuWrap1.curvel /= 1.5
            print(f"You pressed ‘/’. curvel:{rocuWrap1.curvel}")

        elif keyboard.is_pressed("c"):
            k = keyboard.read_key()
            print("You pressed ‘c’ - will reset object to start pose.")
            sp_rcc, sq_rcc = rocuWrap1.get_start_pose()  # this is the robots starting position in rcc
            sp_wc, sq_wc = rocuWrap1.rcc_to_wc(sp_rcc, sq_rcc)
            rocuWrap1.SetTargetPose(sp_wc, sq_wc)

        elif keyboard.is_pressed("d"):
            k = keyboard.read_key()
            print("You pressed ‘d’ - will move to robot's current end-effector pose.")
            if rocuWrap1.cu_js is not None:
                sp_rcc, sq_rcc = rocuWrap1.get_cur_pose(rocuWrap1.cu_js)
                sp_wc, sq_wc = rocuWrap1.rcc_to_wc(sp_rcc, sq_rcc)
                rocuWrap1.SetTargetPose(sp_wc, sq_wc)

        elif keyboard.is_pressed("q"):
            k = keyboard.read_key()
            break

        elif keyboard.is_pressed("s"):
            k = keyboard.read_key()
            print("You pressed ‘s’.")

        elif keyboard.is_pressed("v"):
            k = keyboard.read_key()
            rocuWrap1.vizi_spheres = not rocuWrap1.vizi_spheres
            print(f"You pressed 'v' - vizi_spheres is now {rocuWrap1.vizi_spheres}.")

        # ---------------------------------
        #    World Processing
        # ---------------------------------

        my_world.step(render=True)
        step_index = my_world.current_time_step_index
        if not my_world.is_playing():
            elap = time.time() - last_play_time
            if elap > 5:
                print(f"**** Click Play to start simulation ***** si:{step_index} elap:{elap:.2f}")
                last_play_time = time.time()
            continue

        rocuWrap1.CircleTarget()

        if (step_index % 100) == 0:
            elap = time.time() - loop_start
            cp,co = rocuWrap1.GetTargetPose()
            print(f"si:{step_index} time:{elap:.2f} cp:{cp} co:{co}")
        if step_index < 2:
            print(f"resetting world step:{step_index}")
            my_world.reset()
            rocuWrap1.Reset()
            if rocuWrap2 is not None:
                rocuWrap2.Reset()

        # ---------------------------------
        #    Obstacles Processing
        # ---------------------------------

        if step_index == 50 or step_index % 1000 == 0:
            print("Updating world, reading w.r.t.", rocuWrap1.robot_prim_path)
            obstacles = usd_help.get_obstacles_from_stage(
                # only_paths=[obstacles_path],
                reference_prim_path=rocuWrap1.robot_prim_path,
                ignore_substring=[
                    rocuWrap1.robot_prim_path,
                    "/World/target",
                    "/World/defaultGroundPlane",
                    "/curobo",
                ],
            ).get_collision_check_world()
            print(f"obstacle objects:{len(obstacles.objects)}")

            rocuWrap1.update_world(obstacles)
            print("Updated World")
            carb.log_info("Synced CuRobo world from stage.")

        # position and orientation of target virtual cube:

        rocuWrap1.UpdateJointState()
        rocuWrap1.ProcessCollisionSpheres()
        rocuWrap1.HandleTargetProcessing()
        rocuWrap1.ExecuteMoGenCmdPlan()

        if rocuWrap2 is not None:
            rocuWrap2.UpdateJointState()
            rocuWrap2.ProcessCollisionSpheres()
            rocuWrap2.HandleTargetProcessing()
            rocuWrap2.ExecuteMoGenCmdPlan()

    simulation_app.close()

if __name__ == "__main__":
    main()
