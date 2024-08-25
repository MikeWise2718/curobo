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
import omni

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
from rocuwrap import RocuWrapper, RocuMoveMode, RocuConfig
import curobo.curobolib as curobolib
from matman import MatMan
from motomod import MotoMan

def DefineTrays(stage, matman):

    mm = MotoMan(stage, matman)
    # cage
    # mm.AddCage()

    a90 = np.pi/2

    # moto_50mp
    # mm.AddMoto50mp("moto1",rot=[-a90,0,a90],pos=[0,0,0.1])
    # mm.AddMoto50mp("moto2",rot=[-a90,0,a90],pos=[0.1,0.1,0.1])

    # moto_tray
    zang = 5*np.pi/4
    zang = np.pi/2
    zang = 0
    xoff = 0.20
    yoff = 0.15

    mm.AddMotoTray("tray1", "rgb000", rot=[a90,0,zang],pos=[+xoff,+yoff,0.03])
    mm.AddMotoTray("tray2", "000000", rot=[a90,0,zang],pos=[-xoff,+yoff,0.03])
    mm.AddMotoTray("tray3", "myc000", rot=[a90,0,zang],pos=[-xoff,-yoff,0.03])
    mm.AddMotoTray("tray4", "000000", rot=[a90,0,zang],pos=[+xoff,-yoff,0.03])
    return mm



def main():
    # ---------------------------------
    #    Misc Initialization
    # ---------------------------------
    print("curobolib dir")
    print(dir(curobolib))
    args = get_args()

    add_extensions(simulation_app, args.headless_mode)

    # assuming obstacles are in objects_path:
    my_world = World(stage_units_in_meters=0.05)
    stage = my_world.stage
    matman = MatMan(stage)

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

    # Setup RocuConfig
    RocuConfig.external_asset_path = assetpath
    RocuConfig.external_robot_configs_path = configpath
    RocuConfig.robot_config_path = robot_cfg_path
    RocuConfig.my_world = my_world
    RocuConfig.world_cfg = world_cfg
    RocuConfig.matman = matman

    # Robot Positionining
    prerot1 = get_vek(args.robprerot)
    pos1 = get_vek(args.robpos)
    ori1 = get_vek(args.robori)
    numrobs = 1

    match args.jakas:
        case "R":
            prerot1, pos1, ori1 = [0, 0, 60], [-0.2, 0, 1], [0, 150, 180]
            numrobs = 1
        case "L":
            prerot1, pos1, ori1 = [0, 0, -90], [+0.2, 0, 1], [0, -150, 180]
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

    # Setup Robots

    # mode = RocuMoveMode.FollowTargetWithMoGen
    modesel = args.movmode
    if modesel is not None:
        modesel = modesel.lower()
    mode = RocuMoveMode.FollowTargetWithMoGen
    if modesel in ["ikv", "ikt", "inkt", "inv"]:
        mode = RocuMoveMode.FollowTargetWithInvKin
    elif modesel in ["mg","mo","mog", "mogen"]:
        mode = RocuMoveMode.FollowTargetWithMoGen
    elif modesel in ["r", "re", "rea", "reach"]:
        mode = RocuMoveMode.ReachabilityWithInvKin
    reactive = args.reactive
    reach_pp = args.reach_partial_pose
    hold_pp = args.hold_partial_pose
    con_grasp = args.constrain_grasp_approach

    rocuWrap1: RocuWrapper = RocuWrapper("1")
    rocuWrap1.LoadAndPositionRobot(prerot1, pos1, ori1)
    rocuWrap1.SetRobotMoveMode(mode, reactive, reach_pp, hold_pp, con_grasp, n_obstacle_cuboids, n_obstacle_mesh)

    if numrobs == 2:
        rocuWrap2: RocuWrapper = RocuWrapper("2")
        rocuWrap2.LoadAndPositionRobot(prerot2, pos2, ori2)
        rocuWrap2.SetRobotMoveMode(mode, reactive, reach_pp, hold_pp, con_grasp, n_obstacle_cuboids, n_obstacle_mesh)
    else:
        rocuWrap2 = None

    # Setup Trays and Phones
    if args.add_trays:
        mm = DefineTrays(stage, matman)
    else:
        mm = None

    # Setup Grid
    if args.ngrid is not None:
        ng = args.ngrid
        rocuWrap1.SetGridSize(ng, ng, ng)
        if rocuWrap2 is not None:
            rocuWrap2.SetGridSize(ng, ng, ng)

    if args.gridspan is not None:
        span = args.gridspan
        rocuWrap1.SetGridSpan(span, span, span)
        if rocuWrap2 is not None:
            rocuWrap2.SetGridSpan(span, span, span)

    if args.gridsuccessrad is not None:
        rad = args.gridsuccessrad
        rocuWrap1.grid_succ_rad = rad
        if rocuWrap2 is not None:
            rocuWrap2.grid_succ_rad = rad

    if args.gridfailrad is not None:
        rad = args.gridfailrad
        rocuWrap1.grid_fail_rad = rad
        if rocuWrap2 is not None:
            rocuWrap2.grid_fail_rad = rad

    if args.gridtimerticks is not None:
        rocuWrap1.SetGridTimerTicks(args.gridtimerticks)
        if rocuWrap2 is not None:
            rocuWrap2.SetGridTimerTicks(args.gridtimerticks)

    usd_help.load_stage(my_world.stage)
    usd_help.add_world_to_stage(world_cfg, base_frame="/World")

    my_world.scene.add_default_ground_plane()
    step_index = 0

    last_play_time = 0
    loop_start = time.time()

    # Front view
    viewsel = args.view
    if viewsel in ["L", "l"]:
        set_camera_view(eye=[0.0, +2.5, 1.0], target=[0, 0, 0], camera_prim_path="/OmniverseKit_Persp")
    if viewsel in ["R", "r"]:
        set_camera_view(eye=[0.0, -2.5, 1.0], target=[0, 0, 0], camera_prim_path="/OmniverseKit_Persp")
    if viewsel in ["F", "f"]:
        set_camera_view(eye=[+2.5, 0.0, 1.0], target=[0, 0, 0], camera_prim_path="/OmniverseKit_Persp")
    if viewsel in ["B", "f"]:
        set_camera_view(eye=[-2.5, 0.0, 1.0], target=[0, 0, 0], camera_prim_path="/OmniverseKit_Persp")
    elif viewsel in ["T", "t"]:
        set_camera_view(eye=[0.0, 0, 2.5], target=[0, 0, 0], camera_prim_path="/OmniverseKit_Persp")

    timeline = omni.timeline.get_timeline_interface()
    zingle_step = False
            # self.timeline.play()
    # ---------------------------------
    #    LOOP
    # ---------------------------------
    while simulation_app.is_running():

        # ---------------------------------
        #    Keyboard Input Processing
        # ---------------------------------
        dokeyboard = step_index > 10

        if dokeyboard:
            if keyboard.is_pressed("a"):
                k = keyboard.read_key()
                rocuWrap1.ToggleCirclingTarget()
                if rocuWrap2 is not None:
                    rocuWrap2.ToggleCirclingTarget()
                print(f"You pressed ‘a’. circle_target is now:{rocuWrap1.circle_target}")

            elif keyboard.is_pressed("b"):
                k = keyboard.read_key()
                print("You pressed ‘b’.")

            elif keyboard.is_pressed("*"):
                k = keyboard.read_key()
                rocuWrap1.curvel *= 1.5
                if rocuWrap2 is not None:
                    rocuWrap2.curvel = rocuWrap1.curvel
                print(f"You pressed ‘*’. curvel:{rocuWrap1.curvel}")

            elif keyboard.is_pressed("/"):
                k = keyboard.read_key()
                rocuWrap1.curvel /= 1.5
                if rocuWrap2 is not None:
                    rocuWrap2.curvel = rocuWrap1.curvel
                print(f"You pressed ‘/’. curvel:{rocuWrap1.curvel}")

            elif keyboard.is_pressed("c"):
                k = keyboard.read_key()
                print("You pressed ‘c’ - will reset object to start pose.")
                sp_rcc, sq_rcc = rocuWrap1.get_start_pose()  # this is the robots starting pose in rcc
                sp_wc, sq_wc = rocuWrap1.rcc_to_wc(sp_rcc, sq_rcc)
                rocuWrap1.SetTargetPose(sp_wc, sq_wc)
                if rocuWrap2 is not None:
                    sp_rcc, sq_rcc = rocuWrap2.get_start_pose()
                    sp_wc, sq_wc = rocuWrap2.rcc_to_wc(sp_rcc, sq_rcc)
                    rocuWrap2.SetTargetPose(sp_wc, sq_wc)

            elif keyboard.is_pressed("d"):
                k = keyboard.read_key()
                print("You pressed ‘d’ - will move to robot's current end-effector pose.")
                if rocuWrap1.cu_js is not None:
                    # sp_rcc, sq_rcc = rocuWrap1.get_cur_pose(rocuWrap1.cu_js)
                    sp_rcc, sq_rcc = rocuWrap1.get_cur_pose()
                    sp_wc, sq_wc = rocuWrap1.rcc_to_wc(sp_rcc, sq_rcc)
                    rocuWrap1.SetTargetPose(sp_wc, sq_wc)
                if rocuWrap2 is not None:
                    if rocuWrap2.cu_js is not None:
                        # sp_rcc, sq_rcc = rocuWrap2.get_cur_pose(rocuWrap2.cu_js)
                        sp_rcc, sq_rcc = rocuWrap2.get_cur_pose()
                        sp_wc, sq_wc = rocuWrap2.rcc_to_wc(sp_rcc, sq_rcc)
                        rocuWrap2.SetTargetPose(sp_wc, sq_wc)

            elif keyboard.is_pressed("e"):
                k = keyboard.read_key()
                rocuWrap1.dump_robot_transforms()

            elif keyboard.is_pressed("q"):
                k = keyboard.read_key()
                break

            elif keyboard.is_pressed("m"):
                k = keyboard.read_key()
                rocuWrap1.toggle_material()
                rocuWrap1.check_alarm_status()
                if rocuWrap2 is not None:
                    rocuWrap2.toggle_material()
                    rocuWrap2.check_alarm_status()
                print("You pressed ‘m’")

            elif keyboard.is_pressed("t"):
                k = keyboard.read_key()
                rocuWrap1.toggle_show_joints_close_to_limits()
                if rocuWrap2 is not None:
                    rocuWrap2.toggle_show_joints_close_to_limits()
                print("You pressed ‘t’")

            elif keyboard.is_pressed("r"):
                k = keyboard.read_key()
                rocuWrap1.ShowReachability(clear=True)
                if rocuWrap2 is not None:
                    rocuWrap2.ShowReachability(clear=False)
                print("You pressed ‘r’ - showing reachability")

            elif keyboard.is_pressed("v"):
                k = keyboard.read_key()
                rocuWrap1.vizi_spheres = not rocuWrap1.vizi_spheres
                if rocuWrap2 is not None:
                    rocuWrap2.vizi_spheres = rocuWrap1.vizi_spheres
                print(f"You pressed 'v' - vizi_spheres is now {rocuWrap1.vizi_spheres}.")

            elif keyboard.is_pressed("z"):
                k = keyboard.read_key()
                zingle_step = not zingle_step
                print(f"You pressed 'z' - for zingle stepping - now {zingle_step}.")

        # ---------------------------------
        #    World Processing
        # ---------------------------------

        my_world.step(render=True)
        step_index = my_world.current_time_step_index
        if not my_world.is_playing():
            elap = time.time() - last_play_time
            if elap > 60:
                print(f"**** Click Play to start simulation ***** si:{step_index} elap:{elap:.2f}")
                last_play_time = time.time()
            continue

        if step_index == 1: # step index will be set to 1 when play is pressed, so we should reset only
            print(f"resetting world step:{step_index}")
            my_world.reset()

        rocuWrap1.StartStep(step_index)
        if rocuWrap2 is not None:
            rocuWrap2.StartStep(step_index)

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
            if rocuWrap2 is not None:
                rocuWrap2.update_world(obstacles)
            print("Updated World")
            carb.log_info("Synced CuRobo world from stage.")

        # position and orientation of target virtual cube:

        if zingle_step:
            print("Pauseing timeline - zingle stepping")
            timeline.pause()

        rocuWrap1.EndStep()
        if rocuWrap2 is not None:
            rocuWrap2.EndStep()

    simulation_app.close()

if __name__ == "__main__":
    main()
