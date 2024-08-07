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
from RobotCuroboWrap import RobotCuroboWrapper



def main():
#---------------------------------
#    Misc Initialization
#---------------------------------
    args = get_args()

    add_extensions(simulation_app, args.headless_mode)

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
    n_obstacle_cuboids = 30
    n_obstacle_mesh = 100

    usd_help = UsdHelper()
    target_pose = None
    # tensor_args = TensorDeviceType()

    # convoluted way to get the robot config path
    robot_cfg_path = get_robot_configs_path()
    if args.external_robot_configs_path is not None:
        robot_cfg_path = args.external_robot_configs_path
    if os.path.isfile(args.robot):
        robot_cfg_path = args.robot
    else:
        robot_cfg_path = join_path(robot_cfg_path, args.robot)


#---------------------------------
#    Robot Initialization
#---------------------------------
    robwrap: RobotCuroboWrapper = RobotCuroboWrapper()

    prerot1 = get_vek(args.robprerot)
    pos1 = get_vek(args.robpos)
    ori1 = get_vek(args.robori)
    assetpath = args.external_asset_path
    configpath = args.external_robot_configs_path
    robwrap.Initialize(robot_cfg_path, assetpath, configpath, my_world)
    robwrap.SetMoGenOptions(reactive=args.reactive,
                            reach_partial_pose=args.reach_partial_pose,
                            hold_partial_pose=args.hold_partial_pose,
                            constrain_grasp_approach=args.constrain_grasp_approach,
                            vizi_spheres=args.visualize_spheres)
    robwrap.PositionRobot(prerot1, pos1, ori1)


#---------------------------------
#    World Initialization
#---------------------------------
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


#---------------------------------
#    Motion Gen Initialization
#---------------------------------

    robwrap.InitMotionGen( n_obstacle_cuboids, n_obstacle_mesh, world_cfg)


#-------------------------------------------------------
#    Post Motion Gen Initialization World initiailzation
#-------------------------------------------------------
    # Make a target to follow
    sp_rcc, sq_rcc = robwrap.motion_gen.get_start_pose()
    sp_wc, sq_wc = robwrap.tranman.rcc_to_wc(sp_rcc, sq_rcc)
    if type(sq_wc) is Gf.Quatd:
        sq_wc = quatd_to_list4(sq_wc)

    target = cuboid.VisualCuboid(
        "/World/target",
        position=sp_wc,
        orientation=sq_wc,
        color=np.array([1.0, 0, 0]),
        size=0.05,
    )

#---------------------------------
#    Warming up motion gen
#---------------------------------
    robwrap.Warmup()
    robwrap.SetupMoGenPlanConfig()

#---------------------------------
#    loop initialization
#---------------------------------


    usd_help.load_stage(my_world.stage)
    usd_help.add_world_to_stage(world_cfg, base_frame="/World")

    cmd_plan = None
    cmd_idx = 0
    my_world.scene.add_default_ground_plane()
    i_step = 0
    past_cmd = None
    pose_metric = None
    step_index = 0
    velmag = 0.0
    start = time.time()
    static_robo = True
    cube_position, cube_orientation = target.get_world_pose()
    # articulation_controller = robwrap.robot.get_articulation_controller()
    circle_target = False
    curcen, curori = target.get_world_pose()
    curvel = 0.1
    curang = 0
    currad = 0.02

    # Front view
    set_camera_view(eye=[0.0, 2.5, 1.0], target=[0,0,0], camera_prim_path="/OmniverseKit_Persp")
    # Overhead view
    # set_camera_view(eye=[0.0, 0, 4.0], target=[0,0,0], camera_prim_path="/OmniverseKit_Persp")
#---------------------------------
#    LOOP
#---------------------------------
    while simulation_app.is_running():

#---------------------------------
#    Inpot Processing
#---------------------------------

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
            sp_rcc, sq_rcc = robwrap.motion_gen.get_start_pose() # this is the robots starting position in rcc
            if robwrap is not None:
                sp_wc, sq_wc = robwrap.tranman.rcc_to_wc(sp_rcc, sq_rcc)
                if type(sq_wc) is Gf.Quatd:
                    sq_wc = quatd_to_list4(sq_wc)
            target.set_world_pose(position=sp_wc, orientation=sq_wc)

        elif keyboard.is_pressed("d"):
            k = keyboard.read_key()
            print("You pressed ‘d’ - will move to robot's current end-effector pose.")
            if robwrap.cu_js is not None:
                sp_rcc, sq_rcc = robwrap.motion_gen.get_cur_pose(robwrap.cu_js)
                sp_wc, sq_wc = robwrap.tranman.rcc_to_wc(sp_rcc, sq_rcc)
                target.set_world_pose(position=sp_wc, orientation=sq_wc)

        elif keyboard.is_pressed("e"):
            k = keyboard.read_key()
            robwrap.tranman.dump_robot_transforms(robwrap.robot_prim_path)

        elif keyboard.is_pressed("q"):
            k = keyboard.read_key()
            break

        elif keyboard.is_pressed("s"):
            k = keyboard.read_key()
            print("You pressed ‘s’.")

        elif keyboard.is_pressed("v"):
            k = keyboard.read_key()
            robwrap.vizi_spheres = not robwrap.vizi_spheres
            print(f"You pressed 'v' - vizi_spheres is now {robwrap.vizi_spheres}.")


#---------------------------------
#    World Processing
#---------------------------------

        my_world.step(render=True)
        if not my_world.is_playing():
            if i_step % 500 == 0:
                print(f"**** Click Play to start simulation ***** si:{step_index}")
            i_step += 1
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
            robwrap.Reset()

#---------------------------------
#    Obstacles Processing
#---------------------------------

        if step_index == 50 or step_index % 1000 == 0:
            print("Updating world, reading w.r.t.", robwrap.robot_prim_path)
            obstacles = usd_help.get_obstacles_from_stage(
                # only_paths=[obstacles_path],
                reference_prim_path=robwrap.robot_prim_path,
                ignore_substring=[
                    robwrap.robot_prim_path,
                    "/World/target",
                    "/World/defaultGroundPlane",
                    "/curobo",
                ],
            ).get_collision_check_world()
            print(f"obstacle objects:{len(obstacles.objects)}")

            robwrap.motion_gen.update_world(obstacles)
            print("Updated World")
            carb.log_info("Synced CuRobo world from stage.")

        # position and orientation of target virtual cube:
        cube_position, cube_orientation = target.get_world_pose()

#---------------------------------
#    Robot Processing
#---------------------------------
        robwrap.UpdateJointState(past_cmd)

        robwrap.HandleCollisionSpheres()

        trigger = robwrap.calc_trigger(cube_position, cube_orientation, circle_target, cmd_plan)

#--------------------------------------
#    Robot Motion Planning
#---------------------------------------
        if trigger:
            print("cube moved")
            # Set EE teleop goals, use cube for simple non-vr init:
            ee_pos_rcc, ee_ori_rcc = robwrap.tranman.wc_to_rcc(cube_position, cube_orientation)
            if type(ee_ori_rcc) is Gf.Quatd:
                ee_ori_rcc = quatd_to_list4(ee_ori_rcc)

            # compute curobo solution:
            ik_goal = Pose(
                # position=tensor_args.to_device(ee_translation_goal),
                # quaternion=tensor_args.to_device(ee_orientation_teleop_goal),
                position=robwrap.tensor_args.to_device(ee_pos_rcc),
                quaternion=robwrap.tensor_args.to_device(ee_ori_rcc),
            )
            robwrap.plan_config.pose_cost_metric = pose_metric
            try:
                result = robwrap.motion_gen.plan_single(robwrap.cu_js.unsqueeze(0), ik_goal, robwrap.plan_config)
            except Exception as e:
                print(f"Exception in motion_gen.plan_single e:{e}")

            print("motion_gen.plan_single success:", result.success)
            # ik_result = ik_solver.solve_single(ik_goal, cu_js.position.view(1,-1), cu_js.position.view(1,1,-1))

            succ = result.success.item()  # ik_result.success.item()
            if num_targets == 1:
                if args.constrain_grasp_approach:
                    pose_metric = PoseCostMetric.create_grasp_approach_metric()
                if args.reach_partial_pose is not None:
                    reach_vec = robwrap.motion_gen.tensor_args.to_device(args.reach_partial_pose)
                    pose_metric = PoseCostMetric(
                        reach_partial_pose=True, reach_vec_weight=reach_vec
                    )
                if args.hold_partial_pose is not None:
                    hold_vec = robwrap.motion_gen.tensor_args.to_device(args.hold_partial_pose)
                    pose_metric = PoseCostMetric(hold_partial_pose=True, hold_vec_weight=hold_vec)
            if succ:
                num_targets += 1
                cmd_plan = result.get_interpolated_plan()
                cmd_plan = robwrap.motion_gen.get_full_js(cmd_plan)
                print(f"Plan Success with {len(cmd_plan.position)} steps")
                # get only joint names that are in both:
                idx_list = []
                common_js_names = []
                for x in robwrap.sim_js_names:
                    if x in cmd_plan.joint_names:
                        idx_list.append(robwrap.robot.get_dof_index(x))
                        common_js_names.append(x)
                # idx_list = [robot.get_dof_index(x) for x in sim_js_names]

                cmd_plan = cmd_plan.get_ordered_joint_state(common_js_names)

                cmd_idx = 0

            else:
                msg =  f"Plan did not converge to a solution. Status:{result.status}. No action is being taken."
                print(msg)
                carb.log_warn(msg)
            robwrap.target_pose = cube_position
            robwrap.target_orientation = cube_orientation
        robwrap.past_pose = cube_position
        robwrap.past_orientation = cube_orientation

#--------------------------------------
#    Robot Command Step Execution
#---------------------------------------

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
            #  articulation_controller.apply_action(art_action)
            robwrap.ApplyAction(art_action)
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
