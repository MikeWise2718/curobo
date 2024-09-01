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
import numpy as np
from enum import Enum

# CuRobo
# from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.types.math import Pose
# from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import (
    PoseCostMetric,
)

from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

from pxr import Gf, Sdf, Usd, UsdGeom

############################################################
import torch

from rocu_components import RocuConfiguator, RocuMoveMode, RocuTranMan
from mgr_ut import get_args, get_vek, print_mat, list4to_quatd, quatd_to_list4, get_sphere_entry
from omni.isaac.debug_draw import _debug_draw

args = get_args()


RocuConfig = RocuConfiguator()


class GridRenderStyle(Enum):
    DEBUG_SPHERES = 1
    OV_SPHERES = 2

class GridRenderFilter(Enum):
    ALL = 1
    SUCCESS_ONLY = 2
    FAIL_ONLY = 3


class ReachGridMan():
    def __init__(self, robot):
        self.robot = robot
        self.tensor_args = robot.tensor_args
        self.tranman = robot.rocuTranman
        self.ik_config_grid: IKSolverConfig = None
        self.ik_solver_grid: IKSolver = None
        self.n_x, self.n_y, self.n_z = 9, 9, 9
        self.max_x, self.max_y, self.max_z = 0.5, 0.5, 0.5
        self.grid_succ_rad = 40
        self.grid_fail_rad = 20
        self.count_unique_solutions = True
        self.pose_metric = PoseCostMetric()
        self.num_targets = 0
        self.reachability_grid_visible = False
        self.grid_render_style = GridRenderStyle.DEBUG_SPHERES
        self.grid_render_filter = GridRenderFilter.ALL

    def InitSolver(self, n_obstacle_cuboids, n_obstacle_mesh):
        self.n_obstacle_cuboids = n_obstacle_cuboids
        self.n_obstacle_mesh = n_obstacle_mesh
        self.ik_config_grid = IKSolverConfig.load_from_robot_config(
            self.robot.robot_cfg,
            self.robot.world_cfg,
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
        self.ik_solver_grid = IKSolver(self.ik_config_grid)

    def InitGridSize(self, n_x, n_y, n_z, max_x, max_y, max_z, grid_succ_rad=40, grid_fail_rad=20):
        self.n_x, self.n_y, self.n_z = n_x, n_y, n_z
        self.max_x, self.max_y, self.max_z = max_x, max_y, max_z
        self.grid_succ_rad = grid_succ_rad
        self.grid_fail_rad = grid_fail_rad
        self._calc_pose_grid()

    def SetGridSize(self, n_x=9, n_y=9, n_z=9):
        self.n_x, self.n_y, self.n_z = n_x, n_y, n_z
        self._calc_pose_grid()

    def SetGridSpan(self, max_x=0.5, max_y=0.5, max_z=0.5):
        self.max_x, self.max_y, self.max_z = max_x, max_y, max_z
        self._calc_pose_grid()

    def _calc_pose_grid(self):
        x = np.linspace(-self.max_x, self.max_x, self.n_x)
        y = np.linspace(-self.max_y, self.max_y, self.n_y)
        z = np.linspace(-self.max_z, self.max_z, self.n_z)
        x, y, z = np.meshgrid(x, y, z, indexing="ij")

        self.position_arr = np.zeros((self.n_x * self.n_y * self.n_z, 3))
        self.position_arr[:, 0] = x.flatten()
        self.position_arr[:, 1] = y.flatten()
        self.position_arr[:, 2] = z.flatten()

    def InitPositionGridOffset(self):
        pos = self.ik_solver_grid.get_retract_config().view(1, -1)
        fk_state = self.ik_solver_grid.fk(pos)
        self.InitPositionGridOffsetElems(fk_state.ee_pose)

    def InitPositionGridOffsetElems(self, ee_pose):
        # self.gpr = self._calc_pose_grid()
        # self.position_grid_offset = self.tensor_args.to_device(self.gpr)
        self._calc_pose_grid()
        self.position_grid_offset = self.tensor_args.to_device(self.position_arr)
        self.goal_pose = ee_pose
        self.goal_pose = self.goal_pose.repeat(self.position_grid_offset.shape[0])
        self.goal_pose.position += self.position_grid_offset

    def CalcReachabilityToPosOri(self, pos, ori):
        ee_pos_rcc, ee_ori_rcc = self.tranman.wc_to_rcc(pos, ori)
        if type(ee_ori_rcc) is Gf.Quatd:
            ee_ori_rcc = quatd_to_list4(ee_ori_rcc)

        # compute curobo solution:
        print("ik_goal-p:", ee_pos_rcc, " q:", ee_ori_rcc)
        ik_goal = Pose(
            position=self.tensor_args.to_device(ee_pos_rcc),
            quaternion=self.tensor_args.to_device(ee_ori_rcc),
        )

        self.goal_pose.position[:] = ik_goal.position[:] + self.position_grid_offset
        self.goal_pose.quaternion[:] = ik_goal.quaternion[:]

        st_time = time.time()
        ik_result = self.ik_solver_grid.solve_batch(self.goal_pose)
        ntrue = torch.sum(ik_result.success).item()
        ntry = len(ik_result.success)

        elap = (time.time() - st_time)
        selap = ik_result.solve_time
        msg = f"IK completed: Poses: {str(self.goal_pose.batch)}  Success: {ntrue} of {ntry}  Solve time(s): {selap:.3f}  elap:{elap:3f}"
        print(msg)

        return ik_result

    def CalcPoseGrid(self, joint_posistions, needAsNumpy=False):
        fkstate = self.ik_solver_grid.fk(joint_posistions)
        fk_pose = fkstate.ee_pose
        sp_rcc, sq_rcc = fk_pose.position.squeeze(), fk_pose.quaternion.squeeze()
        sp_wc, sq_wc = self.tranman.rcc_to_wc(sp_rcc, sq_rcc)
        if needAsNumpy:
            sp_wc_np = self.tranman.gfvec_to_numpy(sp_wc)
            sq_wc_np = self.tranman.quat_to_numpy(sq_wc)
            return sp_wc_np, sq_wc_np
        return sp_wc, sq_wc

    def ToggleDebugSphereMode(self):
        if self.grid_render_style == GridRenderStyle.DEBUG_SPHERES:
            self.grid_render_style = GridRenderStyle.OV_SPHERES
        else:
            self.grid_render_style = GridRenderStyle.DEBUG_SPHERES
        print(f"Grid render style: {self.grid_render_style}")

    def RotateGridFilter(self):
        match self.grid_render_filter:
            case GridRenderFilter.ALL:
                self.grid_render_filter = GridRenderFilter.SUCCESS_ONLY
            case GridRenderFilter.SUCCESS_ONLY:
                self.grid_render_filter = GridRenderFilter.FAIL_ONLY
            case GridRenderFilter.FAIL_ONLY:
                self.grid_render_filter = GridRenderFilter.ALL
        print(f"Grid render filter: {self.grid_render_filter}")

    def ChangeGridSize(self, fak = 1.5):
        show = self.reachability_grid_visible
        if show:
            self.ClearReachabilityGrid()


        # we need to reinitialize the solvers since the cuda graph has changed - which takes awhile
        self.InitSolver(self.n_obstacle_cuboids, self.n_obstacle_mesh)
        self.InitPositionGridOffset()

        self.ik_solver_grid = IKSolver(self.ik_config_grid)
        self.n_x = int(self.n_x*fak)
        self.n_y = int(self.n_y*fak)
        self.n_z = int(self.n_z*fak)
        print(f"New grid size: {self.n_x}x{self.n_y}x{self.n_z} for a total of {self.n_x*self.n_y*self.n_z} poses")
        self.SetGridSize(self.n_x, self.n_y, self.n_z)
        self.InitPositionGridOffset()
        if show:
            self.ShowReachabilityGridToPosOri(self.last_pos, self.last_ori)

    def ChangeGridSpan(self, fak = 1.5):
        show = self.reachability_grid_visible
        if show:
            self.ClearReachabilityGrid()
        self.max_x = self.max_x*fak
        self.max_y = self.max_y*fak
        self.max_z = self.max_z*fak
        self.SetGridSpan(self.max_x, self.max_y, self.max_z)
        self.InitPositionGridOffset()
        if show:
            self.ShowReachabilityGridToPosOri(self.last_pos, self.last_ori)

    def ShowReachabilityGridToPosOri(self, pos, ori, clear=True):
        self.last_pos = pos
        self.last_ori = ori
        res = self.CalcReachabilityToPosOri(pos, ori)
        if self.count_unique_solutions:
            unique = res.get_batch_unique_solution()
        else:
            unique = None
        self._show_reachability_grid(self.goal_pose, res.success, unique=unique, clear=clear)



    def ClearReachabilityGrid(self):
        draw = _debug_draw.acquire_debug_draw_interface()
        draw.clear_points()
        self.reachability_grid_visible  = False

    def _show_reachability_grid(self, pose, success, unique=None, clear=True):
        self.reachability_grid_visible  = True
        self.cur_pose = pose
        self.cur_success = success
        self.cur_unique = unique

        draw = _debug_draw.acquire_debug_draw_interface()
        if clear:
            draw.clear_points()
        cpu_pos = pose.position.cpu().numpy()
        b, _ = cpu_pos.shape
        point_list = []
        colors = []
        sizes = []
        multi_color =  (0, 0, 1, 0.25)
        single_sucess_color = (0, 1, 0, 0.25)
        error_color = (1, 0, 1, 0.25)
        fail_color = (1, 0, 0, 0.25)
        succ_rad = self.grid_succ_rad
        if self.grid_render_filter == GridRenderFilter.FAIL_ONLY:
            succ_rad = 0
        fail_rad = self.grid_fail_rad
        if self.grid_render_filter == GridRenderFilter.SUCCESS_ONLY:
            fail_rad = 0
        color_on_solution_number = self.count_unique_solutions and unique is not None
        for i in range(b):
            # get list of points:
            # point_list += [(cpu_pos[i, 0], cpu_pos[i, 1], cpu_pos[i, 2])]
            pt_rcc = (cpu_pos[i, 0], cpu_pos[i, 1], cpu_pos[i, 2])
            pt_wc, qt_wc = self.tranman.rcc_to_wc(pt_rcc, [1, 0, 0, 0])
            point_list += [pt_wc]
            if unique is not None:
                nsol = len(unique[i])
            else:
                nsol = 1
            if success[i].item():
                if color_on_solution_number:
                    if nsol == 0:
                        clr = error_color
                    elif nsol == 1:
                        clr = single_sucess_color
                    else:
                        clr = multi_color
                else:
                    clr = single_sucess_color
                colors += [clr]
                sizes += [succ_rad]
            else:
                colors += [fail_color]
                sizes += [fail_rad]
        # sizes = [40.0 for _ in range(b)]
        if self.grid_render_style==GridRenderStyle.DEBUG_SPHERES:
            draw.draw_points(point_list, colors, sizes)
        else:
            self.RenderOvSpheres(point_list, colors, sizes)

    def RenderOvSpheres(self, point_list, colors, sizes):
        pass

    def UpdateWorldObsticles(self, obstacles):
        if self.ik_solver_grid is not None:
            self.ik_solver_grid.update_world(obstacles)
