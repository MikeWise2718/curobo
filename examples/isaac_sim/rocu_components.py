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
import numpy as np
from enum import Enum

from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

from pxr import Gf, Sdf, Usd, UsdGeom
from rotations import euler_angles_to_quat, matrix_to_euler_angles, rot_matrix_to_quat
from omni.isaac.core.utils.stage import get_current_stage
############################################################
import torch

# ########## OV #################;;;;;
import typing

from mgr_ut import get_args, print_mat, list4to_quatd

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

    def gfvec_to_numpy(self, vek):
        lst = [vek[0], vek[1], vek[2]]
        arr_np = np.array(lst, dtype=np.float32)
        return arr_np

    def quat_to_numpy(self, q):
        if hasattr(q, "cpu"):
            rv = q.cpu().numpy()
        else:
            lst = [q[0], q[1], q[2], q[3]]
            rv = np.array(lst, dtype=np.float32)
        return rv

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
