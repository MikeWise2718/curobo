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
from omni.isaac.core.utils.viewports import set_camera_view

def setcamview(viewsel):
    if viewsel is None:
        return
    viewsel = viewsel.lower()
    match viewsel:
        case "top-view" | "t":
            set_camera_view(eye=[0.0, 0, 2.5], target=[0, 0, 0], camera_prim_path="/OmniverseKit_Persp")
        case "front-view" | "f":
            set_camera_view(eye=[-2.5, 0.0, 1.0], target=[0, 0, 0], camera_prim_path="/OmniverseKit_Persp")
        case "back-view" | "b":
            set_camera_view(eye=[2.5, 0.0, 1.0], target=[0, 0, 0], camera_prim_path="/OmniverseKit_Persp")
        case "left-view" | "l":
            set_camera_view(eye=[0.0, +2.5, 1.0], target=[0, 0, 0], camera_prim_path="/OmniverseKit_Persp")
        case "right-view" | "r":
            set_camera_view(eye=[0.0, -2.5, 1.0], target=[0, 0, 0], camera_prim_path="/OmniverseKit_Persp")


class KeyMan():
    def __init__(self, sim, rocurWrap1, rocurWrap2, motoman):
        self.sim = sim
        self.key = None
        self.rocurWrap1 = rocurWrap1
        self.rocurWwap2 = rocurWrap2
        self.mm = motoman
        self.zingle_step = False
        loop_start = time.time()
        self.w_pressed_time = loop_start
        self.p_pressed_time = loop_start
        self.o_pressed_time = loop_start
        self.po_pressed_time = loop_start
        self.traycmd = None
        self.traynum = "1"
        self.traycmdchar = "p"



    def ProcessKeys(self):

        if keyboard.is_pressed("a"):
            k = keyboard.read_key()
            self.rocuWrap1.ToggleCirclingTarget()
            if self.rocuWrap2 is not None:
                self.rocuWrap2.ToggleCirclingTarget()
            print(f"You pressed ‘a’. circle_target is now:{self.rocuWrap1.circle_target}")

        elif keyboard.is_pressed("b"):
            k = keyboard.read_key()
            if time.time() - self.w_pressed_time < 1.0:
                setcamview("back-view")
            print("You pressed ‘wb’ for back-view.")

        elif keyboard.is_pressed("*"):
            k = keyboard.read_key()
            self.rocuWrap1.curvel *= 1.5
            if self.rocuWrap2 is not None:
                self.rocuWrap2.curvel = self.rocuWrap1.curvel
            print(f"You pressed ‘*’. curvel:{self.rocuWrap1.curvel}")

        elif keyboard.is_pressed("/"):
            k = keyboard.read_key()
            self.rocuWrap1.curvel /= 1.5
            if self.rocuWrap2 is not None:
                self.rocuWrap2.curvel = self.rocuWrap1.curvel
            print(f"You pressed ‘/’. curvel:{self.rocuWrap1.curvel}")

        elif keyboard.is_pressed("c"):
            k = keyboard.read_key()
            print("You pressed ‘c’ - will reset object to start pose.")
            sp_rcc, sq_rcc = self.rocuWrap1.get_start_pose()  # this is the robots starting pose in rcc
            sp_wc, sq_wc = self.rocuWrap1.rcc_to_wc(sp_rcc, sq_rcc)
            self.rocuWrap1.SetTargetPose(sp_wc, sq_wc)
            if self.rocuWrap2 is not None:
                sp_rcc, sq_rcc = self.rocuWrap2.get_start_pose()
                sp_wc, sq_wc = self.rocuWrap2.rcc_to_wc(sp_rcc, sq_rcc)
                self.rocuWrap2.SetTargetPose(sp_wc, sq_wc)

        elif keyboard.is_pressed("d"):
            k = keyboard.read_key()
            print("You pressed ‘d’ - will move to robot's current end-effector pose.")
            if self.rocuWrap1.cu_js is not None:
                self.rocuWrap1.MoveTargetToEepose()
            if self.rocuWrap2 is not None:
                if self.rocuWrap2.cu_js is not None:
                    # sp_rcc, sq_rcc = self.rocuWrap2.get_cur_pose(self.rocuWrap2.cu_js)
                    self.rocuWrap2.MoveTargetToEepose()

        elif keyboard.is_pressed("e"):
            k = keyboard.read_key()
            if time.time() - self.w_pressed_time < 1.0:
                setcamview("front-view")
            print("You pressed ‘we’ for front-view.")

        elif keyboard.is_pressed("q"):
            k = keyboard.read_key()
            print("You pressed ‘q’ - will exit simulation.")
            self.sim.close()

        elif keyboard.is_pressed("l"):
            k = keyboard.read_key()
            if time.time() - self.w_pressed_time < 1.0:
                setcamview("left-view")
            print("You pressed ‘wf’ for left-view.")

        elif keyboard.is_pressed("m"):
            k = keyboard.read_key()
            self.rocuWrap1.toggle_material()
            self.rocuWrap1.check_alarm_status()
            if self.rocuWrap2 is not None:
                self.rocuWrap2.toggle_material()
                self.rocuWrap2.check_alarm_status()
            print("You pressed ‘m’")

        elif keyboard.is_pressed("t"):
            k = keyboard.read_key()
            if time.time() - self.w_pressed_time < 1.0:
                setcamview("top-view")
                print("You pressed ‘wt’ for top-view.")
            else:
                self.rocuWrap1.toggle_show_joints_close_to_limits()
                if self.rocuWrap2 is not None:
                    self.rocuWrap2.toggle_show_joints_close_to_limits()
                print("You pressed ‘t’ for show joints close to limits.")

        elif keyboard.is_pressed("p"):
            k = keyboard.read_key()
            self.p_pressed_time = time.time()
            self.op_pressed_time = time.time()
            self.traycmdchar = "p"
            print("You pressed ‘p’")

        elif keyboard.is_pressed("o"):
            k = keyboard.read_key()
            self.o_pressed_time = time.time()
            self.op_pressed_time = time.time()
            self.traycmdchar = "o"
            print("You pressed ‘o’")

        elif keyboard.is_pressed("r"):
            k = keyboard.read_key()
            if time.time() - self.w_pressed_time < 1.0:
                setcamview("right-view")
                print("You pressed ‘wr' for right-view.")
            else:
                self.rocuWrap1.ShowReachabilityGrid(clear=True)
                if self.rocuWrap2 is not None:
                    self.rocuWrap2.ShowReachabilityGrid(clear=False)
                print("You pressed ‘r’ - showing reachability")

        elif keyboard.is_pressed("w"):
            k = keyboard.read_key()
            self.w_pressed_time = time.time()
            print("You pressed ‘w’")

        # elif keyboard.is_pressed("v"):
        #     k = keyboard.read_key()
        #     self.rocuWrap1.vizi_spheres = not self.rocuWrap1.vizi_spheres
        #     if self.rocuWrap2 is not None:
        #         self.rocuWrap2.vizi_spheres = self.rocuWrap1.vizi_spheres
        #     print(f"You pressed 'v' - vizi_spheres is now {self.rocuWrap1.vizi_spheres}.")

        elif keyboard.is_pressed("z"):
            k = keyboard.read_key()
            self.zingle_step = not self.zingle_step
            print(f"You pressed 'z' - for zingle stepping - now {self.zingle_step}.")

        elif keyboard.is_pressed("0"):
            k = keyboard.read_key()
            print("You pressed ‘0’")
            if time.time() - self.op_pressed_time < 1.0:
                self.traycmd = f"{self.traycmdchar}_{self.traynum}_0"

        elif keyboard.is_pressed("1"):
            k = keyboard.read_key()
            if time.time() - self.op_pressed_time < 1.0:
                self.traycmd = f"{self.traycmdchar}_{self.traynum}_1"

        elif keyboard.is_pressed("2"):
            k = keyboard.read_key()
            if time.time() - self.self.o_pressed_time < 1.0:
                self.traycmd = f"{self.traycmdchar}_{self.traynum}_2"

        elif keyboard.is_pressed("3"):
            k = keyboard.read_key()
            if time.time() - self.op_pressed_time < 1.0:
                self.traycmd = f"{self.traycmdchar}_{self.traynum}_3"

        elif keyboard.is_pressed("4"):
            k = keyboard.read_key()
            if time.time() - self.op_pressed_time < 1.0:
                self.traycmd = f"{self.traycmdchar}_{self.traynum}_4"

        elif keyboard.is_pressed("5"):
            k = keyboard.read_key()
            if time.time() - self.op_pressed_time < 1.0:
                self.traycmd = f"{self.traycmdchar}_{self.traynum}_5"

        elif keyboard.is_pressed("6"):
            k = keyboard.read_key()
            if time.time() - self.op_pressed_time < 1.0:
                self.traycmd = f"{self.traycmdchar}_{self.traynum}_6"

    def TrayCommandReady(self):
        rv = False
        ok = self.traycmd is not None
        if ok:
            cmd, s_traynum, s_phone_num = self.traycmd.split("_")
            traynum = int(s_traynum)
            phone_num = int(s_phone_num)
            rv = cmd in ['p', 'o'] and (0 <= traynum < 4) and (0 <= phone_num < 6)
            if not rv:
                print(f"Bad traycmd:{self.traycmd}")
                self.traycmd = None
        return rv

    def GetTrayCommand(self):
        cmd, traynum, phone_num = self.traycmd.split("_")
        iphonenum = int(phone_num)
        itraynum = int(traynum)
        self.traycmd = None
        return cmd, itraynum, iphonenum