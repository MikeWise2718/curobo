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
import omni.kit.hotkeys.core

from rocuwrap import RocuWrapper

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
        case "diag-view" | "d":
            set_camera_view(eye=[1.7, +1.7, 1.0], target=[0, 0, 0], camera_prim_path="/OmniverseKit_Persp")
        case "right-view" | "r":
            set_camera_view(eye=[0.0, -2.5, 1.0], target=[0, 0, 0], camera_prim_path="/OmniverseKit_Persp")


class KeyMan():
    def __init__(self, sim):
        self.sim = sim
        self.key = None
        self.zingle_step = False
        loop_start = time.time()
        self.a_pressed_time = loop_start
        self.c_pressed_time = loop_start
        self.o_pressed_time = loop_start
        self.p_pressed_time = loop_start
        self.r_pressed_time = loop_start
        self.s_pressed_time = loop_start
        self.t_pressed_time = loop_start
        self.w_pressed_time = loop_start
        self.op_pressed_time = loop_start
        self.traycmd = None
        self.traynum = 1
        self.traycmdchar = "p"
        self.DeRegisterAllHotKeys()
        self.rocuWrap1: RocuWrapper = None
        self.rocuWrap2: RocuWrapper = None
        self.curRocu: RocuWrapper = None
        self.mm = None

    def AddObjects(self, rocuWrap1, rocuWrap2, motoman):
        self.rocuWrap1 = rocuWrap1
        self.rocuWrap2 = rocuWrap2
        self.curRocu = rocuWrap1
        self.mm = motoman

    def PrintHelp(self):
        msgs = []
        msgs.append("mgr Keyboard Commands")
        indent1 = "  "
        indent2 = indent1 + indent1
        msgs.append(indent1+"General Cp,,amds")
        msgs.append(indent2+"? - help")
        msgs.append(indent2+"q - quit")
        msgs.append(indent2+"z - zingle stepping")
        msgs.append(indent1+"Target-cube manipulation")
        msgs.append(indent2+"ct - circle target")
        msgs.append(indent2+"* - increase target-cube velocity")
        msgs.append(indent2+"/ - decrease target-cube velocity")
        msgs.append(indent2+"c - reset target-cube to start pose")
        msgs.append(indent2+"d - reset target-cube to current end-effector pose")
        msgs.append(indent2+"pN - move target to the phone postion N")
        msgs.append(indent2+"oN - move target to 20 cm above the phone postion N")
        msgs.append(indent1+"View commands")
        msgs.append(indent2+"wt - top view")
        msgs.append(indent2+"wr - right view")
        msgs.append(indent2+"wl - left view")
        msgs.append(indent2+"wb - back view")
        msgs.append(indent2+"wfq - front view")
        msgs.append(indent1+"Robot commands")
        msgs.append(indent2+"aN - make robot arm N to active robot")
        msgs.append(indent2+"m - toggle material")
        msgs.append(indent2+"al - show joints close to limits")
        msgs.append(indent2+"tN - make tray N the active tray")
        for msg in msgs:
            print(msg)
        self.DumpHotKeys()


    def ProcessKeys(self):

        now = time.time()
        if keyboard.is_pressed("a"):
            k = keyboard.read_key()
            self.a_pressed_time = now
            # self.rocuWrap1.ToggleCirclingTarget()
            # if self.rocuWrap2 is not None:
            #     self.rocuWrap2.ToggleCirclingTarget()
            print(f"You pressed ‘a’.")

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
            if k=="?":
                print(f"You pressed ‘?’. - getting help  l:{k}")
                self.PrintHelp()
            else:
                self.rocuWrap1.curvel /= 1.5
                if self.rocuWrap2 is not None:
                    self.rocuWrap2.curvel = self.rocuWrap1.curvel
                print(f"You pressed ‘/’. curvel:{self.rocuWrap1.curvel}   l:{k}")

        elif keyboard.is_pressed("c"):
            k = keyboard.read_key()
            if now - self.r_pressed_time < 1.0:
                self.curRocu.ClearReachabilityGrid()
                print("You pressed ‘rc’ - clearing reachability")
            else:
                self.c_pressed_time = now
                # print("You pressed ‘c’ - will reset object to start pose.")
                # sp_rcc, sq_rcc = self.rocuWrap1.get_start_pose()  # this is the robots starting pose in rcc
                # sp_wc, sq_wc = self.rocuWrap1.rcc_to_wc(sp_rcc, sq_rcc)
                # self.curRocu.SetTargetPose(sp_wc, sq_wc)

        elif keyboard.is_pressed("d"):
            k = keyboard.read_key()
            if now - self.w_pressed_time < 1.0:
                setcamview("diag-view")
            else:
                print("You pressed ‘d’ - will move to robot's current end-effector pose.")
                if self.curRocu.cu_js is not None:
                    self.curRocu.MoveTargetToEepose()

        elif keyboard.is_pressed("f"):
            k = keyboard.read_key()
            if now - self.w_pressed_time < 1.0:
                setcamview("front-view")
                print("You pressed ‘wf’ for front-view.")


        elif keyboard.is_pressed("l"):
            k = keyboard.read_key()
            if now - self.w_pressed_time < 1.0:
                setcamview("left-view")
            print("You pressed ‘wf’ for left-view.")

        elif keyboard.is_pressed("m"):
            k = keyboard.read_key()
            self.curRocu.toggle_material()
            self.curRocu.check_alarm_status()
            print("You pressed ‘m’")

        elif keyboard.is_pressed("t"):
            k = keyboard.read_key()
            if now - self.w_pressed_time < 1.0:
                setcamview("top-view")
                print("You pressed ‘wt’ for top-view.")
            elif now - self.a_pressed_time < 1.0:
                self.curRocu.ToggleCirclingTarget()
                print("You pressed ‘at’ for rotate target.")
            else:
                self.t_pressed_time = now
                # self.rocuWrap1.toggle_show_joints_close_to_limits()
                # if self.rocuWrap2 is not None:
                #     self.rocuWrap2.toggle_show_joints_close_to_limits()
                # print("You pressed ‘t’ for show joints close to limits.")

        elif keyboard.is_pressed("p"):
            k = keyboard.read_key()
            self.p_pressed_time = now
            self.op_pressed_time = now
            self.traycmdchar = "p"
            print("You pressed ‘p’")

        elif keyboard.is_pressed("o"):
            k = keyboard.read_key()
            self.o_pressed_time = now
            self.op_pressed_time = now
            self.traycmdchar = "o"
            print("You pressed ‘o’")

        elif keyboard.is_pressed("q"):
            k = keyboard.read_key()
            print("You pressed ‘q’ - will exit simulation.")
            self.sim.close()

        elif keyboard.is_pressed("r"):
            k = keyboard.read_key()
            if now - self.w_pressed_time < 1.0:
                setcamview("right-view")
                print("You pressed ‘wr' for right-view.")
            else:
                if now - self.r_pressed_time < 1.0:
                    self.curRocu.ShowReachabilityGrid(clear=True)
                    print("You pressed ‘rr’ - showing reachability")
                else:
                    self.r_pressed_time = now
                    print("You pressed ‘r’ ")

        elif keyboard.is_pressed("w"):
            k = keyboard.read_key()
            self.w_pressed_time = now
            print("You pressed ‘w’")

        elif keyboard.is_pressed("v"):
            k = keyboard.read_key()
            self.curRocu.ToggleCollisionSphereVisiblity()
            print(f"You pressed 'v' - vizi_spheres is now {self.rocuWrap1.vizi_spheres}.")

        elif keyboard.is_pressed("z"):
            k = keyboard.read_key()
            self.zingle_step = not self.zingle_step
            print(f"You pressed 'z' - for zingle stepping - now {self.zingle_step}.")

        elif keyboard.is_pressed("0"):
            k = keyboard.read_key()
            print("You pressed ‘0’")
            if now - self.op_pressed_time < 1.0:
                self.traycmd = f"{self.traycmdchar}_{self.traynum}_0"
            if now - self.t_pressed_time < 1.0:
               self.traynum = 0
               print(f"You pressed ‘t0’ - curTray is now {self.traynum}")

        elif keyboard.is_pressed("1"):
            k = keyboard.read_key()
            if now - self.op_pressed_time < 1.0:
                self.traycmd = f"{self.traycmdchar}_{self.traynum}_1"
            if now - self.a_pressed_time < 1.0:
               self.curRocu = self.rocuWrap1
               print(f"You pressed ‘a1’ - curRocu is now {self.curRocu.name}")
            if now - self.t_pressed_time < 1.0:
               self.traynum = 1
               print(f"You pressed ‘t1’ - curTray is now {self.traynum}")

        elif keyboard.is_pressed("2"):
            k = keyboard.read_key()
            if now - self.op_pressed_time < 1.0:
                self.traycmd = f"{self.traycmdchar}_{self.traynum}_2"
            if now - self.a_pressed_time < 1.0:
               self.curRocu = self.rocuWrap2
               print(f"You pressed ‘a2’ - curRocu is now {self.curRocu.name}")
            if now - self.t_pressed_time < 1.0:
               self.traynum = 2
               print(f"You pressed ‘t2’ - curTray is now {self.traynum}")

        elif keyboard.is_pressed("3"):
            k = keyboard.read_key()
            if now - self.op_pressed_time < 1.0:
                self.traycmd = f"{self.traycmdchar}_{self.traynum}_3"
            if now - self.t_pressed_time < 1.0:
               self.traynum = 3
               print(f"You pressed ‘t3’ - curTray is now {self.traynum}")

        elif keyboard.is_pressed("4"):
            k = keyboard.read_key()
            if now - self.op_pressed_time < 1.0:
                self.traycmd = f"{self.traycmdchar}_{self.traynum}_4"

        elif keyboard.is_pressed("5"):
            k = keyboard.read_key()
            if now - self.op_pressed_time < 1.0:
                self.traycmd = f"{self.traycmdchar}_{self.traynum}_5"


    def TrayCommandReady(self):
        rv = False
        ok = self.traycmd is not None
        if ok:
            cmd, s_t_num, s_phone_num = self.traycmd.split("_")
            t_num = int(s_t_num)
            phone_num = int(s_phone_num)
            rv = cmd in ['p', 'o'] and (0 <= t_num < 4) and (0 <= phone_num < 6)
            if not rv:
                print(f"Bad traycmd:{self.traycmd}")
                self.traycmd = None
        return rv

    def GetTrayCommand(self):
        cmd, s_tnum, s_phone_num = self.traycmd.split("_")
        iphonenum = int(s_phone_num)
        itranum = int(s_tnum)
        self.traycmd = None
        return cmd, itranum, iphonenum

    def DumpHotKeys(self):
        hotkey_registry = omni.kit.hotkeys.core.get_hotkey_registry()
        discovered_hotkeys = hotkey_registry.get_all_hotkeys()

        # To print current hotkeys
        for ik, hotkey in enumerate(discovered_hotkeys):
            print(f"{ik} - {hotkey}")
            print(f"       {hotkey.key_combination}")

    def DeRegisterAllHotKeys(self):
        hotkey_registry = omni.kit.hotkeys.core.get_hotkey_registry()
        discovered_hotkeys = hotkey_registry.get_all_hotkeys()
        print(f"There are now {len(discovered_hotkeys)} hotkeys.")
        delete_list = discovered_hotkeys.copy()
        for hotkey in delete_list:
            try:
                hotkey_registry.deregister_hotkey(hotkey)
            except Exception as e:
                print(f"Failed to deregister hotkey:{hotkey} {e}")
        discovered_hotkeys = hotkey_registry.get_all_hotkeys()
        print(f"After deleteion there are now {len(discovered_hotkeys)} hotkeys.")

        # To delete hotkey
        # hotkey_registry.deregister_hotkey("omni.kit.manipulator.tool.snap", "S")