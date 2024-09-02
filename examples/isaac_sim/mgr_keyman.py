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
from rocu_reachability import ReachGridMan, GridRenderFilter, GridRenderStyle

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
        self.g_pressed_time = loop_start
        self.o_pressed_time = loop_start
        self.p_pressed_time = loop_start
        self.q_pressed_time = loop_start
        self.r_pressed_time = loop_start
        self.s_pressed_time = loop_start
        self.t_pressed_time = loop_start
        self.u_pressed_time = loop_start
        self.v_pressed_time = loop_start
        self.w_pressed_time = loop_start
        self.op_pressed_time = loop_start
        self.traycmd = None
        self.traynum = 1
        self.nseeds = 8
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
        nseeds = -1
        grid_render_filter:GridRenderFilter = GridRenderFilter.ALL
        grid_render_style:GridRenderStyle = GridRenderStyle.DEBUG_SPHERES
        currobo = "None"
        if self.curRocu is not None and self.curRocu.rgm is not None:
            rgm: ReachGridMan = self.curRocu.rgm
            nseeds = rgm.nseeds
            grid_render_filter = rgm.grid_render_filter
            grid_render_style = rgm.grid_render_style
            currobo = self.curRocu.name

        msgs.append(indent1+"General Cp,,amds")
        msgs.append(indent2+"? - help")
        msgs.append(indent2+"q - quit")
        msgs.append(indent2+"z - zingle stepping")
        msgs.append(indent1+"Target-cube manipulation")
        msgs.append(indent2+"ct - circle target")
        msgs.append(indent2+"c* - increase target-cube velocity")
        msgs.append(indent2+"c/ - decrease target-cube velocity")
        msgs.append(indent2+"cc - reset target-cube to start pose")
        msgs.append(indent2+"cd - reset target-cube to current end-effector pose")
        msgs.append(indent1+"View commands")
        msgs.append(indent2+"vt - top view")
        msgs.append(indent2+"vr - right view")
        msgs.append(indent2+"vl - left view")
        msgs.append(indent2+"vb - back view")
        msgs.append(indent2+"vf - front view")
        msgs.append(indent1+"Robot commands")
        msgs.append(indent2+f"aN - make robot arm N to active robot - current:{currobo}")
        msgs.append(indent2+"am - toggle material")
        msgs.append(indent2+"al - show joints close to limits")
        msgs.append(indent1+"Tray commands")
        msgs.append(indent2+f"tN - make tray N the active tray - current tray is {self.traynum}")
        msgs.append(indent2+"pN - move target to the phone postion N on tray")
        msgs.append(indent2+"oN - move target to 20 cm above the phone postion N")
        msgs.append(indent1+"Grid commands")
        msgs.append(indent2+"gr - draw or redraw grid")
        msgs.append(indent2+"gc - clear grid")
        msgs.append(indent2+"g* - finer grid (more points)")
        msgs.append(indent2+"g/ - decrease grid (less points)")
        msgs.append(indent2+"ge - extend grid span (more volume)")
        msgs.append(indent2+"gd - decrease grid span (less volume)")
        msgs.append(indent2+f"gs - change grid render style - current mode is {grid_render_style}")
        msgs.append(indent2+f"gf - change grid render filter - current filter is {grid_render_filter}")
        msgs.append(indent2+f"gp - increase number of seeds by factor 2 - current number of seeds is {nseeds}")
        msgs.append(indent2+f"gq - decrease number of seeds by factor 2 - current number of seeds is {nseeds}")

        for msg in msgs:
            print(msg)
        self.DumpHotKeys()

    def ProcessKeys(self):

        now = time.time()
        if keyboard.is_pressed("*"):
            k = keyboard.read_key()
            if now - self.c_pressed_time < 1.0:
                print(f"You pressed a* curvel:{self.curRocu.curvel}")
                self.curRocu.curvel *= 1.5
            elif now - self.g_pressed_time < 1.0:
                print(f"You pressed g* increasing grid size.")
                self.curRocu.ChangeGridSize(1.5)

        elif keyboard.is_pressed("/"):
            k = keyboard.read_key()
            if k=="?":
                print(f"You pressed ? - getting help  l:{k}")
                self.PrintHelp()
            else:
                if now - self.c_pressed_time < 1.0:
                    self.curRocu.curvel /= 1.5
                    print(f"You pressed a/. curvel:{self.curRocu.curvel}   l:{k}")
                elif now - self.g_pressed_time < 1.0:
                    print(f"You pressed g/ decreasing grid size.")
                    self.curRocu.ChangeGridSize(0.6666667)

        elif keyboard.is_pressed("a"):
            k = keyboard.read_key()
            self.a_pressed_time = now
            print(f"You pressed a.")

        elif keyboard.is_pressed("b"):
            k = keyboard.read_key()
            if now - self.v_pressed_time < 1.0:
                setcamview("back-view")
            print("You pressed vb for back-view.")

        elif keyboard.is_pressed("c"):
            k = keyboard.read_key()
            if now - self.g_pressed_time < 1.0:
                self.curRocu.ClearReachabilityGrid()
                print("You pressed rc - clearing reachability")
            else:
                self.c_pressed_time = now
                print("You pressed c")

        elif keyboard.is_pressed("d"):
            k = keyboard.read_key()
            if now - self.v_pressed_time < 1.0:
                setcamview("diag-view")
                print("You pressed vd for diag-view.")
            elif now - self.c_pressed_time < 1.0:
                print("You pressed cd - will move to robot's current end-effector pose.")
                if self.curRocu.cu_js is not None:
                    self.curRocu.MoveTargetToEepose()
            elif now - self.a_pressed_time < 1.0:
                print("You pressed ad - will move to robot's start pose.")
                if self.curRocu.cu_js is not None:
                    self.curRocu.MoveTargetToStartPose()
            elif now - self.g_pressed_time < 1.0:
                self.curRocu.ChangeGridSpan(0.6666667)
                print("You pressed gd to decrease grid span")
            else:
                self.d_pressed_time = now
                print("You pressed d")

        elif keyboard.is_pressed("e"):
            k = keyboard.read_key()
            if now - self.g_pressed_time < 1.0:
                self.curRocu.ChangeGridSpan(1.5)
                print("You pressed ge to extend grid span")

        elif keyboard.is_pressed("f"):
            k = keyboard.read_key()
            if now - self.v_pressed_time < 1.0:
                setcamview("front-view")
                print("You pressed vf for front-view.")
            elif now - self.g_pressed_time < 1.0:
                cur = self.curRocu.ChangeGridRenderFilter()
                print(f"You pressed gf for change grid filter - now:{cur}.")


        elif keyboard.is_pressed("g"):
            k = keyboard.read_key()
            self.g_pressed_time = now
            print("You pressed g")


        elif keyboard.is_pressed("l"):
            k = keyboard.read_key()
            if now - self.v_pressed_time < 1.0:
                setcamview("left-view")
            print("You pressed vf for left-view.")

        elif keyboard.is_pressed("m"):
            k = keyboard.read_key()
            if now - self.a_pressed_time < 1.0:
                self.curRocu.toggle_material()
                self.curRocu.check_alarm_status()
                print("You pressed am - changing material on arm")

        elif keyboard.is_pressed("o"):
            k = keyboard.read_key()
            self.o_pressed_time = now
            self.op_pressed_time = now
            self.traycmdchar = "o"
            print("You pressed o")

        elif keyboard.is_pressed("p"):
            k = keyboard.read_key()
            if now - self.g_pressed_time < 1.0:
                print("You pressed gp for increase seeds.")
                self.curRocu.ChangeNumSeeds(2.0)
            else:
                self.p_pressed_time = now
                self.op_pressed_time = now
                self.traycmdchar = "p"
                print("You pressed p")

        elif keyboard.is_pressed("q"):
            k = keyboard.read_key()
            if now - self.g_pressed_time < 1.0:
                print("You pressed gq for decrease seeds.")
                self.curRocu.ChangeNumSeeds(0.5)
            if now - self.q_pressed_time < 1.0:
                print("You pressed qq - will exit simulation.")
                self.sim.close()
            else:
                self.q_pressed_time = now
                print("You pressed q")

        elif keyboard.is_pressed("r"):
            k = keyboard.read_key()
            if now - self.v_pressed_time < 1.0:
                setcamview("right-view")
                print("You pressed vr for right-view.")
            else:
                if now - self.g_pressed_time < 1.0:
                    self.curRocu.ShowReachabilityGrid(clear=True)
                    print("You pressed gr - showing reachability")
                else:
                    self.r_pressed_time = now
                    print("You pressed r ")

        elif keyboard.is_pressed("s"):
            k = keyboard.read_key()
            if now - self.g_pressed_time < 1.0:
                cur = self.curRocu.ChangeGridRenderStyle()
                print(f"You pressed gs for change grid style - now:{cur}")

        elif keyboard.is_pressed("t"):
            k = keyboard.read_key()
            if now - self.v_pressed_time < 1.0:
                setcamview("top-view")
                print("You pressed vt for top-view.")
            elif now - self.a_pressed_time < 1.0:
                self.curRocu.ToggleCirclingTarget()
                print("You pressed ‘at’ for rotate target.")
            else:
                self.t_pressed_time = now

        elif keyboard.is_pressed("w"):
            k = keyboard.read_key()
            self.w_pressed_time = now
            print("You pressed w")

        elif keyboard.is_pressed("u"):
            k = keyboard.read_key()
            self.u_pressed_time = now
            print("You pressed u")

        elif keyboard.is_pressed("v"):
            k = keyboard.read_key()
            if now - self.v_pressed_time < 1.0:
                self.curRocu.ToggleCollisionSphereVisiblity()
                print(f"You pressed vv - vizi_spheres is now {self.curRocu.vizi_spheres}.")
            else:
                self.v_pressed_time = now
                print("You pressed v")

        elif keyboard.is_pressed("z"):
            k = keyboard.read_key()
            self.zingle_step = not self.zingle_step
            print(f"You pressed 'z' - for zingle stepping - now {self.zingle_step}.")

        elif keyboard.is_pressed("0"):
            k = keyboard.read_key()
            print("You pressed 0")
            if now - self.op_pressed_time < 1.0:
                self.traycmd = f"{self.traycmdchar}_{self.traynum}_0"
            if now - self.t_pressed_time < 1.0:
               self.traynum = 0
               print(f"You pressed t0 - curTray is now {self.traynum}")

        elif keyboard.is_pressed("1"):
            k = keyboard.read_key()
            if now - self.op_pressed_time < 1.0:
                self.traycmd = f"{self.traycmdchar}_{self.traynum}_1"
            if now - self.a_pressed_time < 1.0:
               self.curRocu = self.rocuWrap1
               print(f"You pressed a1 - curRocu is now {self.curRocu.name}")
            if now - self.t_pressed_time < 1.0:
               self.traynum = 1
               print(f"You pressed t1 - curTray is now {self.traynum}")

        elif keyboard.is_pressed("2"):
            k = keyboard.read_key()
            if now - self.op_pressed_time < 1.0:
                self.traycmd = f"{self.traycmdchar}_{self.traynum}_2"
            if now - self.a_pressed_time < 1.0:
               self.curRocu = self.rocuWrap2
               print(f"You pressed a2 - curRocu is now {self.curRocu.name}")
            if now - self.t_pressed_time < 1.0:
               self.traynum = 2
               print(f"You pressed t2 - curTray is now {self.traynum}")

        elif keyboard.is_pressed("3"):
            k = keyboard.read_key()
            if now - self.op_pressed_time < 1.0:
                self.traycmd = f"{self.traycmdchar}_{self.traynum}_3"
            if now - self.t_pressed_time < 1.0:
               self.traynum = 3
               print(f"You pressed t3 - curTray is now {self.traynum}")

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
