import argparse
from pxr import Gf, Sdf, Usd, UsdGeom
from typing import Dict
import numpy as np

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
    help="Robot Orientation - Euler Angles - default = [0,0,0]",
)
parser.add_argument(
    "-rbpr",
    "--robprerot",
    type=str,
    default=None,
    help="Robot Pre-Rotation - Euler Angles - default = [0,0,0]",
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

def get_args():
    args = parser.parse_args()
    return args

# def to_list3(gft):
#     lst = [gft[0], gft[1], gft[2]]
#     return lst


def quatd_to_list4(q: Gf.Quatd):
    w = q.GetReal()
    im = q.GetImaginary()
    x = im[0]
    y = im[1]
    z = im[2]
    lst = [w, x, y, z]
    return lst


def list4to_quatd(lst: list):
    q = Gf.Quatd(lst[0], Gf.Vec3d(lst[1], lst[2], lst[3]))
    return q


def get_vek(s:str, default=[0.0, 0.0, 0.0]):
    if s is None:
        return np.array(default)
    if s=="":
        return np.array(default)
    if s[0]=="[":
        s = s[1:]
    if s[-1]=="]":
        s = s[:-1]
    sar = s.split(",")
    far = np.array([float(x) for x in sar])
    return far

def print_gfvec3d(name, v: Gf.Vec3d):
    s = f"x:{v[0]:.3f} y:{v[1]:.3f} z:{v[2]:.3f}"
    if name is not None:
        s = name + " " + s
    print(s)

def fmtrow(ncol,r):
    s = ""
    if ncol > len(r):
        ncol = len(r)
    try:
        for i in range(ncol):
            s += f"{r[i]:6.3f} "
    except Exception as e:
        s = f"error {e} formating row:{r}"
    return s

def print_mat(name, rowdim, coldim, mat):
    lls = []
    for i in range(rowdim):
        r = mat.GetRow(i)
        lls.append(fmtrow(coldim, r))
    if name is not None:
        print(name)
    for ls in lls:
        print(ls)

def get_sphere_entry(config_spheres: Dict, idx: int):
    # first get counts
    cskeys = list(config_spheres.keys())
    if len(cskeys) == 0:
        print("error retriving sphere entry - no spheres found")
        return None
    cs_idx = 0
    nseen = 0
    nlstseen = 0
    curkey = cskeys[cs_idx]
    nseen += len(config_spheres[curkey])
    while nseen <= idx:
        cs_idx += 1
        if cs_idx >= len(cskeys):
            msg = f"error retriving sphere entry - not enough spheres found idx:{idx}"
            print(msg)
            return None
        curkey = cskeys[cs_idx]
        nlstseen = nseen
        nseen += len(config_spheres[curkey])

    newidx = idx - nlstseen
    sph_spec = config_spheres[curkey][0]
    sph_spec["keyname"] = curkey
    sph_spec["keyidx"] = newidx
    return sph_spec
