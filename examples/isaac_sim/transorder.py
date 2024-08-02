from typing import Optional
import keyboard
import time
import os

import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "--headless_mode",
    type=str,
    default=None,
    help="To run headless, use one of [native, websocket], webrtc might not work.",
)
args = parser.parse_args()



from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": False,
        "width": "1920",
        "height": "1080",
    }
)

from omni.isaac.core import World
from omni.isaac.core.objects import cuboid, sphere, cone

from omni.isaac.kit import SimulationApp
from omni.isaac.core.utils.extensions import enable_extension
from curobo.util.logger import log_warn
from curobo.geom.types import WorldConfig
from curobo.util_file import (
    get_assets_path,
    get_filename,
    get_path_of_dir,
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
)
from curobo.util.usd_helper import UsdHelper
from omni.isaac.core.utils.viewports import set_camera_view

def add_extensions(simulation_app, headless_mode: Optional[str] = None):
    ext_list = [
        "omni.kit.asset_converter",
        "omni.kit.tool.asset_importer",
        "omni.isaac.asset_browser",
    ]
    if headless_mode is not None:
        log_warn("Running in headless mode: " + headless_mode)
        ext_list += ["omni.kit.livestream." + headless_mode]
    [enable_extension(x) for x in ext_list]
    simulation_app.update()

    return True


add_extensions(simulation_app, args.headless_mode)


def main():
    my_world = World(stage_units_in_meters=0.05)
    stage = my_world.stage

    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)

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
    my_world.scene.add_default_ground_plane()

    sp_wc = np.array([0, 0, 0])
    sq_wc = np.array([1, 0, 0, 0])

    # red_cone = cone.VisualCone(
    #         "/World/target",
    #         position=sp_wc,
    #         orientation=sq_wc,
    #         color=np.array([1.0, 0, 0]),
    #         radius=0.05,
    #         height=0.10,
    #     )

    from pxr import Usd, UsdGeom, Gf, Sdf

    UsdGeom.Xform.Define(stage, '/welt')
    redConePrim = UsdGeom.Cone.Define(stage, '/welt/red_cone')
    redConePrim.AddScaleOp().Set((0.3,0.3,0.3))
    redConePrim.AddTranslateOp().Set((0,0,5))
    redConePrim.AddRotateYOp().Set(45)
    redConePrim.GetDisplayColorAttr().Set([(1,0,0)])

    blueConePrim = UsdGeom.Cone.Define(stage, '/welt/blue_cone')
    blueConePrim.AddScaleOp().Set((0.3,0.3,0.3))
    blueConePrim.AddRotateYOp().Set(45)
    blueConePrim.AddTranslateOp().Set((0,0,5))
    blueConePrim.GetDisplayColorAttr().Set([(0,0,1)])


    usd_help = UsdHelper()

    usd_help.load_stage(my_world.stage)
    usd_help.add_world_to_stage(world_cfg, base_frame="/World")

    set_camera_view(eye=[0.0, 12.5, 1.0], target=[0,0,0], camera_prim_path="/OmniverseKit_Persp")


    while simulation_app.is_running():
        if keyboard.is_pressed("q"):
            k = keyboard.read_key()
            break

        my_world.step(render=True)
        step_index = my_world.current_time_step_index

        if not my_world.is_playing():
            if step_index % 500 == 0:
                print(f"**** Click Play to start simulation - q to quit ***** si:{step_index}")


        # cp,cq = red_cone.get_world_pose()


    print(f"Done - is_running:{simulation_app.is_running()}")
    simulation_app.close()

if __name__ == "__main__":
    main()
