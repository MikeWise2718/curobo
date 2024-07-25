import sys
import typing
from pxr import Usd, UsdGeom, Gf

def get_world_transform_xform(prim: Usd.Prim) -> typing.Tuple[Gf.Vec3d, Gf.Rotation, Gf.Vec3d]:
    """
    Get the local transformation of a prim using Xformable.
    See https://openusd.org/release/api/class_usd_geom_xformable.html
    Args:
        prim: The prim to calculate the world transformation.
    Returns:
        A tuple of:
        - Translation vector.
        - Rotation quaternion, i.e. 3d vector plus angle.
        - Scale vector.
    """
    xform = UsdGeom.Xformable(prim)
    time = Usd.TimeCode.Default() # The time at which we compute the bounding box
    world_transform: Gf.Matrix4d = xform.ComputeLocalToWorldTransform(time)
    translation: Gf.Vec3d = world_transform.ExtractTranslation()
    rotation: Gf.Rotation = world_transform.ExtractRotation()
    scale: Gf.Vec3d = Gf.Vec3d(*(v.GetLength() for v in world_transform.ExtractRotationMatrix()))
    return translation, rotation, scale

#############
# Full Usage
#############

from pxr import Sdf

def rads(degs: Gf.Vec3d) -> Gf.Vec3d:
    fak = 3.14159265358979323846 / 180.0
    rv =  Gf.Vec3d(fak*degs[0], fak*degs[1], fak*degs[2])
    return rv


def set_xform_trans(prerot:Gf.Vec3d, transvek: Gf.Vec3d, rotxyz: Gf.Vec3d, order="ZYX") -> typing.Tuple[Gf.Vec3d, Gf.Rotation]:
# Create an in-memory Stage with /World Xform prim as the default prim
    stage: Usd.Stage = Usd.Stage.CreateInMemory()
    default_prim: Usd.Prim = UsdGeom.Xform.Define(stage, Sdf.Path("/World")).GetPrim()
    stage.SetDefaultPrim(default_prim)

    xform: Usd.Prim = UsdGeom.Xform.Define(stage, default_prim.GetPath().AppendPath("Xform"))

    xform.AddRotateXYZOp(opSuffix='prerot').Set(value=prerot) # degrees not radians
    xform.AddTranslateOp().Set(value=transvek)
    # xform.AddRotateXYZOp().Set(value=rotxyz)
    match order:
        case "ZYX":
            xform.AddRotateXOp().Set(value=rotxyz[0])
            xform.AddRotateYOp().Set(value=rotxyz[1])
            xform.AddRotateZOp().Set(value=rotxyz[2])
        case "XYZ":
            xform.AddRotateZOp().Set(value=rotxyz[2])
            xform.AddRotateYOp().Set(value=rotxyz[1])
            xform.AddRotateXOp().Set(value=rotxyz[0])

    (transvek, rot, scale) = get_world_transform_xform(xform)
    return (transvek, rot)

def forward_trans(transvek,rot, vec):
    # should take 0,0,0 (bottom of robot) into 0,0,1
    # should take 0,0,1 (top of robot) into 0, -0.7, 0.29 (approx 1-0.7)
    rotmat3d = Gf.Matrix3d(rot)
    print(f"\nrotv3d:{rotmat3d}")
    v = transvek + vec*rotmat3d
    return v

def backward_trans(transvek, rot, vec):
    inv_rotmat3d = Gf.Matrix3d(rot).GetTranspose()
    print(f"\ninv_rotv3d:{inv_rotmat3d}")
    v = (vec - transvek)*inv_rotmat3d
    return v


(transvek,rot) = set_xform_trans(Gf.Vec3d(0,0,0), Gf.Vec3d(0,0,1), Gf.Vec3d(135,0,0))
print(sys.executable)
print(sys.version)
print(f"trans:{transvek}, rot:{rot}")

# rotmat3d = Gf.Matrix3d(rot)
# print(f"\nrotv3d:{rotmat3d}")

# v = Gf.Vec3d(1,0,0)*rotmat3d
# print(f"\nrotv3d:{v}")

v0 = Gf.Vec3d(0.01,0.00,1)
print(f"v0:{v0}")
fv = forward_trans(transvek, rot, v0)
print(f"fv:{fv}")
v1 = backward_trans(transvek, rot, fv)
print(f"v1:{v1}")
