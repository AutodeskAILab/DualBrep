"""Iso-surface extraction from the decoded SDF/UDF volumes (marching cubes).

* ``sdf2mesh`` – marching cubes on the SDF volume at iso = 0 (the surface).
* ``udf2mesh`` – marching cubes on the UDF volume at a small threshold to
                 recover the BRep wireframe / edge tubes.

All vertices are returned in the normalized cube [-1, 1]^3.
"""
import mcubes


def sdf2mesh(v_sdf):
    """Marching cubes on an (R, R, R) SDF volume at iso = 0."""
    res = v_sdf.shape[0]
    v, f = mcubes.marching_cubes(v_sdf, 0.0)
    f = f[:, [2, 1, 0]]
    v = v / (res - 1) * 2 - 1
    return v, f


def udf2mesh(v_udf, v_threshold=0.01):
    """Marching cubes on an (R, R, R) UDF volume at ``v_threshold``."""
    res = v_udf.shape[0]
    v, f = mcubes.marching_cubes(v_udf, v_threshold)
    f = f[:, [2, 1, 0]]
    v = v / (res - 1) * 2 - 1
    return v, f
