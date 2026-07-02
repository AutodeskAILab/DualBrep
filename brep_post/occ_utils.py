# from occwl.shape import BRepTools_ShapeSet
from OCC.Core import BRepBndLib, TopoDS
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRepTools import BRepTools_WireExplorer
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Extend.DataExchange import read_step_file
from OCC.Core.TopAbs import (
    TopAbs_FACE,
    TopAbs_VERTEX,
    TopAbs_EDGE,
    TopAbs_SHELL,
    TopAbs_WIRE,
    TopAbs_SOLID,
    TopAbs_COMPOUND,
    TopAbs_COMPSOLID, TopAbs_REVERSED, TopAbs_FORWARD
)
from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Line, GeomAbs_Circle, GeomAbs_Ellipse
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Trsf
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.TopoDS import topods
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopExp import topexp
from OCC.Core.TopTools import TopTools_IndexedMapOfShape
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.TopTools import TopTools_HSequenceOfShape
from OCC.Core.ShapeAnalysis import ShapeAnalysis_FreeBounds, ShapeAnalysis_Wire, ShapeAnalysis_Surface
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepTools import breptools
from OCC.Core.GeomLProp import GeomLProp_SLProps
import math
import numpy as np

def diable_occ_log():
    from OCC.Core.Message import Message_Alarm, message
    printers = message.DefaultMessenger().Printers()
    for idx in range(printers.Length()):
        printers.Value(idx + 1).SetTraceLevel(Message_Alarm)


def read_brep(v_file):
    shape_set = BRepTools_ShapeSet()
    with open(v_file, "r") as fp:
        shape_set.ReadFromString(fp.read())
    shapes = []
    for i in range(shape_set.NbShapes()):
        shapes.append(shape_set.Shape(i+1))
    shp, success = list_of_shapes_to_compound(shapes)
    # Get solid from compound
    exp = TopExp_Explorer(shp, TopAbs_SOLID)
    shp = topods_Solid(exp.Current())
    return shp

def read_solids(v_file):
    shape = read_step_file(str(v_file), verbosity=False)
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    solids = []
    while exp.More():
        s = exp.Current()
        exp.Next()
        if s.ShapeType() == TopAbs_SOLID:
            solids.append(topods.Solid(s))
    return solids

def normalize_shape(v_shape, v_bounding=1., v_raise_error=True):
    v, f = get_triangulations(v_shape, 0.01, 0.1, v_raise_error=v_raise_error)
    v = np.array(v)
    xmin, ymin, zmin = v.min(axis=0)
    xmax, ymax, zmax = v.max(axis=0)

    scale_x = v_bounding * 2 / (xmax - xmin)
    scale_y = v_bounding * 2 / (ymax - ymin)
    scale_z = v_bounding * 2 / (zmax - zmin)
    scaleFactor = min(scale_x, scale_y, scale_z)
    translation1 = gp_Vec(-(xmax + xmin) / 2, -(ymax + ymin) / 2, -(zmax + zmin) / 2)
    trsf1 = gp_Trsf()
    trsf1.SetTranslationPart(translation1)
    trsf2 = gp_Trsf()
    trsf2.SetScaleFactor(scaleFactor)
    trsf2.Multiply(trsf1)

    transformer = BRepBuilderAPI_Transform(trsf2)
    transformer.Perform(v_shape)
    shape = transformer.Shape()

    v, f = get_triangulations(shape, 0.01, 0.1, True)
    v = np.array(v)
    xmin, ymin, zmin = v.min(axis=0)
    xmax, ymax, zmax = v.max(axis=0)
    max_extent = max(xmax-xmin, ymax-ymin, zmax-zmin)
    if max_extent > (v_bounding * 2) + 0.05 or max_extent < (v_bounding * 2) - 0.05:
        raise Exception("Bounding box after normalization exceeds the expected size. OCC bug.")

    return shape, scaleFactor, xmin, ymin, zmin, xmax, ymax, zmax


def normalize_shape_orthogonal(v_shape, v_bounding=1.):
    boundingBox = Bnd_Box()
    BRepBndLib.brepbndlib.Add(v_shape, boundingBox)
    xmin, ymin, zmin, xmax, ymax, zmax = boundingBox.Get()
    scaleFactor = math.sqrt((xmax - xmin)**2+(ymax - ymin)**2+(zmax - zmin)**2)
    scaleFactor = v_bounding * 2 / scaleFactor
    translation1 = gp_Vec(-(xmax + xmin) / 2, -(ymax + ymin) / 2, -(zmax + zmin) / 2)
    trsf1 = gp_Trsf()
    trsf1.SetTranslationPart(translation1)
    trsf2 = gp_Trsf()
    trsf2.SetScaleFactor(scaleFactor)
    trsf2.Multiply(trsf1)

    transformer = BRepBuilderAPI_Transform(trsf2)
    transformer.Perform(v_shape)
    shape = transformer.Shape()
    return shape


def get_wires(shp):
    exp = TopExp_Explorer(shp, TopAbs_WIRE)
    wires = []
    while exp.More():
        s = exp.Current()
        exp.Next()
        wires.append(topods.Wire(s))
    return wires

def get_faces(shp):
    exp = TopExp_Explorer(shp, TopAbs_FACE)
    faces = []
    while exp.More():
        s = exp.Current()
        exp.Next()
        faces.append(topods.Face(s))
    return faces

def get_edges(shp):
    exp = TopExp_Explorer(shp, TopAbs_EDGE)
    edges = []
    while exp.More():
        s = exp.Current()
        exp.Next()
        edges.append(topods.Edge(s))
    return edges


def is_seam_edge(edge, shape):
    """Check if an edge is a seam edge on any face in the shape.

    A seam edge appears on periodic surfaces (cylinder, cone, sphere, torus)
    where the surface connects to itself.
    """
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while face_explorer.More():
        face = topods.Face(face_explorer.Current())
        if BRep_Tool.IsClosed(edge, face):
            return True
        face_explorer.Next()
    return False


def get_primitives(v_shape, v_type, v_remove_half=False, v_remove_seam=False):
    """Get primitives of a given type from a shape.

    Args:
        v_shape: The shape to explore
        v_type: The type of primitive to get (TopAbs_EDGE, TopAbs_FACE, etc.)
        v_remove_half: If True, remove duplicate edges (edge and its reverse)
        v_remove_seam: If True and v_type is TopAbs_EDGE, remove seam edges
    """
    assert v_shape is not None
    explorer = TopExp_Explorer(v_shape, v_type)
    items = []
    while explorer.More():
        current = explorer.Current()
        if v_remove_half:
            if current in items or current.Reversed() in items:
                explorer.Next()
                continue
        if v_remove_seam and v_type == TopAbs_EDGE:
            edge = topods.Edge(current)
            if is_seam_edge(edge, v_shape):
                explorer.Next()
                continue
        items.append(current)
        explorer.Next()
    return items


def get_triangulations(v_shape, v_resolution1=0.1, v_resolution2=0.1, v_raise_error=False):
    if v_resolution1 > 0:
        mesh = BRepMesh_IncrementalMesh(v_shape, v_resolution1, True, v_resolution2)
    v = []
    f = []
    face_explorer = TopExp_Explorer(v_shape, TopAbs_FACE)
    while face_explorer.More():
        face = face_explorer.Current()
        loc = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, loc)
        if triangulation is None:
            if v_raise_error:
                raise ValueError("Face without triangulation")
            print("Ignore face without triangulation")
            face_explorer.Next()
            continue
        cur_vertex_size = len(v)
        # Triangulation nodes are stored in the face's LOCAL frame; the face's
        # placement within the shape lives in `loc`. Apply it so vertices come
        # out in global coords (consistent with edge adaptors / STEP export).
        # Without this, a shape carrying a non-identity location (e.g. the
        # translation BRepBuilderAPI_Transform leaves on the faces in
        # normalize_shape) is returned shifted out of the expected bounds.
        trsf = loc.Transformation()
        for i in range(1, triangulation.NbNodes() + 1):
            pnt = triangulation.Node(i).Transformed(trsf)
            v.append([pnt.X(), pnt.Y(), pnt.Z()])
        for i in range(1, triangulation.NbTriangles() + 1):
            t = triangulation.Triangle(i)
            if face.Orientation() == TopAbs_REVERSED:
                f.append([t.Value(3) + cur_vertex_size - 1, t.Value(2) + cur_vertex_size - 1,
                          t.Value(1) + cur_vertex_size - 1])
            else:
                f.append([t.Value(1) + cur_vertex_size - 1, t.Value(2) + cur_vertex_size - 1,
                          t.Value(3) + cur_vertex_size - 1])
        face_explorer.Next()
    return v, f


def get_ordered_edges(v_face):
    edges = []
    for wire in get_primitives(v_face, TopAbs_WIRE):
        wire_explorer = BRepTools_WireExplorer(wire)
        local_edges = []
        while wire_explorer.More():
            edge = TopoDS.topods.Edge(wire_explorer.Current())
            local_edges.append(edge)
            wire_explorer.Next()
        edges.append(local_edges)
    return edges

def get_curvature(v_solid):
    mean_curvatures = []
    gs_curvatures = []
    areas = []
    faces = get_primitives(v_solid, TopAbs_FACE)
    for face in faces:
        props = GProp_GProps()
        brepgprop.SurfaceProperties(face, props)
        u_min, u_max, v_min, v_max = breptools.UVBounds(face)

        area = abs(props.Mass())
        mc = []
        gc = []
        for u in np.linspace(u_min, u_max, num=10):
            for v in np.linspace(v_min, v_max, num=10):
                props = GeomLProp_SLProps(BRepAdaptor_Surface(face, True).Surface().Surface(), u, v, 2, 1e-6)
                if props.IsCurvatureDefined():
                    mc.append(abs(props.MeanCurvature()))
                    gc.append(abs(props.GaussianCurvature()))
        mean_curvatures.append(np.mean(mc))
        gs_curvatures.append(np.mean(gc))
        areas.append(area)
    mean_curvatures = np.array(mean_curvatures)
    gs_curvatures = np.array(gs_curvatures)
    areas = np.array(areas)
    return mean_curvatures, gs_curvatures, areas, len(faces)


def get_curve_length(v_edge, v_sample_resolution=32):
    curve = BRepAdaptor_Curve(v_edge)
    range_start = curve.FirstParameter()
    range_end = curve.LastParameter()
    sample_u = np.linspace(range_start, range_end, num=v_sample_resolution)
    last_point = curve.Value(sample_u[0])
    length = 0
    for u in sample_u[1:]:
        pnt = curve.Value(u)
        length += last_point.Distance(pnt)
        last_point = pnt

    return length

def get_complexity(v_solid):
    mean_curvatures, gs_curvatures, areas, num_faces = get_curvature(v_solid)
    mean_curvatures = np.clip(mean_curvatures, 0, 3) / 3 
    gs_curvatures = np.clip(gs_curvatures, 0, 3) / 3
    areas = np.clip(areas, 0, 3) / 3
    num_faces = np.clip(num_faces, 1, 50) / 50
    score = np.mean(mean_curvatures * areas) + np.mean(gs_curvatures * areas) + num_faces / 20
    return score

        