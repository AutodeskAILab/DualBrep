import copy
import random

import numpy as np
import torch
import torch.nn as nn
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeEdge, \
    BRepBuilderAPI_MakeVertex
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh

from OCC.Core.GeomAPI import GeomAPI_PointsToBSplineSurface, GeomAPI_PointsToBSpline
from OCC.Core.GeomAbs import GeomAbs_C0, GeomAbs_C1, GeomAbs_C2, GeomAbs_C3, GeomAbs_G1
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Wire, ShapeAnalysis_Shell, ShapeAnalysis_ShapeTolerance
from OCC.Core.ShapeExtend import ShapeExtend_WireData
from OCC.Core.ShapeFix import ShapeFix_Face, ShapeFix_Wire, ShapeFix_Edge, ShapeFix_Shell, ShapeFix_Solid, \
    ShapeFix_ComposeShell, ShapeFix_ShapeTolerance
from OCC.Core.TColgp import TColgp_Array1OfPnt
from OCC.Core.TColgp import TColgp_Array2OfPnt
from OCC.Core.TopAbs import TopAbs_COMPOUND, TopAbs_FORWARD, TopAbs_REVERSED, TopAbs_FACE, TopAbs_WIRE, \
    TopAbs_EDGE, TopAbs_SHELL, TopAbs_SOLID
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.gp import gp_Pnt, gp_XYZ, gp_Vec
from OCC.Display.SimpleGui import init_display

from OCC.Core.TopoDS import topods, TopoDS_Shell, TopoDS_Builder, TopoDS_Vertex
from OCC.Core.TopExp import TopExp_Explorer, topexp
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_VERTEX
from OCC.Core.BRep import BRep_Tool

from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape

from OCC.Core.ShapeAnalysis import ShapeAnalysis_FreeBounds
from OCC.Core.TopTools import TopTools_HSequenceOfShape
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib

from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from random import randint

from OCC.Core.BRepCheck import BRepCheck_Analyzer

from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.Interface import Interface_Static
from OCC.Core.IFSelect import IFSelect_RetDone

from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import trimesh

from itertools import combinations

from .occ_utils import get_primitives, get_ordered_edges

import importlib.util

CONFIRM_CHAMFERDIST_NO_AVAILABLE = False
if importlib.util.find_spec("chamferdist") is None or CONFIRM_CHAMFERDIST_NO_AVAILABLE:
    CHAMFERDIST_AVAILABLE = False
    # print("chamferdist package not found. ChamferDistance will be computed using only PyTorch.")
    from .chamferdist_torch import ChamferDistanceTorch as ChamferDistance
else:
    CHAMFERDIST_AVAILABLE = True
    # print("chamferdist package found. ChamferDistance will be computed using chamferdist package.")
    from chamferdist import ChamferDistance

# EDGE_FITTING_TOLERANCE = [1e-5, 1e-4, 1e-3, 5e-3, 8e-3, 5e-2, ]
# FACE_FITTING_TOLERANCE = [1e-4, 1e-3, 1e-2, 3e-2, 5e-2, 8e-2, ]

EDGE_FITTING_TOLERANCE = [1e-3, 5e-3, 8e-3, 5e-2, ]
FACE_FITTING_TOLERANCE = [1e-3, 1e-2, 3e-2, 5e-2, 8e-2, ]

face_fiting_deg_min, face_fiting_deg_max = 3, 8
edge_fitting_deg_min, edge_fitting_deg_max = 0, 8

ROUGH_FITTING_TOLERANCE = 1e-1

FIX_TOLERANCE = 1e-2
FIX_PRECISION = 1e-2
# CONNECT_TOLERANCE = [0.02, ]
CONNECT_TOLERANCE = [2e-3, 6e-3, 1e-2, 1.5e-2, 2e-2, 2.5e-2, 5e-2, 8e-2, ]
SEWING_TOLERANCE = 1e-1
REMOVE_EDGE_TOLERANCE = 1e-3
TRANSFER_PRECISION = 1e-6
MAX_DISTANCE_THRESHOLD = 1e-1
USE_VARIATIONAL_SMOOTHING = True
FIX_CLOSE_TOLERANCE = 1
FIX_GAP_TOLERANCE = 1e-1
weight_CurveLength, weight_Curvature, weight_Torsion = 1, 1, 1
IS_VIZ_WIRE, IS_VIZ_FACE, IS_VIZ_SHELL = False, False, False
CONTINUITY = GeomAbs_C2

INTERPOLATION_PRECISION = 0.05


# EDGE_FITTING_TOLERANCE = [5e-3, 8e-3, 5e-2]
# FACE_FITTING_TOLERANCE = [2e-2, 5e-2, 8e-2]
#
# FIX_TOLERANCE = 1e-2
# FIX_PRECISION = 1e-2
# CONNECT_TOLERANCE = 0.025
# TRIM_FIX_TOLERANCE = 0.025
# SEWING_TOLERANCE = 2e-2
# TRANSFER_PRECISION = 1e-3
# MAX_DISTANCE_THRESHOLD = 1e-1
# USE_VARIATIONAL_SMOOTHING = False
# weight_CurveLength, weight_Curvature, weight_Torsion = 0.4, 0.4, 0.2
# IS_VIZ_WIRE, IS_VIZ_FACE, IS_VIZ_SHELL = False, False, True
# CONTINUITY = GeomAbs_C2


def interpolation_face_points(face, is_use_cuda=False, precision=INTERPOLATION_PRECISION):
    if type(face) is np.ndarray:
        if is_use_cuda:
            face = torch.from_numpy(face).cuda()
        else:
            face = torch.from_numpy(face)

    res = face.shape[0]
    x_density = (res * torch.linalg.norm(face[0, 0] - face[0, 1], dim=-1) / precision).to(torch.long)
    y_density = (res * torch.linalg.norm(face[0, 0] - face[1, 0], dim=-1) / precision).to(torch.long)
    x_density, y_density = max(x_density, res), max(y_density, res)
    x = torch.linspace(-1., 1., x_density).to(face.device)
    y = torch.linspace(-1, 1, y_density).to(face.device)
    x, y = torch.meshgrid(x, y, indexing='ij')
    coords = torch.stack([x, y], dim=-1).reshape(1, -1, 1, 2)
    interpolation_face = torch.nn.functional.grid_sample(face[None].permute(0, 3, 1, 2),
                                                         coords, align_corners=True)[0, :, :, 0].permute(1, 0)
    # trimesh.PointCloud(face.cpu().numpy()).export(r'E:\data\img2brep\debug.ply')
    assert interpolation_face.shape[0] >= res * res
    return interpolation_face


class Shape:
    def __init__(self, v_face_point, v_edge_points, v_connectivity, is_use_cuda=True):
        self.recon_face_points = v_face_point
        self.recon_edge_points = v_edge_points
        self.edge_face_connectivity = v_connectivity.astype(np.int64)
        self.device = torch.device('cuda') if is_use_cuda else torch.device('cpu')

        # Build denser points
        self.interpolation_face = []
        for face in self.recon_face_points:
            self.interpolation_face.append(interpolation_face_points(face, is_use_cuda=is_use_cuda))

        self.chamferdist = ChamferDistance()

        self.remove_edge_idx_src = []
        self.remove_edge_idx_new = []

        self.have_data = True

        pass

    def remove_half_edges(self, edge2face_threshold=0.3, is_check_intersection=False, face2face_threshold=0.06):
        edge_face_connectivity = self.edge_face_connectivity
        cache_dict = {}
        edge_face_connectivity = edge_face_connectivity.astype(np.int64)
        for conec in edge_face_connectivity:
            if (conec[1], conec[2]) in cache_dict:
                cache_dict[(conec[1], conec[2])].append(conec[0])
            elif (conec[2], conec[1]) in cache_dict:
                cache_dict[(conec[2], conec[1])].append(conec[0])
            else:
                cache_dict[(conec[1], conec[2])] = [conec[0]]

        edges = []
        edge_face_connectivity = []

        # Duplicate or delete edges if it appears once
        new_dict = {}
        for key, value in cache_dict.items():
            # Check validity
            face_id1 = key[0]
            face_id2 = key[1]

            # first check the validity of the intersection
            # if is_check_intersection:
            #     face1 = torch.from_numpy(self.recon_face_points[face_id1]).to(self.device)
            #     face2 = torch.from_numpy(self.recon_face_points[face_id2]).to(self.device)
            #     dist_face1_to_face2 = torch.sqrt(self.chamferdist(
            #             face1.reshape(1, -1, 3),
            #             self.interpolation_face[face_id2][None], bidirectional=False, batch_reduction=None, point_reduction=None))
            #     dist_face2_to_face1 = torch.sqrt(self.chamferdist(
            #             face2.reshape(1, -1, 3),
            #             self.interpolation_face[face_id1][None], bidirectional=False, batch_reduction=None, point_reduction=None))
            #     dis_face_to_face = torch.min(dist_face1_to_face2.min(), dist_face2_to_face1.min())
            #
            #     if dis_face_to_face > face2face_threshold:
            #         for item in value:
            #             self.remove_edge_idx_src.append(item)
            #         continue

            edge1 = torch.from_numpy(self.recon_edge_points[value[0]]).to(self.device)
            distance11 = torch.sqrt(self.chamferdist(
                    edge1[None],
                    self.interpolation_face[face_id1][None]))
            distance12 = torch.sqrt(self.chamferdist(
                    edge1[None],
                    self.interpolation_face[face_id2][None]))

            distance1 = (distance11 + distance12) / 2

            if face_id1 == face_id2:
                edges.append(self.recon_edge_points[value[0]])
                edge_face_connectivity.append([len(edges) - 1, face_id1, face_id2])

            elif len(value) == 2:
                edge2 = torch.from_numpy(self.recon_edge_points[value[1]]).to(self.device)
                distance21 = torch.sqrt(self.chamferdist(
                        edge2[None],
                        self.interpolation_face[face_id1][None]))
                distance22 = torch.sqrt(self.chamferdist(
                        edge2[None],
                        self.interpolation_face[face_id2][None]))

                distance2 = (distance21 + distance22) / 2
                if distance1 > edge2face_threshold and distance2 > edge2face_threshold:
                    for item in value:
                        self.remove_edge_idx_src.append(item)
                    continue
                elif distance1 < edge2face_threshold and distance1 < distance2:
                    edges.append(self.recon_edge_points[value[0]])
                    edge_face_connectivity.append([len(edges) - 1, face_id1, face_id2])
                else:
                    edges.append(self.recon_edge_points[value[1]])
                    edge_face_connectivity.append([len(edges) - 1, face_id1, face_id2])

            elif len(value) == 1:
                if distance1 > edge2face_threshold:
                    self.remove_edge_idx_src.append(value[0])
                    continue
                edges.append(self.recon_edge_points[value[0]])
                edge_face_connectivity.append([len(edges) - 1, face_id1, face_id2])

        if len(edges) == 0 or len(edge_face_connectivity) == 0:
            self.have_data = False
            return

        self.recon_edge_points = np.stack(edges, axis=0)
        self.edge_face_connectivity = np.stack(edge_face_connectivity, axis=0)
        pass

    def check_openness(self, v_threshold=0.95):
        recon_edge = self.recon_edge_points
        dirs = (recon_edge[:, [0, -1]] - np.mean(recon_edge, axis=1, keepdims=True))
        cos_dir = ((dirs[:, 0] * dirs[:, 1]).sum(axis=1) /
                   (np.linalg.norm(dirs[:, 0], axis=1) * np.linalg.norm(dirs[:, 1], axis=1) + 1e-6))
        self.openness = cos_dir > v_threshold

        delta = recon_edge[:, [0, -1]].mean(axis=1)
        # recon_edge[self.openness, 0] = delta[self.openness]
        # recon_edge[self.openness, -1] = delta[self.openness]

    def build_fe(self):
        # face_edge_adj store the edge idx list of each face
        self.face_edge_adj = [[] for _ in range(self.recon_face_points.shape[0])]
        for edge_face1_face2 in self.edge_face_connectivity:
            edge, face1, face2 = edge_face1_face2
            if face1 == face2:
                self.face_edge_adj[face1].append(edge)
                continue
            assert edge not in self.face_edge_adj[face1]
            self.face_edge_adj[face1].append(edge)
            self.face_edge_adj[face2].append(edge)

    def build_vertices(self, v_threshold=1e-1):
        num_faces = self.recon_face_points.shape[0]
        edges = self.recon_edge_points

        inv_edge_face_connectivity = {}
        for edge_idx, face_idx1, face_idx2 in self.edge_face_connectivity:
            inv_edge_face_connectivity[(face_idx1, face_idx2)] = edge_idx
            inv_edge_face_connectivity[(face_idx2, face_idx1)] = edge_idx

        pair1 = []
        is_end_point = []
        face_adj = np.zeros((num_faces, num_faces), dtype=bool)
        face_adj[self.edge_face_connectivity[:, 1], self.edge_face_connectivity[:, 2]] = True
        # Set the diagonal to False
        np.fill_diagonal(face_adj, False)
        face_adj = np.logical_or(face_adj, face_adj.T)

        # Find the 3-node ring of the face
        def dis(a, b):
            return np.linalg.norm(a - b)

        for face1 in range(num_faces):
            for face2 in range(num_faces):
                if not face_adj[face1, face2]:
                    continue
                e12 = inv_edge_face_connectivity[(face1, face2)]
                for face3 in range(num_faces):
                    if not face_adj[face1, face3] or not face_adj[face2, face3]:
                        continue
                    e13 = inv_edge_face_connectivity[(face1, face3)]
                    e23 = inv_edge_face_connectivity[(face2, face3)]

                    def closest_endpoints(edge1, edge2):
                        dis1 = dis(edges[edge1, 0], edges[edge2, 0])
                        dis2 = dis(edges[edge1, 0], edges[edge2, -1])
                        dis3 = dis(edges[edge1, -1], edges[edge2, 0])
                        dis4 = dis(edges[edge1, -1], edges[edge2, -1])
                        dises = [dis1, dis2, dis3, dis4]
                        return np.argmin(dises), np.min(dises)

                    i12 = closest_endpoints(e12, e13)
                    i23 = closest_endpoints(e13, e23)

                    # Start point of e12 and start point of e13 and start point of e23
                    if i12[0] == 0 and i23[0] == 0:
                        pair1.append([e12, e13, e23])
                        is_end_point.append([False, False, False])
                    # Start point of e12 and start point of e13 and end point of e23
                    elif i12[0] == 0 and i23[0] == 1:
                        pair1.append([e12, e13, e23])
                        is_end_point.append([False, False, True])
                    # Start point of e12 and end point of e13 and start point of e23
                    elif i12[0] == 1 and i23[0] == 2:
                        pair1.append([e12, e13, e23])
                        is_end_point.append([False, True, False])
                    # Start point of e12 and end point of e13 and end point of e23
                    elif i12[0] == 1 and i23[0] == 3:
                        pair1.append([e12, e13, e23])
                        is_end_point.append([False, True, True])
                    # End point of e12 and start point of e13 and start point of e23
                    elif i12[0] == 2 and i23[0] == 0:
                        pair1.append([e12, e13, e23])
                        is_end_point.append([True, False, False])
                    # End point of e12 and start point of e13 and end point of e23
                    elif i12[0] == 2 and i23[0] == 1:
                        pair1.append([e12, e13, e23])
                        is_end_point.append([True, False, True])
                    # End point of e12 and end point of e13 and start point of e23
                    elif i12[0] == 3 and i23[0] == 2:
                        pair1.append([e12, e13, e23])
                        is_end_point.append([True, True, False])
                    # End point of e12 and end point of e13 and end point of e23
                    elif i12[0] == 3 and i23[0] == 3:
                        pair1.append([e12, e13, e23])
                        is_end_point.append([True, True, True])
                    else:
                        pass

        if len(pair1) == 0:
            self.pair1 = None
            self.is_end_point = None
            return
        self.pair1 = np.asarray(pair1).astype(np.int64)
        self.is_end_point = np.asarray(is_end_point).astype(bool)
        idx = np.zeros_like(self.is_end_point).astype(np.int64)
        idx[self.is_end_point] = 15
        vertex_clusters = edges[self.pair1, idx]

        mean_dis = np.linalg.norm(vertex_clusters[:, 0] - vertex_clusters[:, 1], axis=1) + \
                   np.linalg.norm(vertex_clusters[:, 1] - vertex_clusters[:, 2], axis=1) + \
                   np.linalg.norm(vertex_clusters[:, 2] - vertex_clusters[:, 0], axis=1)
        mean_dis /= 3

        flag = np.ones_like(self.is_end_point[:, 0])
        flag[mean_dis > v_threshold] = 0
        self.pair1 = self.pair1[flag]
        self.is_end_point = self.is_end_point[flag]
        pass

    def remove_isolated_edges(self):
        # Using the pair to fix the isolate edges
        # If the edge is not in the pair, and it is not closed, then it is isolated edge
        if self.pair1 is None:
            return
        edge_face_connectivity = self.edge_face_connectivity
        is_edge_in_pair = np.zeros(self.recon_edge_points.shape[0], dtype=bool)
        pair_edge_idx = np.unique(self.pair1.flatten())
        is_edge_closed = self.openness
        is_edge_in_pair[pair_edge_idx] = True
        is_edge_isolate = np.logical_and(~is_edge_in_pair, ~is_edge_closed)
        for iso_edge_idx in np.where(is_edge_isolate)[0]:
            edge_face1_face2 = edge_face_connectivity[edge_face_connectivity[:, 0] == iso_edge_idx, 1:]
            if edge_face1_face2.shape[0] == 0:
                continue
            face_idx1, face_idx2 = edge_face1_face2[0]
            if iso_edge_idx in self.face_edge_adj[face_idx1]:
                self.face_edge_adj[face_idx1].remove(iso_edge_idx)
            elif iso_edge_idx in self.face_edge_adj[face_idx2]:
                self.face_edge_adj[face_idx2].remove(iso_edge_idx)
            edge_face_connectivity = edge_face_connectivity[edge_face_connectivity[:, 0] != iso_edge_idx]
            self.remove_edge_idx_new.append(iso_edge_idx)

    def drop_edges(self, max_drop_num=2):
        recon_edge_points = self.recon_edge_points
        remove_edges_idx = [[] for _ in range(self.recon_face_points.shape[0])]
        for face_idx, face_edge_adj_c in enumerate(self.face_edge_adj):
            if len(face_edge_adj_c) == 0:
                continue
            max_combination_num = min(len(face_edge_adj_c) - 1, max_drop_num)
            all_combinations = []
            for combinations_num in range(1, max_combination_num + 1):
                if len(face_edge_adj_c) - combinations_num == 0:
                    continue
                all_combinations += list(combinations(face_edge_adj_c, len(face_edge_adj_c) - combinations_num))
            all_combinations += [face_edge_adj_c]

            connected_loss = []
            for sampled_face_edge_adj in all_combinations:
                # cannot remove the closed edge
                dropp_edges = list(set(face_edge_adj_c) - set(sampled_face_edge_adj))
                if len(dropp_edges) != 0 and self.openness[dropp_edges[0]]:
                    connected_loss.append(1e6)
                    continue
                face_edges = recon_edge_points[list(sampled_face_edge_adj)]
                face_edge_endpoint = np.concatenate([face_edges[:, 0], face_edges[:, -1]])
                dist_matrix = np.linalg.norm(face_edge_endpoint[:, np.newaxis] - face_edge_endpoint, axis=2)
                dist_matrix = dist_matrix + np.eye(dist_matrix.shape[0]) * 1e6
                connected_loss_c = dist_matrix.min(axis=0).sum()
                connected_loss.append(connected_loss_c)
            if len(connected_loss) <= 1:
                continue
            best_combination_idx = np.argmin(connected_loss)
            if best_combination_idx < len(all_combinations) - 1:
                for edge_idx in face_edge_adj_c:
                    if edge_idx not in all_combinations[best_combination_idx]:
                        if not self.openness[edge_idx]:
                            remove_edges_idx[face_idx].append(edge_idx)

        remove_edges_idx_real = []
        for face_idx, edge_idx_list in enumerate(remove_edges_idx):
            for edge_idx in edge_idx_list:
                idx = np.where(self.edge_face_connectivity[:, 0] == edge_idx)[0]
                if len(idx) == 0:
                    continue
                face_idx1, face_idx2 = self.edge_face_connectivity[idx[0], 1:]
                another_face_idx = face_idx1 if face_idx1 != face_idx else face_idx2
                if edge_idx not in remove_edges_idx[another_face_idx]:
                    continue
                remove_edges_idx_real.append(edge_idx)
                self.edge_face_connectivity = np.delete(self.edge_face_connectivity, idx, axis=0)
                if edge_idx in self.face_edge_adj[face_idx1]:
                    self.face_edge_adj[face_idx1].remove(edge_idx)
                if edge_idx in self.face_edge_adj[face_idx2]:
                    self.face_edge_adj[face_idx2].remove(edge_idx)
        self.remove_edge_idx_new.extend(np.unique(remove_edges_idx_real))

    def build_geom(self, is_replace_edge=False, connected_tolerance=0.1):
        # self.recon_geom_faces = [create_surface(points) for points in self.recon_face_points]
        # self.recon_curves = [create_edge(points) for points in self.recon_edge_points]
        # self.recon_topo_faces = [BRepBuilderAPI_MakeFace(geom_face, TRANSFER_PRECISION).Face() for geom_face in self.recon_geom_faces]
        # self.recon_edge = [BRepBuilderAPI_MakeEdge(curve).Edge() for curve in self.recon_curves]

        self.clinder_face_idx = []
        self.replace_edge_idx = []
        if not is_replace_edge:
            return

        for face_idx, geom_face in enumerate(self.recon_geom_faces):
            face_edges_idx_list = self.face_edge_adj[face_idx]
            # face_edges = [self.recon_edge[edge_idx] for edge_idx in face_edges_idx_list]
            # connected_wire = create_wire_from_unordered_edges(face_edges, connected_tolerance)
            topo_face = self.recon_topo_faces[face_idx]
            if ((geom_face.IsUPeriodic() or geom_face.IsVPeriodic()) and len(face_edges_idx_list) == 2
                    and len(get_primitives(topo_face, TopAbs_WIRE)) == 1):
                self.clinder_face_idx.append(face_idx)
                face_edge_from_geom_face = get_primitives(topo_face, TopAbs_EDGE)
                for edge_idx in face_edges_idx_list:
                    recon_topo_edge = self.recon_topo_curves[edge_idx]
                    near_edge_from_geom_face = []
                    for edge_from_geom_face in face_edge_from_geom_face:
                        is_similar, dis = check_edges_similarity(recon_topo_edge, edge_from_geom_face)
                        if is_similar:
                            near_edge_from_geom_face.append((edge_from_geom_face, dis))
                    # assert len(near_edge_from_geom_face) == 1
                    if len(near_edge_from_geom_face) == 0:
                        continue
                    if len(near_edge_from_geom_face) > 1:
                        near_edge_from_geom_face = sorted(near_edge_from_geom_face, key=lambda x: x[1])
                    self.recon_topo_curves[edge_idx] = near_edge_from_geom_face[0][0]
                    self.replace_edge_idx.append(edge_idx)
        pass


def apply_transform_batch(tensor, transform):
    src_shape = tensor.shape
    if len(src_shape) > 3:
        tensor = tensor.reshape(tensor.shape[0], -1, 3)
    scales = transform[:, :3].view(-1, 1, 3)
    offsets = transform[:, 3:].view(-1, 1, 3)
    centers = tensor.mean(dim=1, keepdim=True)
    scaled_tensor = (tensor - centers) * scales + centers + offsets
    if len(src_shape) > 3:
        scaled_tensor = scaled_tensor.reshape(*src_shape)
    return scaled_tensor


def apply_offset_batch(tensor, offsets):
    src_shape = tensor.shape
    if len(src_shape) > 3:
        tensor = tensor.reshape(tensor.shape[0], -1, 3)
    offsets = offsets.view(-1, 1, 3)
    scaled_tensor = tensor + offsets
    if len(src_shape) > 3:
        scaled_tensor = scaled_tensor.reshape(*src_shape)
    return scaled_tensor


def optimize(
        v_interpolation_face, recon_edge_points, recon_face_points,
        edge_face_connectivity, is_end_point, pair1,
        face_edge_adj, v_islog=True, v_max_iter=1000, use_cuda=True):
    device = torch.device('cuda') if torch.cuda.is_available() and use_cuda else torch.device('cpu')
    interpolation_face = []
    for item in v_interpolation_face:
        # interpolation_face.append(torch.from_numpy(item.copy()).to(device))
        interpolation_face.append(item.to(device))
    padded_points = pad_sequence(interpolation_face, batch_first=True, padding_value=10)
    edge_points = torch.from_numpy(recon_edge_points.copy()).to(device)
    face_points = torch.from_numpy(recon_face_points.copy()).to(device)
    edge_face_connectivity = torch.from_numpy(edge_face_connectivity.copy()).to(device)
    if pair1 is not None:
        pair1 = torch.from_numpy(pair1.copy()).to(device)
    idx = np.zeros_like(is_end_point).astype(np.int64)
    idx[is_end_point] = 15
    idx = torch.from_numpy(idx).to(device)

    edge_st = nn.Parameter(
            torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.float32, device=device).unsqueeze(0).repeat(edge_points.shape[0], 1))
    face_t = nn.Parameter(
            torch.tensor([0, 0, 0], dtype=torch.float32, device=device).unsqueeze(0).repeat(face_points.shape[0], 1))

    edge_src_st = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.float32, device=device).unsqueeze(0).repeat(edge_points.shape[0], 1)
    face_src_t = torch.tensor([0, 0, 0], dtype=torch.float32, device=device).unsqueeze(0).repeat(face_points.shape[0], 1)

    edge_st.requires_grad = True
    face_t.requires_grad = True
    optimizer = torch.optim.AdamW([edge_st, face_t], lr=1e-3, betas=(0.95, 0.999), eps=1e-08)
    # optimizer = torch.optim.SGD([edge_st, face_t], lr=5e-4)

    init_max_iter = v_max_iter
    final_max_iter = init_max_iter * 3
    init_loss = float('inf')
    best_loss = float('inf')
    is_optimization_diverged = False
    if v_islog:
        pbar = tqdm(total=v_max_iter, desc='Geom Optimization', unit='iter')

    chamferdist = ChamferDistance()
    iter = 0
    while iter < v_max_iter:
        transformed_edges = apply_transform_batch(edge_points, edge_st)
        transformed_padded_points = apply_offset_batch(padded_points, face_t)
        dis_matrix1 = chamferdist(
                transformed_edges[edge_face_connectivity[:, 0]],
                transformed_padded_points[edge_face_connectivity[:, 1]],
                batch_reduction=None, point_reduction="sum", bidirectional=False)
        dis_matrix2 = chamferdist(
                transformed_edges[edge_face_connectivity[:, 0]],
                transformed_padded_points[edge_face_connectivity[:, 2]],
                batch_reduction=None, point_reduction="sum", bidirectional=False)
        adj_distance_loss = (dis_matrix1 + dis_matrix2 + 1e-2 * (dis_matrix1 - dis_matrix2).abs()).mean()

        # For loop version
        # for edge_idx, face_idx1, face_idx2 in self.edge_face_connectivity:
        #     edge_to_face1 = self.dist(self.transformed_recon_edge[edge_idx], self.face_points_candidates[face_idx1])
        #     edge_to_face2 = self.dist(self.transformed_recon_edge[edge_idx], self.face_points_candidates[face_idx2])
        #
        #     adj_distance_loss_c = (edge_to_face1 + edge_to_face2) / 2
        #     adj_distance_loss += adj_distance_loss_c

        if pair1 is None:
            corners_loss = torch.zeros_like(adj_distance_loss)
        else:
            corners = transformed_edges[pair1, idx]
            corners_loss = torch.linalg.norm(corners[:, 0] - corners[:, 1], dim=-1) + \
                           torch.linalg.norm(corners[:, 1] - corners[:, 2], dim=-1) + \
                           torch.linalg.norm(corners[:, 0] - corners[:, 2], dim=-1)
            corners_loss = (corners_loss / 3).mean()

        if len(face_edge_adj) == 0:
            wire_connected_loss = torch.zeros_like(adj_distance_loss)
        else:
            max_length = max(len(face_edge_idx) for face_edge_idx in face_edge_adj)
            all_endpoints = []
            for face_edge_idx in face_edge_adj:
                if len(face_edge_idx) == 0:
                    continue
                face_edge_endpoint = torch.cat((transformed_edges[face_edge_idx, 0, :],
                                                transformed_edges[face_edge_idx, -1, :]), dim=0)
                padded_endpoints = torch.nn.functional.pad(face_edge_endpoint, (0, 0, 0, max_length * 2 - face_edge_endpoint.shape[0]),
                                                           value=1e6)
                all_endpoints.append(padded_endpoints)

            batch_endpoints = torch.stack(all_endpoints)
            dist_matrix = torch.cdist(batch_endpoints, batch_endpoints)
            dist_matrix = dist_matrix + torch.eye(dist_matrix.shape[-1], device=device).unsqueeze(0) * 1e6
            connected_loss = dist_matrix.min(dim=-1)[0]
            wire_connected_loss = connected_loss[connected_loss < 100].mean()

        loss = (adj_distance_loss + corners_loss + wire_connected_loss +
                1e-8 * (edge_st - edge_src_st).norm(p=1, dim=1).sum() + 1e-8 * (face_t - face_src_t).norm(p=1, dim=1).sum())

        if iter == 0:
            init_loss = loss.item()
        if loss.item() < best_loss:
            best_loss = loss.item()

        if iter > 30 and best_loss > init_loss:
            is_optimization_diverged = True
            print(f'Optimization diverged, best loss: {best_loss}, init loss: {init_loss}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if iter == v_max_iter - 2 and corners_loss.item() > 0.001:
            if v_max_iter + init_max_iter > final_max_iter:
                v_max_iter = final_max_iter
            else:
                v_max_iter += init_max_iter
            if v_islog:
                pbar.total = v_max_iter
        if v_islog:
            pbar.set_postfix(loss=loss.item(),
                             adj=adj_distance_loss.cpu().item(),
                             corner=corners_loss.cpu().item(),
                             connect=wire_connected_loss.cpu().item())
            pbar.update(1)
        iter += 1

    if v_islog:
        print('Optimization finished!')
        pbar.close()

    if is_optimization_diverged:
        return recon_face_points, recon_edge_points

    transformed_edges = apply_transform_batch(edge_points, edge_st).detach().cpu().numpy()
    transformed_faces = apply_offset_batch(face_points, face_t).detach().cpu().numpy()
    return transformed_faces, transformed_edges


# @ray.remote(num_gpus=0.05)
def optimize_ray(interpolation_face, recon_edge_points,
                 edge_face_connectivity, is_end_point, pair1, face_edge_adj, v_max_iter=1000):
    return optimize(interpolation_face, recon_edge_points,
                    edge_face_connectivity, is_end_point, pair1, face_edge_adj, v_islog=False, v_max_iter=v_max_iter)


def get_edge_vertexes(edge):
    vertex_explorer = TopExp_Explorer(edge, TopAbs_VERTEX)
    vertexes = []
    while vertex_explorer.More():
        vertex = topods.Vertex(vertex_explorer.Current())
        vertexes.append(vertex)
        vertex_explorer.Next()
    return vertexes


def explore_edges(shape):
    edge_explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    edges = []
    while edge_explorer.More():
        edge = topods.Edge(edge_explorer.Current())
        edges.append(edge)
        edge_explorer.Next()
    return edges


def get_edge_length(edge, NUM_SEGMENTS=100):
    curve_data = BRep_Tool.Curve(edge)
    if curve_data and len(curve_data) == 3:
        curve_handle, first, last = curve_data
        segment_length = (last - first) / NUM_SEGMENTS
        edge_length = 0
        for i in range(NUM_SEGMENTS):
            u1 = first + segment_length * i
            u2 = first + segment_length * (i + 1)
            edge_length += np.linalg.norm(np.array(curve_handle.Value(u1).Coord()) - np.array(curve_handle.Value(u2).Coord()))
        return edge_length
    else:
        return 0


def check_edges_similarity(edge1, edge2, dis_threshold=1e-1, unit_sample_num=1000):
    def sample_edge(edge):
        sample_num = min(int(unit_sample_num * get_edge_length(edge)), 1000)
        curve_data = BRep_Tool.Curve(edge)
        if curve_data and len(curve_data) == 3:
            curve_handle, first, last = curve_data
            points = []
            for i in range(sample_num):
                param = first + (last - first) * i / (sample_num - 1)
                point = curve_handle.Value(param)
                points.append(point)
            return points
        else:
            return None

    edge1_sample_points = sample_edge(edge1)
    edge2_sample_points = sample_edge(edge2)

    if edge1_sample_points is None or edge2_sample_points is None:
        return False, -1

    dis_list1 = []
    for p1, p2 in zip(edge1_sample_points, edge2_sample_points):
        dis = p1.Distance(p2)
        dis_list1.append(dis)

    dis_list2 = []
    for p1, p2 in zip(edge1_sample_points, edge2_sample_points[::-1]):
        dis = p1.Distance(p2)
        dis_list2.append(dis)

    dis_list = dis_list1 if sum(dis_list1) < sum(dis_list2) else dis_list2

    is_similar = all([dis < dis_threshold for dis in dis_list])
    return is_similar, np.mean(dis_list)


def calculate_wire_bounding_box_length(wire):
    bbox = Bnd_Box()
    brepbnd = brepbndlib()
    brepbndlib.Add(wire, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    length = (xmax - xmin) + (ymax - ymin) + (zmax - zmin)
    return length


def viz_shapes(shapes, transparency=0, backend_str=None):
    display, start_display, add_menu, add_function_to_menu = init_display(backend_str=backend_str)
    if len(shapes) == 1:
        display.DisplayShape(shapes[0], update=True, transparency=transparency)
    else:
        for shape in shapes:
            display.DisplayShape(shape, update=True, color=Colors.random_color(), transparency=transparency)
    display.FitAll()
    start_display()


def check_edges_in_face(face, edges, dist_tol=CONNECT_TOLERANCE):
    for edge in edges:
        edge_vertexes = get_edge_vertexes(edge)
        for vertex in edge_vertexes:
            dist_shape_shape = BRepExtrema_DistShapeShape(vertex, face)
            min_dist = dist_shape_shape.Value()
            if min_dist > dist_tol:
                return False
    return True


class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

    @staticmethod
    def random_color():
        return Quantity_Color(randint(0, 128) / 255.0,
                              randint(0, 128) / 255.0,
                              randint(0, 128) / 255.0,
                              Quantity_TOC_RGB)


def create_surface(points, use_variational_smoothing=USE_VARIATIONAL_SMOOTHING):
    def fit_face(uv_points_array, precision, use_variational_smoothing=USE_VARIATIONAL_SMOOTHING):
        if use_variational_smoothing:
            # weight_CurveLength, weight_Curvature, weight_Torsion = 1, 1, 1
            return GeomAPI_PointsToBSplineSurface(uv_points_array, weight_CurveLength, weight_Curvature, weight_Torsion,
                                                  face_fiting_deg_max, CONTINUITY, precision).Surface()
        else:
            return GeomAPI_PointsToBSplineSurface(uv_points_array, face_fiting_deg_min, face_fiting_deg_max, CONTINUITY,
                                                  precision).Surface()

    def set_face_uv_periodic(geom_face, points, tol=2e-3):
        u_intervals = np.sqrt(np.sum((points - np.roll(points, axis=0, shift=1)) ** 2, axis=2)).mean(axis=1)
        v_intervals = np.sqrt(np.sum((points - np.roll(points, axis=1, shift=1)) ** 2, axis=2)).mean(axis=0)

        u_min, u_max, v_min, v_max = geom_face.Bounds()
        us = geom_face.Value(u_min, v_min)
        ue = geom_face.Value(u_max, v_min)
        if us.Distance(ue) < np.mean(u_intervals):
            geom_face.SetUPeriodic()

        vs = geom_face.Value(u_min, v_min)
        ve = geom_face.Value(u_min, v_max)
        if vs.Distance(ve) < np.mean(v_intervals):
            geom_face.SetVPeriodic()
        return geom_face

    def eval_fitting_face(approx_face, uv_points_array):
        errors = []
        key_points = uv_points_array[::4, ::4, :].reshape(-1, 3)
        # key_points = uv_points_array.reshape(-1, 3)
        for point in key_points:
            topo_face = BRepBuilderAPI_MakeFace(approx_face, TRANSFER_PRECISION).Face()
            vertex = BRepBuilderAPI_MakeVertex(gp_Pnt(float(point[0]), float(point[1]), float(point[2]))).Vertex()
            min_dist = BRepExtrema_DistShapeShape(vertex, topo_face).Value()
            errors.append(min_dist)
        rmse = np.sqrt(np.mean(np.array(errors)))
        max_error = np.max(np.array(errors))
        return rmse + max_error

    num_u_points, num_v_points = points.shape[0], points.shape[1]
    uv_points_array = TColgp_Array2OfPnt(1, num_u_points, 1, num_v_points)
    for u_index in range(1, num_u_points + 1):
        for v_index in range(1, num_v_points + 1):
            pt = points[u_index - 1, v_index - 1]
            point_3d = gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2]))
            uv_points_array.SetValue(u_index, v_index, point_3d)

    approx_face_list = []
    error_list = []
    for precision in FACE_FITTING_TOLERANCE:
        try:
            approx_face = fit_face(uv_points_array, precision, use_variational_smoothing)
            error = eval_fitting_face(approx_face, points)
            approx_face_list.append(approx_face)
            error_list.append(error)
            if error < 1e-2:
                break
        except Exception as e:
            continue

    if len(approx_face_list) > 0:
        approx_face = approx_face_list[np.argmin(error_list)]
    else:
        approx_face = fit_face(uv_points_array, ROUGH_FITTING_TOLERANCE, use_variational_smoothing=False)

    approx_face = set_face_uv_periodic(approx_face, points)

    return approx_face


def create_edge(points, use_variational_smoothing=USE_VARIATIONAL_SMOOTHING):
    def fit_edge(u_points_array, precision, use_variational_smoothing=USE_VARIATIONAL_SMOOTHING):
        if use_variational_smoothing:
            # weight_CurveLength, weight_Curvature, weight_Torsion = 1, 1, 1
            return GeomAPI_PointsToBSpline(u_points_array, weight_CurveLength, weight_Curvature, weight_Torsion,
                                           edge_fitting_deg_max, CONTINUITY, precision).Curve()
        else:
            return GeomAPI_PointsToBSpline(u_points_array, edge_fitting_deg_min, edge_fitting_deg_max, CONTINUITY, precision).Curve()

    def eval_fitting_edge(approx_edge, u_points_array):
        errors = []
        key_points = u_points_array[::2, :].reshape(-1, 3)
        for point in key_points:
            topo_edge = BRepBuilderAPI_MakeEdge(approx_edge).Edge()
            vertex = BRepBuilderAPI_MakeVertex(gp_Pnt(float(point[0]), float(point[1]), float(point[2]))).Vertex()
            min_dist = BRepExtrema_DistShapeShape(vertex, topo_edge).Value()
            errors.append(min_dist)
        rmse = np.sqrt(np.mean(np.array(errors)))
        max_error = np.max(np.array(errors))
        return rmse + max_error

    num_u_points = points.shape[0]
    u_points_array = TColgp_Array1OfPnt(1, num_u_points)
    for u_index in range(1, num_u_points + 1):
        pt = points[u_index - 1]
        point_2d = gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2]))
        u_points_array.SetValue(u_index, point_2d)

    approx_edge_list = []
    error_list = []
    for precision in EDGE_FITTING_TOLERANCE:
        try:
            approx_edge = fit_edge(u_points_array, precision, use_variational_smoothing)
            error = eval_fitting_edge(approx_edge, points)
            approx_edge_list.append(approx_edge)
            error_list.append(error)
            if error < 1e-2:
                break
        except Exception as e:
            continue

    if len(approx_edge_list) > 0:
        approx_edge = approx_edge_list[np.argmin(error_list)]
    else:
        approx_edge = fit_edge(u_points_array, ROUGH_FITTING_TOLERANCE, use_variational_smoothing=False)

    return approx_edge


def create_wire_from_unordered_edges(face_edges, connected_tolerance, max_retry_times=3,
                                     is_sort_by_length=True, seam_required=False):
    # if seam_required!=-1:
    #     new_list = []
    #     for edge in face_edges:
    #         new_list.append(edge)
    #     new_list.append(face_edges[seam_required].Reversed())
    #     face_edges = new_list

    wire_array = None
    for i in range(max_retry_times):
        random.shuffle(face_edges)
        edges_seq = TopTools_HSequenceOfShape()
        for edge in face_edges:
            edges_seq.Append(edge)
        wire_array_c = ShapeAnalysis_FreeBounds.ConnectEdgesToWires(edges_seq, connected_tolerance, False)

        # Check if all wires is valid
        all_wire_valid = True
        for i in range(1, wire_array_c.Length() + 1):
            wire_c = wire_array_c.Value(i)
            wire_analyzer = BRepCheck_Analyzer(wire_c)
            if not wire_analyzer.IsValid():
                all_wire_valid = False
                break
        if not all_wire_valid:
            break

        if wire_array is None:
            wire_array = wire_array_c
            continue

        if wire_array_c.Length() < wire_array.Length():
            wire_array = wire_array_c

    if wire_array is None or wire_array.Length() == 0:
        return None

    # if wire_array.Length() != 1:
    #     wire_array_c = ShapeAnalysis_FreeBounds.ConnectWiresToWires(wire_array, connected_tolerance, False)

    wire_list = [wire_array.Value(i) for i in range(1, wire_array.Length() + 1)]

    if is_sort_by_length:
        wire_list = sorted(wire_list, key=calculate_wire_bounding_box_length, reverse=True)

    return wire_list


def _edge_endpoints_np(edge):
    """3D start/end points of an edge's curve, as numpy arrays (or None if no 3D curve)."""
    cd = BRep_Tool.Curve(edge)
    if cd is None or cd[0] is None:
        return None
    c, f, l = cd
    p0, p1 = c.Value(f), c.Value(l)
    return (np.array([p0.X(), p0.Y(), p0.Z()]), np.array([p1.X(), p1.Y(), p1.Z()]))


def create_wire_from_ordered_edges(face_edges, connected_tolerance, max_jump=0.15):
    """Backup wire builder: assemble a face's boundary edges into closed loops by chaining
    each edge to the nearest free endpoint (no global distance threshold), then build one wire
    per loop and weld its (small) junction gaps with ShapeFix_Wire.

    ``ConnectEdgesToWires`` reconnects an *unordered* bag of edges with a single global tolerance,
    which cannot both bridge the largest junction gap of a big face and avoid over-merging distinct
    vertices in dense regions -- so it leaves open wires. Here the edges are already the boundary of
    one face (half-edges owned by it), so nearest-endpoint chaining recovers the loop order and each
    gap is welded only between the two genuinely-consecutive edges. Returns a list of wires (largest
    first) or None."""
    eps, edges = [], []
    for e in face_edges:
        pe = _edge_endpoints_np(e)
        if pe is not None:
            edges.append(e)
            eps.append(pe)
    n = len(edges)
    if n < 2:
        return None

    used = [False] * n
    loops = []          # each: (list of (edge_idx, reversed), max_gap)
    for s in range(n):
        if used[s]:
            continue
        used[s] = True
        order = [(s, False)]
        head, cur = eps[s][0], eps[s][1]
        gaps = []
        while True:
            best, best_d, best_rev = -1, 1e18, False
            for j in range(n):
                if used[j]:
                    continue
                d0 = float(np.linalg.norm(eps[j][0] - cur))
                d1 = float(np.linalg.norm(eps[j][1] - cur))
                if d0 < best_d:
                    best, best_d, best_rev = j, d0, False
                if d1 < best_d:
                    best, best_d, best_rev = j, d1, True
            if best < 0 or best_d > max_jump:
                break
            used[best] = True
            order.append((best, best_rev))
            gaps.append(best_d)
            cur = eps[best][0] if best_rev else eps[best][1]
        closure = float(np.linalg.norm(cur - head))
        loops.append((order, max(gaps + [closure]) if gaps else closure))

    wire_list = []
    for order, gap in loops:
        if len(order) < 2:
            continue
        wd = ShapeExtend_WireData()
        for idx, rev in order:
            e = edges[idx]
            wd.Add(e.Reversed() if rev else e)
        tol = max(connected_tolerance, gap * 1.5)
        fixer = ShapeFix_Wire()
        fixer.Load(wd.Wire())
        fixer.SetPrecision(tol)
        fixer.SetMaxTolerance(tol)
        fixer.SetClosedWireMode(True)
        fixer.FixReorder()
        fixer.FixConnected()
        fixer.FixGaps3d()
        fixer.FixClosed()
        wire_list.append(fixer.Wire())
    if not wire_list:
        return None
    return sorted(wire_list, key=calculate_wire_bounding_box_length, reverse=True)


def set_tolerance(v_item, v_precision):
    tolorancer = ShapeFix_ShapeTolerance()
    tolorancer.SetTolerance(v_item, v_precision)
    return v_item


def get_tolerance(v_item, v_type):
    tolorancer = ShapeAnalysis_ShapeTolerance()
    return tolorancer.Tolerance(v_item, v_type)


def create_trimmed_face_from_wire(geom_face, face_edges, wire_list, connected_tolerance):
    topo_face = BRepBuilderAPI_MakeFace(geom_face, 1e-6).Face()
    is_debug = False
    if len(wire_list) == 6:
        # is_debug=True
        pass
    if (geom_face.IsUPeriodic() or geom_face.IsVPeriodic()) and len(get_primitives(topo_face, TopAbs_WIRE)) == 1 and len(face_edges) == 2:
        length_edge1 = get_edge_length(face_edges[0])
        length_edge2 = get_edge_length(face_edges[1])
        if abs(length_edge1 - length_edge2) < 0.01:
            final_wire = get_primitives(topo_face, TopAbs_WIRE)[0]
            final_wire = set_tolerance(final_wire, connected_tolerance)
            wire_list = [final_wire]
            # is_debug=True
            pass
    face_fixer = ShapeFix_Face()
    face_fixer.Init(geom_face, connected_tolerance, True)
    fixed_wire_list = []
    for wire in wire_list:
        wire_fixer = ShapeFix_Wire(wire, topo_face, connected_tolerance)
        wire_fixer.SetModifyTopologyMode(True)
        wire_fixer.SetModifyGeometryMode(True)
        wire_fixer.FixSmall(False, REMOVE_EDGE_TOLERANCE)
        wire_fixer.SetMaxTolerance(connected_tolerance)
        wire_fixer.SetPrecision(connected_tolerance)
        wire_fixer.FixGaps3d()
        # wire_fixer.FixGaps2d()
        # wire_fixer.Perform()

        # when only one edge, and being gap fixing, but still not closed, skip
        # if wire_fixer.Wire().NbChildren() == 1 and not wire_fixer.Wire().Closed():
        #     continue

        # try to fix the missing edge when mutil edges are connected
        # if wire_fixer.Wire().NbChildren() > 1:
        #     wire_fixer.SetMaxTolerance(FIX_CLOSE_TOLERANCE)
        #     wire_fixer.SetPrecision(FIX_PRECISION)
        #     wire_fixer.SetClosedWireMode(True)
        #     wire_fixer.FixClosed()

        fixed_wire = wire_fixer.Wire()

        # assert fixed_wire.Closed()
        # if not fixed_wire.Closed():
        #     continue

        fixed_wire = set_tolerance(fixed_wire, connected_tolerance)
        face_fixer.Add(fixed_wire)
        fixed_wire_list.append(fixed_wire)

    if len(fixed_wire_list) == 0:
        return None

    try:
        face_fixer.FixWireTool().SetModifyGeometryMode(True)
        face_fixer.FixWireTool().SetMaxTolerance(connected_tolerance)
        face_fixer.FixWireTool().SetPrecision(connected_tolerance)
        face_fixer.FixWireTool().SetFixShiftedMode(True)
        face_fixer.FixWireTool().SetClosedWireMode(True)
        # face_fixer.FixWireTool().Perform()

        face_fixer.SetAutoCorrectPrecisionMode(False)
        face_fixer.SetPrecision(connected_tolerance)
        face_fixer.SetMaxTolerance(connected_tolerance)
        face_fixer.SetFixOrientationMode(True)
        face_fixer.SetFixMissingSeamMode(True)
        face_fixer.SetFixWireMode(True)
        face_fixer.SetFixLoopWiresMode(False)
        face_fixer.SetFixIntersectingWiresMode(False)
        face_fixer.SetFixPeriodicDegeneratedMode(False)
        face_fixer.SetFixSmallAreaWireMode(False)
        face_fixer.Perform()

        # face_fixer.FixAddNaturalBound()
        face_fixer.FixOrientation()
        face_fixer.FixMissingSeam()
        # face_fixer.FixWiresTwoCoincEdges()
        face_fixer.FixIntersectingWires()
        # face_fixer.FixPeriodicDegenerated()
        face_fixer.FixOrientation()

    except Exception as e:
        print(f"Error fixing face {e}")
        return None

    face_occ = face_fixer.Face()
    if face_occ.IsNull():
        return None
    face_occ = set_tolerance(face_occ, connected_tolerance)

    if is_debug:
        viz_shapes([face_occ])

    face_analyzer = BRepCheck_Analyzer(face_occ)
    if face_analyzer.IsValid():
        return face_occ
    else:
        return None


def drop_edges(face_edges_np, is_edge_closed, drop_edge_num=0, accepted_connected_loss=0.2):
    saved_edge_idx = list(combinations(range(face_edges_np.shape[0]), face_edges_np.shape[0] - drop_edge_num))
    saved_edge_idx += [list(range(face_edges_np.shape[0]))]
    connected_loss = []
    for edge_idx in saved_edge_idx:
        drop_edge_idx = list(set(range(face_edges_np.shape[0])) - set(edge_idx))
        if len(drop_edge_idx) != 0 and is_edge_closed[drop_edge_idx[0]]:
            connected_loss.append(1e6)
            continue
        new_face_edges_np = face_edges_np[list(edge_idx)]
        face_edges_endpoints = np.concatenate([new_face_edges_np[:, 0], new_face_edges_np[:, -1]])
        dist_matrix = np.linalg.norm(face_edges_endpoints[:, np.newaxis] - face_edges_endpoints, axis=2)
        dist_matrix = dist_matrix + np.eye(dist_matrix.shape[0]) * 1e6
        connected_loss_c = dist_matrix.min(axis=0).sum()
        connected_loss.append(connected_loss_c)

    connected_loss = np.array(connected_loss)
    accepted_combination_idx = list(np.where(connected_loss < accepted_connected_loss)[0])
    if len(connected_loss) - 1 in accepted_combination_idx:
        accepted_combination_idx.remove(len(connected_loss) - 1)

    if len(accepted_combination_idx) == 0:
        return None

    optional_drop_edge_idx = []
    for combinations_idx in accepted_combination_idx:
        drop_edge_idx = list(set(range(face_edges_np.shape[0])) - set(saved_edge_idx[combinations_idx]))
        optional_drop_edge_idx.append(drop_edge_idx)

    return optional_drop_edge_idx


def create_trimmed_face1(geom_face, face_edges, connected_tolerance, face_edges_numpy=None, is_edge_closed=None, drop_edge_num=0):
    if drop_edge_num > 0 and face_edges_numpy is not None:
        if len(face_edges) - drop_edge_num < 1:
            return None, None, False
        optional_drop_edge_idx = drop_edges(face_edges_numpy, is_edge_closed, drop_edge_num=drop_edge_num)
        if optional_drop_edge_idx is None:
            return None, None, False
        optional_face_edges = []
        for drop_edge_idx in optional_drop_edge_idx:
            optional_face_edges.append([face_edges[i] for i in range(len(face_edges)) if i not in drop_edge_idx])
    else:
        optional_face_edges = [face_edges]

    for face_edges in optional_face_edges:
        wire_list = create_wire_from_unordered_edges(face_edges, connected_tolerance)
        if wire_list is None:
            continue
        trimmed_face = create_trimmed_face_from_wire(geom_face, face_edges, wire_list, connected_tolerance)
        if trimmed_face is None or trimmed_face.IsNull():
            continue
        shape_tol_setter = ShapeFix_ShapeTolerance()
        shape_tol_setter.SetTolerance(trimmed_face, connected_tolerance)
        face_analyzer = BRepCheck_Analyzer(trimmed_face, False)
        is_face_valid = face_analyzer.IsValid()
        if is_face_valid:
            return wire_list, trimmed_face, is_face_valid
    return None, None, False


def create_trimmed_face2(geom_face, topo_face, face_edges, connected_tolerance):
    # try to find the replaced edge
    topo_face_edges = explore_edges(topo_face)
    replace_dict = {}
    for idx1, checked_edge in enumerate(face_edges):
        optional_edges = {}
        for idx2, replaced_edge in enumerate(topo_face_edges):
            is_similar, mean_dis = check_edges_similarity(checked_edge, replaced_edge)
            if is_similar:
                optional_edges[idx2] = mean_dis
        if len(optional_edges) == 0:
            continue
        optimal_edge_idx = min(optional_edges, key=optional_edges.get)
        replace_dict[idx1] = optimal_edge_idx
        face_edges[idx1] = topo_face_edges[optimal_edge_idx]
    return create_trimmed_face1(geom_face, face_edges, connected_tolerance)


def try_create_trimmed_face(geom_face, topo_face, face_edges, connected_tolerance, face_edges_numpy, is_edge_closed, max_drop_edge_num=2):
    wire_list1, trimmed_face1, is_face_valid1 = create_trimmed_face1(geom_face, face_edges, connected_tolerance)
    if is_face_valid1:
        return wire_list1, trimmed_face1, True

    for drop_edge_num in range(1, max_drop_edge_num + 1):
        wire_list1, trimmed_face1, is_face_valid1 = create_trimmed_face1(geom_face, face_edges, connected_tolerance, face_edges_numpy,
                                                                         is_edge_closed, drop_edge_num)
        if is_face_valid1:
            return wire_list1, trimmed_face1, True

    if trimmed_face1 is None:
        return wire_list1, trimmed_face1, False
    #
    # wire_list2, trimmed_face2, is_face_valid2 = create_trimmed_face2(geom_face, topo_face, face_edges,
    #                                                                  connected_tolerance)
    # if is_face_valid2:
    #     return wire_list2, trimmed_face2, True
    #
    # return wire_list2, trimmed_face2, False


def get_separated_surface(trimmed_faces, v_precision1=1e-3, v_precision2=1e-1):
    points = []
    faces = []
    num_points = 0
    for face in trimmed_faces:
        loc = TopLoc_Location()
        mesh = BRepMesh_IncrementalMesh(face, v_precision1, False, v_precision2)
        triangulation = BRep_Tool.Triangulation(face, loc)
        if triangulation is None:
            continue

        v_points = np.zeros((triangulation.NbNodes(), 3), dtype=np.float32)
        f_faces = np.zeros((triangulation.NbTriangles(), 3), dtype=np.int64)
        for i in range(0, triangulation.NbNodes()):
            pnt = triangulation.Node(i + 1)
            v_points[i, 0] = pnt.X()
            v_points[i, 1] = pnt.Y()
            v_points[i, 2] = pnt.Z()
        for i in range(0, triangulation.NbTriangles()):
            tri = triangulation.Triangles().Value(i + 1)
            f_faces[i, 0] = tri.Get()[0] + num_points - 1
            f_faces[i, 1] = tri.Get()[1] + num_points - 1
            f_faces[i, 2] = tri.Get()[2] + num_points - 1
        points.append(v_points)
        faces.append(f_faces)
        num_points += v_points.shape[0]
    if len(points) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    points = np.concatenate(points, axis=0, dtype=np.float32)
    faces = np.concatenate(faces, axis=0, dtype=np.int64)
    return points, faces


def get_solid(trimmed_faces, connected_tolerance):
    try:
        # Sew shells
        random.shuffle(trimmed_faces)
        sewing = BRepBuilderAPI_Sewing()
        sewing.SetTolerance(connected_tolerance)
        for face in trimmed_faces:
            sewing.Add(face)
        sewing.Perform()
        # sewing.Dump()
        sewn_shell = sewing.SewedShape()
        if sewn_shell.ShapeType() == TopAbs_COMPOUND:
            return None
            best_shell = None
            best_num = 0
            for shell in get_primitives(sewn_shell, TopAbs_SHELL):
                if best_shell is None or len(get_primitives(shell, TopAbs_FACE)) > best_num:
                    best_shell = shell
                    best_num = len(get_primitives(shell, TopAbs_FACE))
            if best_shell is None:
                return None
            sewn_shell = best_shell
        shape_tol_setter = ShapeFix_ShapeTolerance()
        shape_tol_setter.SetTolerance(sewn_shell, connected_tolerance)

        # Fix if it is not valid
        if not BRepCheck_Analyzer(sewn_shell).IsValid():
            fix_shell = ShapeFix_Shell(sewn_shell)
            fix_shell.SetPrecision(connected_tolerance)
            fix_shell.SetFixFaceMode(True)
            fix_shell.SetFixOrientationMode(True)
            fix_shell.Perform()
            sewn_shell = fix_shell.Shell()
            shape_tol_setter = ShapeFix_ShapeTolerance()
            shape_tol_setter.SetTolerance(sewn_shell, connected_tolerance)

        # Solid
        maker = BRepBuilderAPI_MakeSolid()
        maker.Add(sewn_shell)
        maker.Build()
        solid = maker.Solid()
        shape_tol_setter = ShapeFix_ShapeTolerance()
        shape_tol_setter.SetTolerance(solid, connected_tolerance)

        # Fix if it is not valid
        if not BRepCheck_Analyzer(solid).IsValid() or True:
            fix_solid = ShapeFix_Solid(solid)
            fix_solid.SetPrecision(connected_tolerance)
            fix_solid.SetMaxTolerance(connected_tolerance)
            fix_solid.SetFixShellMode(True)
            fix_solid.SetFixShellOrientationMode(True)
            fix_solid.SetCreateOpenSolidMode(False)
            fix_solid.Perform()
            solid = fix_solid.Solid()

        set_tolerance(solid, connected_tolerance)
        if solid.ShapeType() == TopAbs_SOLID and BRepCheck_Analyzer(solid).IsValid():
            return solid
        else:
            return None
    except Exception as e:
        print(e)
        return None


def get_compound(trimmed_faces, connected_tolerance):
    try:
        # Sew shells
        random.shuffle(trimmed_faces)
        sewing = BRepBuilderAPI_Sewing()
        sewing.SetTolerance(connected_tolerance)
        for face in trimmed_faces:
            sewing.Add(face)
        sewing.Perform()
        # sewing.Dump()
        sewn_shell = sewing.SewedShape()
        if sewn_shell is not None:
            return sewn_shell
        else:
            return None
    except Exception as e:
        print(e)
        return None


def construct_brep(v_shape, connected_tolerance, isdebug=False):
    debug_idx = []
    if isdebug:
        print(f"{Colors.GREEN}################################ 1. Fit primitives ################################{Colors.RESET}")

    recon_edge_points = v_shape.recon_edge_points
    recon_geom_faces = v_shape.recon_geom_faces
    recon_topo_faces = v_shape.recon_topo_faces
    recon_curves = v_shape.recon_geom_curves
    recon_edge = v_shape.recon_topo_curves
    FaceEdgeAdj = v_shape.face_edge_adj
    is_edge_closed = v_shape.openness

    if False:
        viz_shapes(recon_geom_faces, transparency=0.5)

    if isdebug:
        print(f"{Colors.GREEN}################################ 2. Trim Face ######################################{Colors.RESET}")
    # Cut surface by wire
    is_face_success_list = []
    trimmed_faces = []
    for idx, (geom_face, topo_face, face_edge_idx) in enumerate(zip(recon_geom_faces, recon_topo_faces, FaceEdgeAdj)):
        if isdebug:
            print(f"Process Face {idx}")
        if len(face_edge_idx) == 0:
            if isdebug:
                print(f"Face {idx} has no edge")
            is_face_success_list.append(False)
            continue
        face_edges = [recon_edge[edge_idx] for edge_idx in face_edge_idx]
        face_edges_numpy = recon_edge_points[face_edge_idx]
        is_edge_closed_c = [is_edge_closed[edge_idx] for edge_idx in face_edge_idx]
        wire_list, trimmed_face, is_valid = try_create_trimmed_face(geom_face, topo_face, face_edges, connected_tolerance,
                                                                    face_edges_numpy, is_edge_closed_c)
        is_valid = False if trimmed_face is None else is_valid

        if idx in debug_idx:
            # viz_shapes([geom_face, wire_list])
            pass

        if isdebug and not is_valid:
            print(f"Face {idx} is not valid{Colors.RESET}")

        is_face_success_list.append(is_valid)
        if is_valid:
            trimmed_faces.append(trimmed_face)

    result = [is_face_success_list, None, None]
    # if len(trimmed_faces) > 2:
    if len(trimmed_faces) > int(0.8 * len(recon_geom_faces)):
        v, f = get_separated_surface(trimmed_faces, v_precision1=0.1, v_precision2=0.2)
        separated_surface = trimesh.Trimesh(vertices=v, faces=f)
        result[1] = separated_surface
        result[2] = get_solid(trimmed_faces, connected_tolerance)

    if isdebug:
        print(f"{Colors.GREEN}################################ Construct Done ################################{Colors.RESET}")
    return result


def triangulate_face(v_face):
    loc = TopLoc_Location()
    triangulation = BRep_Tool.Triangulation(v_face, loc)

    if triangulation is None:
        # Mesh
        mesh = BRepMesh_IncrementalMesh(v_face, 0.01)
        triangulation = BRep_Tool.Triangulation(v_face, loc)
        if triangulation is None:
            return np.zeros((0, 3)), np.zeros((0, 3))

    v_points = np.zeros((triangulation.NbNodes(), 3), dtype=np.float32)
    f_faces = np.zeros((triangulation.NbTriangles(), 3), dtype=np.int64)
    for i in range(0, triangulation.NbNodes()):
        pnt = triangulation.Node(i + 1)
        v_points[i, 0] = pnt.X()
        v_points[i, 1] = pnt.Y()
        v_points[i, 2] = pnt.Z()
    for i in range(0, triangulation.NbTriangles()):
        tri = triangulation.Triangles().Value(i + 1)
        f_faces[i, 0] = tri.Get()[0] - 1
        f_faces[i, 1] = tri.Get()[1] - 1
        f_faces[i, 2] = tri.Get()[2] - 1
    return v_points, f_faces


def triangulate_shape(v_shape):
    exp = TopExp_Explorer(v_shape, TopAbs_FACE)
    points = []
    faces = []
    num_points = 0
    while exp.More():
        face = topods.Face(exp.Current())

        loc = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, loc)

        if triangulation is None:
            # Mesh
            mesh = BRepMesh_IncrementalMesh(face, 0.1)
            triangulation = BRep_Tool.Triangulation(face, loc)
            if triangulation is None:
                exp.Next()
                continue

        v_points = np.zeros((triangulation.NbNodes(), 3), dtype=np.float32)
        f_faces = np.zeros((triangulation.NbTriangles(), 3), dtype=np.int64)
        for i in range(0, triangulation.NbNodes()):
            pnt = triangulation.Node(i + 1)
            v_points[i, 0] = pnt.X()
            v_points[i, 1] = pnt.Y()
            v_points[i, 2] = pnt.Z()
        for i in range(0, triangulation.NbTriangles()):
            tri = triangulation.Triangles().Value(i + 1)
            f_faces[i, 0] = tri.Get()[0] + num_points - 1
            f_faces[i, 1] = tri.Get()[1] + num_points - 1
            f_faces[i, 2] = tri.Get()[2] + num_points - 1
        points.append(v_points)
        faces.append(f_faces)
        num_points += v_points.shape[0]
        exp.Next()
    if len(points) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    points = np.concatenate(points, axis=0, dtype=np.float32)
    faces = np.concatenate(faces, axis=0, dtype=np.int64)
    return points, faces


"""
l_v: (num_edges, num_points(16), 3)
"""


def export_edges(l_v, v_file):
    with open(v_file, "w") as f:
        line_str = ""
        num_points = 0
        for edge in l_v:
            # line_str += f"l\n"
            for v in edge:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for i in range(0, edge.shape[0] - 1):
                line_str += f"l {i + num_points + 1} {i + num_points + 2}\n"
            num_points += edge.shape[0]
        f.write(line_str)


def solid_valid_check(solid, tolerance=0.01):
    # Refer to the SolidGen (https://arxiv.org/pdf/2203.13944#_ts1728973340413)
    # A.6 Criteria used to Evaluate Validity of Generated B-reps
    # We consider a B-rep to be valid if it was successfully built from the network’s output and additionally
    # satisfies the following criteria:
    # 1. Triangulatable. Every face in the B-rep must generate at least one triangle, otherwise it cannot be
    #   rendered properly and is not possible to manufacture.
    # 2. Wire ordering. The edges in each wire in the B-rep must be ordered correctly. We check this using
    #   the ShapeAnalysis_Wire::CheckOrder() function in PythonOCC (Paviot, 2008) with a tolerance of 0.01.
    # 3. No wire self-intersection. The wires should not self-intersect to ensure that faces are well-defined
    #   and surfaces trimmed correctly. We check this using the ShapeAnalysis_Wire::CheckSelfIntersection()
    #   function in PythonOCC with a tolerance of 0.01.
    # 4. No bad edges. The edges in shells should be present once or twice but with different orientations.
    # This can be checked with the ShapeAnalysis_Shell::HasBadEdges() function.

    # 1. Check Triangulatable
    face_exp = TopExp_Explorer(solid, TopAbs_FACE)
    while face_exp.More():
        face = topods.Face(face_exp.Current())
        loc = TopLoc_Location()
        mesh = BRepMesh_IncrementalMesh(face, 0.01)
        triangulation = BRep_Tool.Triangulation(face, loc)
        if triangulation is None:
            return False
        wire_exp = TopExp_Explorer(face, TopAbs_WIRE)
        while wire_exp.More():
            wire = topods.Wire(wire_exp.Current())
            wire_analyzer = ShapeAnalysis_Wire(wire, face, tolerance)
            # 2. Check wire ordering
            is_wire_ordered = wire_analyzer.CheckOrder(False)
            # is_wire_ordered = wire.Closed()
            if not is_wire_ordered:
                return False
            # 3. Check no wire self-intersection
            is_no_wire_self_intersection = wire_analyzer.CheckSelfIntersection()
            if not is_no_wire_self_intersection:
                return False
            wire_exp.Next()
        face_exp.Next()

    # 4. Check no bad edges
    shell_exp = TopExp_Explorer(solid, TopAbs_SHELL)
    while shell_exp.More():
        shell = topods.Shell(shell_exp.Current())
        shell_analyzer = ShapeAnalysis_Shell()
        shell_analyzer.LoadShells(shell)
        has_bad_edges = shell_analyzer.HasBadEdges()
        if has_bad_edges:
            return False
        shell_exp.Next()

    return True


def can_be_triangularized(shape):
    faces = get_primitives(shape, TopAbs_FACE)
    for face in faces:
        loc = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, loc)
        if triangulation is None:
            return False
    return True


def my_write_stp_file(v_shape, path):
    # initialize the STEP exporter
    step_writer = STEPControl_Writer()
    dd = step_writer.WS().TransferWriter().FinderProcess()
    Interface_Static.SetCVal("write.step.schema", "AP203")
    # transfer shapes and write file
    step_writer.Transfer(v_shape, STEPControl_AsIs)
    status = step_writer.Write(path)

    if status != IFSelect_RetDone:
        raise AssertionError("load failed")


def solid_to_faceadj_graph(solid, radius=0.01):
    import numpy as np
    from occwl.solid import Solid
    from occwl.viewer import Viewer
    from occwl.graph import face_adjacency

    solid = Solid(solid)
    g = face_adjacency(solid, self_loops=True)

    print(f"Number of nodes (faces): {len(g.nodes)}")
    print(f"Number of edges: {len(g.edges)}")

    v = Viewer()
    v.display(solid, transparency=0.8)

    # Get the points at each face's center
    face_centers = {}
    for face_idx in g.nodes():
        # Display a sphere for each face's center
        face = g.nodes[face_idx]["face"]
        parbox = face.uv_bounds()
        umin, vmin = parbox.min_point()
        umax, vmax = parbox.max_point()
        center_uv = (umin + 0.5 * (umax - umin), vmin + 0.5 * (vmax - vmin))
        center = face.point(center_uv)
        v.display(Solid.make_sphere(center=center, radius=radius))
        face_centers[face_idx] = center

    for fi, fj in g.edges():
        pt1 = face_centers[fi]
        pt2 = face_centers[fj]
        # Make a cylinder for each edge connecting a pair of faces
        up_dir = pt2 - pt1
        height = np.linalg.norm(up_dir)
        if height > 1e-3:
            v.display(
                    Solid.make_cylinder(
                            radius=radius, height=height, base_point=pt1, up_dir=up_dir
                    )
            )

    # Show the viewer
    v.fit()
    v.show()


def solid_to_vertadj_graph(solid, radius=0.01):
    import numpy as np
    from occwl.solid import Solid
    from occwl.viewer import Viewer
    from occwl.graph import vertex_adjacency

    solid = Solid(solid)
    g = vertex_adjacency(solid, self_loops=True)

    print(f"Number of nodes (vertices): {len(g.nodes)}")
    print(f"Number of edges: {len(g.edges)}")

    v = Viewer()
    v.display(solid, transparency=0.5)
    # Get the points for each vertex
    points = {}
    for vert_idx in g.nodes:
        pt = g.nodes[vert_idx]["vertex"].point()
        points[vert_idx] = pt
        v.display(Solid.make_sphere(center=pt, radius=radius))

    # Make a cylinder for each edge connecting a pair of vertices
    for vi, vj in g.edges:
        pt1 = points[vi]
        pt2 = points[vj]
        up_dir = pt2 - pt1
        if np.linalg.norm(up_dir) < 1e-6:
            # This must be a loop with one Vertex
            continue
        v.display(
                Solid.make_cylinder(
                        radius=radius, height=np.linalg.norm(up_dir), base_point=pt1, up_dir=up_dir
                )
        )

    # Show the viewer
    v.fit()
    v.show()
