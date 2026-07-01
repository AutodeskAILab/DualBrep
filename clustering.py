"""Segment a reconstructed BRep mesh into per-face patches using the UDF field.

Given a result folder containing ``recon_sdf.ply`` (the reconstructed surface) and
``udf_g.npy`` (per-face unsigned distance to the nearest BRep edge), ``process_item``
produces ``cluster.ply`` — the surface colored by face label. The default
``hierarchical`` mode grows watershed regions from low-UDF seeds and merges small
clusters. Operates on local folders; ``s3://`` folders are supported when
``cloudpathlib`` is installed.
"""
import shutil, copy
import tempfile
from pathlib import Path
from collections import defaultdict, deque

import numpy as np
import trimesh
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm

try:
    from cloudpathlib import S3Path
except Exception:  # cloudpathlib only needed for s3:// result folders
    S3Path = ()  # isinstance(x, ()) is always False -> local-only path


def download_file(remote_path, tmp_dir):
    """Download a single s3:// file into ``tmp_dir`` and return the local path."""
    from cloudpathlib import S3Path as _S3Path
    local = Path(tmp_dir) / Path(str(remote_path)).name
    _S3Path(str(remote_path)).download_to(str(local))
    return local


# Each result folder is expected to contain "recon_sdf.ply" and "udf_g.npy".
threshold = 0.01 # For coarse segmentation
threshold1 = 0.003 # For hierarchical segmentation pass 1
threshold2 = 0.007 # For hierarchical segmentation pass 2
mode = "hierarchical"  # "coarse", "hierarchical", "udf_mesh", "advanced"
is_merge_small = True  # Whether to merge small clusters in hierarchical segmentation
final_size_threshold = 15

def segment_mesh_by_brep_edges(mesh, vertices_udf,
                                discontinuity_threshold=0.5,
                                min_cluster_size=20):
    """
    Segment mesh faces based on UDF direction discontinuity.

    Args:
        mesh: trimesh object
        vertices_udf: (N, 4) array with distance and direction
        discontinuity_threshold: threshold for direction change (0-2, lower=stricter)
        min_cluster_size: minimum faces per cluster

    Returns:
        face_labels: (num_faces,) cluster ID per face
    """
    udf = vertices_udf[..., 0]
    directions = vertices_udf[..., 1:4]

    num_faces = len(mesh.faces)
    print(f"Segmenting {num_faces} faces...")

    # Step 1: Build adjacency graph
    print("Step 1: Building face adjacency graph...")
    face_adjacency = mesh.face_adjacency  # (num_pairs, 2)
    print(f"  Found {len(face_adjacency)} adjacent face pairs")

    # Step 2: Project directions to the plane of each triangle
    print("Step 2: Projecting directions to triangle planes...")
    face_normals = mesh.face_normals  # (num_faces, 3)

    face_directions = directions  # (num_faces, 3)

    # Normalize
    face_directions_norm = np.linalg.norm(face_directions, axis=1, keepdims=True)
    face_directions = face_directions / (face_directions_norm + 1e-8)

    # Project onto tangent plane: d_tangent = d - (d·n)n
    dot_with_normal = np.sum(face_directions * face_normals, axis=1, keepdims=True)
    face_directions_projected = face_directions - dot_with_normal * face_normals

    # Renormalize after projection
    projected_norm = np.linalg.norm(face_directions_projected, axis=1, keepdims=True)
    face_directions_projected = face_directions_projected / (projected_norm + 1e-8)

    print(f"  Projected {num_faces} face directions")

    # Step 3: Check adjacency according to dot product of directions
    print("Step 3: Computing direction discontinuity across edges...")

    # Get directions for adjacent face pairs
    dirs_f1 = face_directions_projected[face_adjacency[:, 0]]  # (num_pairs, 3)
    dirs_f2 = face_directions_projected[face_adjacency[:, 1]]  # (num_pairs, 3)

    # Compute dot products (cosine similarity)
    dot_products = np.sum(dirs_f1 * dirs_f2, axis=1)

    # Discontinuity score: 1 - |dot|
    # 0 = same direction, 2 = opposite direction
    edge_discontinuity = 1.0 - dot_products

    print(f"  Discontinuity range: [{edge_discontinuity.min():.3f}, {edge_discontinuity.max():.3f}]")
    print(f"  High discontinuity edges (>{discontinuity_threshold}): "
          f"{(edge_discontinuity > discontinuity_threshold).sum()} / {len(edge_discontinuity)}")

    # Step 4: Region growing to find clusters
    print("Step 4: Finding clusters via region growing...")

    # Build graph: connect faces if discontinuity is low
    G = nx.Graph()

    # Add edges where directions are similar
    for i, (f1, f2) in enumerate(face_adjacency):
        if edge_discontinuity[i] < discontinuity_threshold:
            G.add_edge(f1, f2)

    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Find connected components
    components = list(nx.connected_components(G))
    print(f"  Found {len(components)} initial clusters")

    # Assign labels
    face_labels = -np.ones(num_faces, dtype=int)
    for label_id, component in enumerate(components):
        for face_idx in component:
            face_labels[face_idx] = label_id
    unique_labels = np.unique(face_labels)
    label_mapping = {old: new for new, old in enumerate(unique_labels)}
    face_labels = np.array([label_mapping[l] for l in face_labels])

    # Return face_labels and debug data for visualization
    return face_labels, face_directions_projected, edge_discontinuity


def export_connectivity_segments(mesh, face_labels, face_directions, edge_discontinuity,
                                 discontinuity_threshold, output_path):
    """
    Export segments for boundary edges where discontinuity is above threshold.
    Shows where triangles are NOT connected due to direction change.

    Args:
        mesh: trimesh object
        face_labels: (num_faces,) cluster assignment
        face_directions: (num_faces, 3) projected directions
        edge_discontinuity: (num_pairs,) discontinuity scores
        discontinuity_threshold: threshold value
        output_path: path to save PLY file
    """
    print("\nExporting boundary segments for debugging...")

    # Get face centroids
    face_centroids = mesh.triangles_center  # (num_faces, 3)

    # Get face adjacency
    face_adjacency = mesh.face_adjacency  # (num_pairs, 2)

    # Find edges where discontinuity is ABOVE threshold (boundaries)
    boundary_pairs = []
    discontinuity_values = []

    for i, (f1, f2) in enumerate(face_adjacency):
        if edge_discontinuity[i] >= discontinuity_threshold:
            # High discontinuity - this is a boundary edge
            boundary_pairs.append((f1, f2))
            discontinuity_values.append(edge_discontinuity[i])

    n_boundaries = len(boundary_pairs)
    print(f"  Found {n_boundaries} boundary edges (discontinuity >= {discontinuity_threshold})")

    if n_boundaries == 0:
        print("  Warning: No boundary edges found!")
        return

    # Build vertex array: centroids of boundary faces
    vertices = []
    edges = []
    edge_colors = []

    # Color by discontinuity strength: blue (low) -> red (high)
    max_discontinuity = max(discontinuity_values) if discontinuity_values else 1.0

    for i, (f1, f2) in enumerate(boundary_pairs):
        # Add vertices (centroids)
        v1_idx = len(vertices)
        v2_idx = len(vertices) + 1

        vertices.append(face_centroids[f1])
        vertices.append(face_centroids[f2])

        # Add edge
        edges.append([v1_idx, v2_idx])

        # Color by discontinuity value: blue (threshold) -> red (max)
        # Normalize to [0, 1] range above threshold
        disc_normalized = (discontinuity_values[i] - discontinuity_threshold) / (max_discontinuity - discontinuity_threshold + 1e-6)
        disc_normalized = np.clip(disc_normalized, 0, 1)

        # Blue -> Cyan -> Green -> Yellow -> Red
        color_r = int(disc_normalized * 255)
        color_g = int((1 - abs(disc_normalized - 0.5) * 2) * 255)
        color_b = int((1 - disc_normalized) * 255)

        edge_colors.append([color_r, color_g, color_b])

    vertices = np.array(vertices)
    edges = np.array(edges)
    edge_colors = np.array(edge_colors)

    print(f"  Creating PLY with {len(vertices)} vertices and {len(edges)} edges")

    # Save as PLY with edges
    save_connectivity_ply(output_path, vertices, edges, edge_colors)

    print(f"  Saved connectivity segments to: {output_path}")


def save_connectivity_ply(filename, vertices, edges, colors):
    """
    Save connectivity segments to PLY file.

    Args:
        filename: output path
        vertices: (n_verts, 3) vertex coordinates
        edges: (n_edges, 2) edge indices
        colors: (n_edges, 3) RGB colors for edges
    """
    n_verts = len(vertices)
    n_edges = len(edges)

    with open(filename, 'w') as f:
        # PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n_verts}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element edge {n_edges}\n")
        f.write("property int vertex1\n")
        f.write("property int vertex2\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        # Write vertices
        for v in vertices:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")

        # Write edges with colors
        for i, edge in enumerate(edges):
            f.write(f"{edge[0]} {edge[1]} "
                   f"{colors[i][0]} {colors[i][1]} {colors[i][2]}\n")


def find_seed(mesh, face_udf, noise_threshold=1e-3):
    """
    Finds one seed triangle per B-rep face using Watershed segmentation.
    
    Parameters:
    - mesh: A trimesh.Trimesh object.
    - face_udf: np.ndarray of shape (n_faces, D), where column 0 is the UDF scalar.
    - noise_threshold: Float. UDF saddle height below which faces are considered 
                       separated by a true B-rep edge.
                       
    Returns:
    - final_seeds: A list of triangle indices representing the labeled seeds.
    """
    # 1. Extract the scalar UDF
    udf = face_udf[:, 0]
    n_faces = len(udf)

    # 2. Topological Gradient Ascent
    # We find the steepest uphill neighbor for every triangle
    adj = mesh.face_adjacency
    parent = np.arange(n_faces)
    max_neighbor_udf = udf.copy()

    for i in range(len(adj)):
        f1, f2 = adj[i]

        # If f2 is strictly higher than f1's current steepest path
        if (udf[f2], f2) > (max_neighbor_udf[f1], parent[f1]):
            max_neighbor_udf[f1] = udf[f2]
            parent[f1] = f2

        # If f1 is strictly higher than f2's current steepest path
        if (udf[f1], f1) > (max_neighbor_udf[f2], f2):
            max_neighbor_udf[f2] = udf[f1]
            parent[f2] = f1

    # 3. Resolve paths to find roots (Local Maxima)
    # Using path compression for speed
    def find_root(i):
        path = []
        curr = i
        while parent[curr] != curr:
            path.append(curr)
            curr = parent[curr]
        for node in path:
            parent[node] = curr  # Compress path
        return curr

    # Every face is now assigned to the index of its local maximum (its basin root)
    roots = np.array([find_root(i) for i in range(n_faces)])
    
    id_roots = np.unique(roots)
    # Export colored mesh
    color = np.zeros((n_faces, 3), dtype=np.uint8)
    color[id_roots, 0] = 255
    trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, face_colors=color).export("debug.ply")

    # 4. Find boundaries between basins and compute their Saddle Heights
    is_boundary = roots[adj[:, 0]] != roots[adj[:, 1]]
    boundary_edges = adj[is_boundary]

    saddle_heights = {}
    for f1, f2 in boundary_edges:
        r1, r2 = roots[f1], roots[f2]
        if r1 > r2:
            r1, r2 = r2, r1  # Standardize order for dictionary key

        # The saddle height at this specific edge is approximated by the lowest UDF of the pair
        edge_saddle = min(udf[f1], udf[f2])

        # Track the HIGHEST saddle point along the entire boundary between these two basins
        pair = (r1, r2)
        if pair not in saddle_heights or edge_saddle > saddle_heights[pair]:
            saddle_heights[pair] = edge_saddle

    # 5. Merge Over-segmented Basins (Union-Find)
    # If the saddle separating two basins is higher than the noise threshold,
    # it's a false edge. We merge them.
    unique_roots = np.unique(roots)
    basin_parent = {r: r for r in unique_roots}

    def find_basin(b):
        path = []
        curr = b
        while basin_parent[curr] != curr:
            path.append(curr)
            curr = basin_parent[curr]
        for node in path:
            basin_parent[node] = curr
        return curr

    for (r1, r2), saddle in saddle_heights.items():
        if saddle > noise_threshold:
            b1 = find_basin(r1)
            b2 = find_basin(r2)
            if b1 != b2:
                # Merge the basins. Keep the root with the highest UDF peak as the survivor.
                if udf[b1] > udf[b2]:
                    basin_parent[b2] = b1
                else:
                    basin_parent[b1] = b2

    # 6. Extract the final surviving seeds
    final_seeds = set()
    for r in basin_parent:
        final_seeds.add(find_basin(r))

    return list(final_seeds)


def coarse_segmentation(mesh, face_udf, threshold=0.01):
    """
    Segments the mesh by masking faces below a UDF threshold and performing
    region growing on the remaining faces.

    Parameters:
    - mesh: A trimesh.Trimesh object.
    - face_udf: np.ndarray of shape (n_faces,) or (n_faces, D); column 0 used if 2D.
    - threshold: Float. Faces with UDF below this value are masked (cluster_id = -1).

    Returns:
    - n_clusters: int, number of clusters found.
    - labels: np.ndarray of shape (n_faces,), cluster id per face (-1 if masked).
    """
    udf = face_udf[:, 0] if face_udf.ndim == 2 else face_udf
    n_faces = len(udf)

    active = udf >= threshold
    labels = np.full(n_faces, -1, dtype=np.int32)
    # labels[~active] = 0  # Masked faces
    # return 1, labels

    active_indices = np.where(active)[0]
    n_active = len(active_indices)
    if n_active == 0:
        return 0, labels

    # Keep only adjacency edges where both faces are active
    adj = mesh.face_adjacency  # (E, 2)
    both_active = active[adj[:, 0]] & active[adj[:, 1]]
    active_adj = adj[both_active]

    # Remap active face indices to [0, n_active) for the sparse graph
    remap = np.full(n_faces, -1, dtype=np.int32)
    remap[active_indices] = np.arange(n_active)

    if len(active_adj) > 0:
        rows = remap[active_adj[:, 0]]
        cols = remap[active_adj[:, 1]]
        data = np.ones(len(rows) * 2, dtype=np.int8)
        graph = csr_matrix((data, (np.r_[rows, cols], np.r_[cols, rows])),
                           shape=(n_active, n_active))
    else:
        graph = csr_matrix((n_active, n_active), dtype=np.int8)

    n_clusters, comp_labels = connected_components(graph, directed=False)
    labels[active_indices] = comp_labels

    # Check the label and reject small clusters (<5); relabel the cluster ids to be continuous
    cluster_sizes = np.bincount(comp_labels)
    valid_clusters = np.where(cluster_sizes >= 5)[0]
    valid_mask = np.isin(labels, valid_clusters)
    labels[~valid_mask] = -1  # Mask small clusters

    # Relabel to continuous ids: map each valid old id to a new consecutive id
    relabel_map = np.full(n_clusters, -1, dtype=np.int64)
    relabel_map[valid_clusters] = np.arange(len(valid_clusters))
    valid_positions = labels >= 0
    labels[valid_positions] = relabel_map[labels[valid_positions]]
    n_clusters = len(valid_clusters)

    return n_clusters, labels


def hierarchical_segmentation(mesh, face_udf, threshold1=0.003, threshold2=0.01,
                              filter_size=5, avg_udf_threshold=0.003):
    """
    Two-pass hierarchical segmentation mirroring C++ segment_mode3.

    Pass 1: Mask faces with |UDF| < threshold1, region-grow connected components,
            filter small clusters and those with low average UDF.
    Pass 2: For each surviving cluster whose max UDF >= threshold2, attempt to
            split it by re-growing only through faces with UDF >= threshold2.
            If a cluster splits into >= 2 valid sub-clusters, replace it;
            otherwise keep the original.

    Parameters:
    - mesh: A trimesh.Trimesh object.
    - face_udf: np.ndarray of shape (n_faces,) or (n_faces, D); column 0 used if 2D.
    - threshold1: Float. First-pass mask threshold.
    - threshold2: Float. Second-pass split threshold.
    - filter_size: Int. Minimum cluster size.
    - avg_udf_threshold: Float. Minimum average UDF for a cluster to survive.

    Returns:
    - n_clusters: int, number of clusters found.
    - labels: np.ndarray of shape (n_faces,), cluster id per face (-1 if masked).
    """
    udf = np.abs(face_udf[:, 0]) if face_udf.ndim == 2 else np.abs(face_udf)
    n_faces = len(udf)

    # --- Pass 1: threshold masking + connected components ---
    active = udf >= threshold1
    labels = np.full(n_faces, -1, dtype=np.int32)

    active_indices = np.where(active)[0]
    n_active = len(active_indices)
    if n_active == 0:
        return 0, labels

    adj = mesh.face_adjacency  # (E, 2)
    both_active = active[adj[:, 0]] & active[adj[:, 1]]
    active_adj = adj[both_active]

    remap = np.full(n_faces, -1, dtype=np.int32)
    remap[active_indices] = np.arange(n_active)

    if len(active_adj) > 0:
        rows = remap[active_adj[:, 0]]
        cols = remap[active_adj[:, 1]]
        data = np.ones(len(rows) * 2, dtype=np.int8)
        graph = csr_matrix((data, (np.r_[rows, cols], np.r_[cols, rows])),
                           shape=(n_active, n_active))
    else:
        graph = csr_matrix((n_active, n_active), dtype=np.int8)

    n_comp, comp_labels = connected_components(graph, directed=False)
    labels[active_indices] = comp_labels

    # Filter by size and average UDF
    cluster_sizes = np.bincount(comp_labels, minlength=n_comp)
    cluster_udf_sums = np.bincount(comp_labels, weights=udf[active_indices], minlength=n_comp)
    cluster_avg_udf = np.where(cluster_sizes > 0, cluster_udf_sums / cluster_sizes, 0.0)

    valid_clusters = np.where(
        (cluster_sizes >= filter_size) & (cluster_avg_udf > avg_udf_threshold)
    )[0]
    valid_mask = np.isin(labels, valid_clusters)
    labels[~valid_mask] = -1
    # Pass-1 labels retain original comp_labels ids (sparse); relabel once after pass 2.
    valid_set = set(valid_clusters.tolist())

    # --- Pass 2: split clusters with high max UDF using threshold2 ---
    # Compute per-cluster max UDF using original (sparse) cluster ids
    cluster_max_udf = np.full(n_comp, 0.0)
    np.maximum.at(cluster_max_udf, labels[labels >= 0], udf[labels >= 0])

    # Determine which valid clusters need splitting
    needs_split = np.zeros(n_comp, dtype=bool)
    for cid in valid_clusters:
        needs_split[cid] = cluster_max_udf[cid] >= threshold2

    # Faces eligible for pass-2: in a split-candidate cluster AND udf >= threshold2
    high_udf = udf >= threshold2
    labels_clipped = labels.clip(min=0)
    in_split_cluster = (labels >= 0) & needs_split[labels_clipped]
    eligible = in_split_cluster & high_udf

    eligible_indices = np.where(eligible)[0]
    n_eligible = len(eligible_indices)

    if n_eligible > 0:
        # Build sparse graph: both endpoints eligible AND in same pass-1 cluster
        both_eligible = eligible[adj[:, 0]] & eligible[adj[:, 1]]
        same_cluster = labels[adj[:, 0]] == labels[adj[:, 1]]
        p2_edges = adj[both_eligible & same_cluster]

        remap2 = np.full(n_faces, -1, dtype=np.int32)
        remap2[eligible_indices] = np.arange(n_eligible)

        if len(p2_edges) > 0:
            r2 = remap2[p2_edges[:, 0]]
            c2 = remap2[p2_edges[:, 1]]
            d2 = np.ones(len(r2) * 2, dtype=np.int8)
            g2 = csr_matrix((d2, (np.r_[r2, c2], np.r_[c2, r2])),
                            shape=(n_eligible, n_eligible))
        else:
            g2 = csr_matrix((n_eligible, n_eligible), dtype=np.int8)

        n_sub, sub_labels = connected_components(g2, directed=False)

        # Map each sub-component to its parent pass-1 cluster
        p1_of_eligible = labels[eligible_indices]
        sub_to_p1 = np.full(n_sub, -1, dtype=np.int32)
        sub_to_p1[sub_labels] = p1_of_eligible

        # Filter sub-clusters by size
        sub_sizes = np.bincount(sub_labels, minlength=n_sub)
        sub_valid = sub_sizes >= filter_size

        # Count valid sub-clusters per pass-1 cluster
        valid_per_p1 = defaultdict(int)
        for sid in range(n_sub):
            if sub_valid[sid]:
                valid_per_p1[sub_to_p1[sid]] += 1

        # For clusters that split into >= 2 valid sub-clusters,
        # replace pass-1 label with sub-cluster labels
        for cid in valid_clusters:
            if not needs_split[cid] or valid_per_p1.get(cid, 0) < 2:
                continue
            # Clear the original cluster labels; sub-clusters will overwrite
            labels[labels == cid] = -1
            cid_mask = p1_of_eligible == cid
            cid_sub_ids = np.unique(sub_labels[cid_mask])
            for sid in cid_sub_ids:
                if not sub_valid[sid]:
                    continue
                # Assign a new unique label (above any existing comp_label)
                new_id = n_comp
                n_comp += 1
                face_idx = eligible_indices[sub_labels == sid]
                labels[face_idx] = new_id

    # --- Final relabel to continuous ids ---
    unique_labels = np.unique(labels[labels >= 0])
    relabel_map = np.full(labels.max() + 1 if labels.max() >= 0 else 1, -1, dtype=np.int64)
    relabel_map[unique_labels] = np.arange(len(unique_labels))
    valid_pos = labels >= 0
    labels[valid_pos] = relabel_map[labels[valid_pos]]
    n_clusters = len(unique_labels)

    return n_clusters, labels


def udf_mesh_segmentation(mesh: trimesh.Trimesh, udf_mesh: trimesh.Trimesh, min_cluster_size=0):
    face_points = mesh.triangles_center  # (num_faces, 3)

    import igl  # only needed for the optional "udf_mesh" mode
    V = np.asarray(udf_mesh.vertices, dtype=np.float64)
    F = np.asarray(udf_mesh.faces,    dtype=np.int32)
    W = igl.winding_number(V, F, face_points.astype(np.float64))  # (num_faces,)
    is_inside = W > 0.5

    n_faces = len(mesh.faces)
    active = ~is_inside                          # faces outside the B-rep edge region
    labels = np.full(n_faces, -1, dtype=np.int32)

    active_indices = np.where(active)[0]
    n_active = len(active_indices)
    if n_active == 0:
        return 0, labels

    # Build vertex-adjacency: two faces are neighbours if they share ≥ 1 vertex.
    # Construct a sparse face×vertex incidence matrix A; A @ Aᵀ has a nonzero
    # at (i,j) iff faces i and j share at least one vertex — no Python loops.
    faces_arr = mesh.faces                              # (F, 3)
    n_verts   = int(mesh.vertices.shape[0])
    fi = np.repeat(np.arange(n_faces), 3)
    vi = faces_arr.ravel()
    A  = csr_matrix((np.ones(len(fi), dtype=np.int8), (fi, vi)),
                    shape=(n_faces, n_verts))
    shared = (A @ A.T).tocoo()                          # (F, F) sparse
    mask   = (shared.row < shared.col) & (shared.data > 0)
    adj    = np.stack([shared.row[mask], shared.col[mask]], axis=1).astype(np.int32)
    both_active = active[adj[:, 0]] & active[adj[:, 1]]
    active_adj  = adj[both_active]

    # Remap to contiguous [0, n_active) indices for the sparse graph
    remap = np.full(n_faces, -1, dtype=np.int32)
    remap[active_indices] = np.arange(n_active)

    if len(active_adj) > 0:
        rows = remap[active_adj[:, 0]]
        cols = remap[active_adj[:, 1]]
        data = np.ones(len(rows) * 2, dtype=np.int8)
        graph = csr_matrix((data, (np.r_[rows, cols], np.r_[cols, rows])),
                           shape=(n_active, n_active))
    else:
        graph = csr_matrix((n_active, n_active), dtype=np.int8)

    n_comp, comp_labels = connected_components(graph, directed=False)
    labels[active_indices] = comp_labels

    # Drop small clusters; relabel survivors to contiguous ids
    cluster_sizes = np.bincount(comp_labels, minlength=n_comp)
    valid_clusters = np.where(cluster_sizes >= min_cluster_size)[0]
    labels[~np.isin(labels, valid_clusters)] = -1

    relabel_map = np.full(n_comp, -1, dtype=np.int64)
    relabel_map[valid_clusters] = np.arange(len(valid_clusters))
    valid_pos = labels >= 0
    labels[valid_pos] = relabel_map[labels[valid_pos]]

    return len(valid_clusters), labels


def advanced_segmentation(mesh, face_udf, truncation_dist=0.014,
                          n_smooth=3, d_percentile=80, tau_merge=None,
                          min_cluster_size=5):
    """
    Advanced mesh segmentation using dual-edge B-score (direction reversal) with
    an adaptive distance threshold and hierarchical B-guided merging.

    Algorithm:
      1. Compute per-face tangent direction p and reliability r from face_udf.
      2. Smooth d and p for n_smooth rounds on the dual graph.
      3. Score each dual edge: B_ij = r_i * r_j * s_dir
         where s_dir = max(0, -(p_i·e_ij)(p_j·e_ij)) detects opposing directions
         across a potential BRep boundary. (No s_d term — direction reversals occur
         at moderate d inside the UDF tube, not only at d≈0.)
      4. Adaptive threshold: thresh = percentile(d[d < truncation_dist], d_percentile).
         Use vertex-adjacency connected components of faces with d >= thresh as
         initial (possibly over-segmented) regions.
      5. BFS to assign all remaining faces to the nearest region.
      6. Build a region-adjacency graph (RAG) with max B as edge weight.
         Kruskal-style hierarchical merge: merge adjacent-region pairs in ascending
         B order until B >= tau_merge (auto-derived from RAG B distribution).
      7. Re-assign unassigned faces; filter small clusters.

    Parameters:
    - mesh: trimesh.Trimesh object (marching-cube reconstruction, no sharp edges).
    - face_udf: (n_faces, 4) array. Column 0 = distance to nearest BRep edge
      (truncated at ~0.015). Columns 1–3 = gradient direction (may be unnormalised).
    - truncation_dist: faces with d >= this are considered deeply interior (unreliable
      direction for s_d; ignored for the adaptive threshold calculation).
    - n_smooth: rounds of bilateral direction smoothing on the dual graph.
    - d_percentile: percentile of non-truncated d values used as the active-face
      threshold. ~80 works well; raise if under-segmented, lower if over-segmented.
    - tau_merge: B threshold for hierarchical merging. Only merge adjacent-region
      pairs whose max-edge B < tau_merge. Default None: skip merging entirely.
      Merging is useful only when the initial count is significantly over-segmented.
    - min_cluster_size: clusters smaller than this are discarded (label → -1).

    Returns:
    - n_clusters: int, number of clusters found.
    - labels: (n_faces,) int32 array, -1 = unassigned.
    """
    n_faces = len(mesh.faces)
    d = face_udf[:, 0].copy()
    u_raw = face_udf[:, 1:4].copy()

    # Normalise direction vectors (face_udf directions may not be unit-length)
    u_norms = np.linalg.norm(u_raw, axis=1, keepdims=True)
    u = u_raw / (u_norms + 1e-8)

    normals = mesh.face_normals.copy()
    centroids = mesh.triangles_center.copy()
    adj = mesh.face_adjacency          # (E, 2) edge-adjacency
    f1, f2 = adj[:, 0], adj[:, 1]
    E = len(adj)

    # ── Step 1: per-face tangent direction and reliability ─────────────────────
    # p_i = normalise((I - n_i n_i^T) u_i)  [tangent-plane projection]
    # r_i = ||(I - n_i n_i^T) u_i||          [reliability: 0 if u is purely normal]
    dot_un = np.einsum('ij,ij->i', u, normals)
    u_t = u - dot_un[:, None] * normals
    r = np.linalg.norm(u_t, axis=1)
    p = u_t / (r[:, None] + 1e-8)

    # ── Step 2: smooth d and p on the dual graph ───────────────────────────────
    d_s = d.copy()
    p_s = p.copy()
    for _ in range(n_smooth):
        # Smooth d (simple average over 1-ring)
        d_nb = np.zeros(n_faces, dtype=np.float64)
        cnt = np.zeros(n_faces, dtype=np.float64)
        np.add.at(d_nb, f1, d_s[f2])
        np.add.at(d_nb, f2, d_s[f1])
        np.add.at(cnt, f1, 1)
        np.add.at(cnt, f2, 1)
        d_s = (d_nb + d_s) / (cnt + 1)

        # Smooth p with tangent-plane reprojection before accumulation
        pj_proj = p_s[f2] - np.einsum('ij,ij->i', p_s[f2], normals[f1])[:, None] * normals[f1]
        pi_proj = p_s[f1] - np.einsum('ij,ij->i', p_s[f1], normals[f2])[:, None] * normals[f2]
        p_nb = np.zeros((n_faces, 3), dtype=np.float64)
        np.add.at(p_nb, f1, pj_proj)
        np.add.at(p_nb, f2, pi_proj)
        p_sum = p_nb + p_s
        p_s = p_sum / (np.linalg.norm(p_sum, axis=1, keepdims=True) + 1e-8)

    # ── Step 3: B-score for each dual edge ─────────────────────────────────────
    # e_ij: unit vector from c_i to c_j in the average tangent plane
    e_vec = centroids[f2] - centroids[f1]
    n_avg = (normals[f1] + normals[f2]) / 2
    n_avg /= np.linalg.norm(n_avg, axis=1, keepdims=True) + 1e-8
    e_t = e_vec - np.einsum('ij,ij->i', e_vec, n_avg)[:, None] * n_avg
    e_t /= np.linalg.norm(e_t, axis=1, keepdims=True) + 1e-8

    pi_e = np.einsum('ij,ij->i', p_s[f1], e_t)
    pj_e = np.einsum('ij,ij->i', p_s[f2], e_t)
    # s_dir is high when p_i and p_j point toward the edge from opposite sides
    s_dir = np.maximum(0.0, -pi_e * pj_e)

    # B = r_i * r_j * s_dir  (no s_d: direction reversals live inside the UDF
    # tube at moderate d, not only near d=0)
    # Zero out edges where BOTH faces are deeply interior (truncated & unreliable)
    trunc_both = (d[f1] >= truncation_dist) & (d[f2] >= truncation_dist)
    B = r[f1] * r[f2] * s_dir
    B[trunc_both] = 0.0

    print(f"  B: nonzero={( B > 0).sum()}, "
          f"p90={np.percentile(B, 90):.4f}, p99={np.percentile(B, 99):.4f}, "
          f"max={B.max():.4f}")

    # ── Step 4: adaptive threshold + vertex-adjacency connected components ──────
    near_bdry_d = d[d < truncation_dist]
    if len(near_bdry_d) > 10:
        thresh = float(np.percentile(near_bdry_d, d_percentile))
    else:
        thresh = truncation_dist * 0.8
    print(f"  Adaptive threshold (p{d_percentile} of non-trunc d): {thresh:.5f}")

    active = d >= thresh
    active_idx = np.where(active)[0]
    n_active = len(active_idx)

    # Build vertex-adjacency for active faces (shares ≥1 vertex → safer connectivity)
    faces_arr = mesh.faces
    n_verts = int(mesh.vertices.shape[0])
    fi = np.repeat(np.arange(n_faces), 3)
    vi = faces_arr.ravel()
    A = csr_matrix((np.ones(len(fi), dtype=np.int8), (fi, vi)),
                   shape=(n_faces, n_verts))
    shared = (A @ A.T).tocoo()
    vtx_mask = (shared.row < shared.col) & (shared.data > 0)
    adj_vtx = np.stack([shared.row[vtx_mask], shared.col[vtx_mask]], axis=1).astype(np.int32)

    both_active = active[adj_vtx[:, 0]] & active[adj_vtx[:, 1]]
    active_adj = adj_vtx[both_active]

    remap = np.full(n_faces, -1, dtype=np.int32)
    remap[active_idx] = np.arange(n_active)

    if len(active_adj) > 0:
        rows = remap[active_adj[:, 0]]
        cols = remap[active_adj[:, 1]]
        data = np.ones(len(rows) * 2, dtype=np.int8)
        graph = csr_matrix((data, (np.r_[rows, cols], np.r_[cols, rows])),
                           shape=(n_active, n_active))
    else:
        graph = csr_matrix((n_active, n_active), dtype=np.int8)

    nc0, comp_labels = connected_components(graph, directed=False)
    labels = np.full(n_faces, -1, dtype=np.int32)
    labels[active_idx] = comp_labels
    print(f"  Initial clusters (d>={thresh:.4f}, vtx-adj): {nc0}")

    # ── Step 5: BFS to assign all remaining faces ──────────────────────────────
    nbrs = defaultdict(list)
    for fa, fb in adj:
        nbrs[fa].append(fb)
        nbrs[fb].append(fa)

    queue = deque()
    seen = set()
    for fa, fb in adj:
        if labels[fa] >= 0 and labels[fb] < 0:
            queue.append(fb)
        elif labels[fb] >= 0 and labels[fa] < 0:
            queue.append(fa)

    while queue:
        face = queue.popleft()
        if face in seen or labels[face] >= 0:
            seen.add(face)
            continue
        seen.add(face)
        nb_labels = [labels[nb] for nb in nbrs[face] if labels[nb] >= 0]
        if nb_labels:
            labels[face] = nb_labels[0]
            for nb in nbrs[face]:
                if labels[nb] < 0:
                    queue.append(nb)

    # ── Step 6: B-guided hierarchical merge ────────────────────────────────────
    # Build RAG: for each adjacent cluster pair, record max B of shared edges
    rag = defaultdict(list)
    for ei in range(E):
        la, lb = labels[f1[ei]], labels[f2[ei]]
        if la >= 0 and lb >= 0 and la != lb:
            k = (min(la, lb), max(la, lb))
            rag[k].append(B[ei])

    if rag and tau_merge is None:
        b_vals = np.array([max(v) for v in rag.values()])
        print(f"  RAG pairs: {len(rag)}, B range [{b_vals.min():.4f},{b_vals.max():.4f}]"
              f" (tau_merge=None → no merging)")

    if rag and tau_merge is not None:
        # Sort pairs by max B (ascending = weakest boundaries first)
        pairs = sorted((max(v), k[0], k[1]) for k, v in rag.items())
        b_vals = np.array([p[0] for p in pairs])

        print(f"  RAG pairs: {len(pairs)}, B range [{b_vals[0]:.4f},{b_vals[-1]:.4f}], "
              f"tau_merge={tau_merge:.4f}")

        # Union-Find merge
        parent = list(range(nc0))

        def _find(x):
            root = x
            while parent[root] != root:
                root = parent[root]
            while parent[x] != root:
                parent[x], x = root, parent[x]
            return root

        nc = nc0
        merge_map = {}  # old_label → merged root
        for b_val, c1, c2 in pairs:
            if b_val >= tau_merge:
                break
            p1, p2 = _find(c1), _find(c2)
            if p1 != p2:
                parent[p1] = p2
                nc -= 1

        # Apply merged labels
        label_remap = {i: _find(i) for i in range(nc0)}
        valid_pos = labels >= 0
        labels[valid_pos] = np.array([label_remap[l] for l in labels[valid_pos]])
        print(f"  After B-merge: {nc} clusters")

    # ── Step 7: relabel and filter small clusters ──────────────────────────────
    unique_labels = np.unique(labels[labels >= 0])
    sizes = np.bincount(labels[labels >= 0].astype(np.int64),
                        minlength=int(unique_labels.max()) + 1 if len(unique_labels) > 0 else 1)

    # Filter small clusters
    valid = np.array([l for l in unique_labels if sizes[l] >= min_cluster_size])
    if len(valid) == 0:
        return 0, labels.astype(np.int32)

    invalid_mask = ~np.isin(labels, valid)
    labels[invalid_mask] = -1

    # BFS re-assign faces that lost their cluster due to small-cluster filtering
    queue2 = deque()
    seen2 = set()
    for fa, fb in adj:
        if labels[fa] >= 0 and labels[fb] < 0:
            queue2.append(fb)
        elif labels[fb] >= 0 and labels[fa] < 0:
            queue2.append(fa)
    while queue2:
        face = queue2.popleft()
        if face in seen2 or labels[face] >= 0:
            seen2.add(face)
            continue
        seen2.add(face)
        nb_labels = [labels[nb] for nb in nbrs[face] if labels[nb] >= 0]
        if nb_labels:
            labels[face] = nb_labels[0]
            for nb in nbrs[face]:
                if labels[nb] < 0:
                    queue2.append(nb)

    # Relabel to contiguous ids
    unique_labels = np.unique(labels[labels >= 0])
    relabel = np.full(int(labels.max()) + 1 if labels.max() >= 0 else 1, -1, dtype=np.int64)
    relabel[unique_labels] = np.arange(len(unique_labels))
    pos = labels >= 0
    labels[pos] = relabel[labels[pos]]
    n_clusters = len(unique_labels)

    print(f"  Final clusters: {n_clusters}")
    return n_clusters, labels.astype(np.int32)


def merge_small_plane_clusters(mesh, labels,
                               small_threshold=20,
                               plane_normal_dot=0.99,
                               plane_dist_threshold=0.005,
                               plane_inlier_ratio=0.8,
                               final_min_size=final_size_threshold):
    """
    Post-process hierarchical segmentation labels by merging tiny plane clusters.

    Steps:
      1. Identify small clusters (size < small_threshold).
      2. Fit a plane to each small cluster; keep only those whose faces are
         planar (inlier_ratio fraction within plane_dist_threshold / plane_normal_dot).
      3. For each small planar cluster, merge it into the nearest compatible
         plane cluster (small or large) that shares the same plane.
      4. Fit planes for all surviving clusters; assign unlabeled (-1) faces
         to the best-fitting plane cluster (strict threshold).
      5. Remove clusters with size < final_min_size and relabel continuously.

    Parameters:
    - mesh: trimesh.Trimesh
    - labels: (n_faces,) int32 array from hierarchical_segmentation (-1 = unassigned)
    - small_threshold: clusters below this size are candidates for merging
    - plane_normal_dot: minimum |dot(n1, n2)| to consider two planes co-planar (strict)
    - plane_dist_threshold: max distance from plane centroid to consider co-planar (strict)
    - plane_inlier_ratio: fraction of faces that must lie on the fitted plane
    - final_min_size: clusters smaller than this are discarded at the end

    Returns:
    - n_clusters: int
    - labels: (n_faces,) int32 array
    """
    labels = labels.copy()
    face_centroids = mesh.triangles_center   # (n_faces, 3)
    face_normals   = mesh.face_normals       # (n_faces, 3)
    face_areas     = mesh.area_faces         # (n_faces,)

    def fit_plane(mask):
        """Return (normal, d) or None if too few faces or not planar enough."""
        idx = np.where(mask)[0]
        if len(idx) < 3:
            return None
        c  = face_centroids[idx]
        n  = face_normals[idx]
        a  = face_areas[idx]
        total_a = a.sum()
        if total_a < 1e-12:
            return None
        w_n   = (n * a[:, None]).sum(axis=0)
        w_len = np.linalg.norm(w_n)
        if w_len < 1e-8:
            return None
        plane_n = w_n / w_len
        plane_d = np.dot((c * a[:, None]).sum(axis=0) / total_a, plane_n)
        return plane_n, plane_d

    def is_planar(mask, plane_n, plane_d):
        idx = np.where(mask)[0]
        c = face_centroids[idx]
        n = face_normals[idx]
        dists  = np.abs(c @ plane_n - plane_d)
        aligns = np.abs(n @ plane_n)
        inlier = (dists < plane_dist_threshold) & (aligns > plane_normal_dot)
        return inlier.sum() / len(idx) >= plane_inlier_ratio

    def planes_match(pn1, pd1, pn2, pd2):
        dot = np.dot(pn1, pn2)
        if abs(dot) < plane_normal_dot:
            return False
        pd2_adj = pd2 if dot >= 0 else -pd2
        return abs(pd1 - pd2_adj) < plane_dist_threshold

    # ── Step 1 & 2: fit planes for all clusters, mark small ones ──────────────
    unique_labels = np.unique(labels[labels >= 0])
    cluster_planes = {}   # cid -> (normal, d)
    for cid in unique_labels:
        mask   = labels == cid
        result = fit_plane(mask)
        if result is None:
            continue
        pn, pd = result
        if is_planar(mask, pn, pd):
            cluster_planes[cid] = (pn, pd)

    small_planar = [cid for cid in cluster_planes
                    if (labels == cid).sum() < small_threshold]

    # ── Step 3: merge each small planar cluster into the best matching cluster ─
    # Build cluster sizes for tie-breaking (prefer large targets).
    cid_sizes = {cid: int((labels == cid).sum()) for cid in unique_labels}

    # Union-Find over all labels so merges chain correctly.
    uf_parent = {int(c): int(c) for c in unique_labels}

    def uf_find(x):
        while uf_parent[x] != x:
            uf_parent[x] = uf_parent[uf_parent[x]]
            x = uf_parent[x]
        return x

    def uf_union(x, y):
        px, py = uf_find(x), uf_find(y)
        if px == py:
            return
        # Smaller cluster points to larger
        if cid_sizes.get(px, 0) < cid_sizes.get(py, 0):
            uf_parent[px] = py
        else:
            uf_parent[py] = px

    small_planar_set = set(small_planar)
    for small_cid in small_planar:
        pn_s, pd_s = cluster_planes[small_cid]
        best_target = None
        best_size   = -1
        for cid in small_planar_set:
            if cid == small_cid:
                continue
            pn, pd = cluster_planes[cid]
            if not planes_match(pn_s, pd_s, pn, pd):
                continue
            sz = cid_sizes.get(cid, 0)
            if sz > best_size:
                best_size   = sz
                best_target = cid
        if best_target is not None:
            uf_union(small_cid, best_target)

    # Apply union-find remapping to labels
    root_map = {int(c): uf_find(int(c)) for c in unique_labels}
    pos = labels >= 0
    labels[pos] = np.array([root_map[l] for l in labels[pos]], dtype=np.int32)

    # Track which post-merge cluster IDs came from small clusters only.
    merged_small_roots = {root_map[int(c)] for c in small_planar_set}

    # ── Step 4: assign unlabeled (-1) faces to merged small plane clusters ────
    # Refit planes only on clusters that originated from small clusters.
    cluster_planes2 = {}
    for cid in merged_small_roots:
        mask = labels == cid
        if not mask.any():
            continue
        result = fit_plane(mask)
        if result is None:
            continue
        pn, pd = result
        if is_planar(mask, pn, pd):
            cluster_planes2[cid] = (pn, pd)

    unlabeled = np.where(labels == -1)[0]
    if len(unlabeled) > 0 and len(cluster_planes2) > 0:
        ul_c = face_centroids[unlabeled]   # (m, 3)
        ul_n = face_normals[unlabeled]     # (m, 3)
        best_dist = np.full(len(unlabeled), np.inf)
        best_cid  = np.full(len(unlabeled), -1, dtype=np.int32)
        for cid, (pn, pd) in cluster_planes2.items():
            dists  = np.abs(ul_c @ pn - pd)
            aligns = np.abs(ul_n @ pn)
            fits   = (dists < plane_dist_threshold) & (aligns > plane_normal_dot)
            better = fits & (dists < best_dist)
            best_dist[better] = dists[better]
            best_cid[better]  = cid
        assign = best_cid >= 0
        labels[unlabeled[assign]] = best_cid[assign]

    # ── Step 5: remove clusters < final_min_size and relabel ──────────────────
    unique3 = np.unique(labels[labels >= 0])
    if len(unique3) > 0:
        sizes3 = np.bincount(labels[labels >= 0].astype(np.int64),
                             minlength=int(unique3.max()) + 1)
        valid3 = unique3[sizes3[unique3] >= final_min_size]
    else:
        valid3 = np.array([], dtype=np.int64)

    labels[~np.isin(labels, valid3)] = -1

    unique_final = np.unique(labels[labels >= 0])
    if len(unique_final) > 0:
        relabel = np.full(int(labels.max()) + 1, -1, dtype=np.int64)
        relabel[unique_final] = np.arange(len(unique_final))
        pos = labels >= 0
        labels[pos] = relabel[labels[pos]]
        n_clusters = len(unique_final)
    else:
        n_clusters = 0

    return n_clusters, labels.astype(np.int32)


def colorize_mesh(mesh, labels, num_clusters):
    cluster_colors = np.random.randint(0, 255, size=(num_clusters, 3), dtype=np.uint8)
    face_colors = cluster_colors[labels]
    face_colors[labels == -1] = [0, 0, 0]  # Masked faces in black
    mesh_colored = trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.faces,
        face_colors=face_colors
    )
    return mesh_colored


def process_item(folder, v_from_scratch=True):
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            if isinstance(folder, S3Path):
                download_file(folder/"udf_g.npy", temp_dir)
                download_file(folder/"recon_sdf.ply", temp_dir)
            else:
                shutil.copy(folder/"udf_g.npy", temp_dir/"udf_g.npy")
                shutil.copy(folder/"recon_sdf.ply", temp_dir/"recon_sdf.ply")
                # shutil.copy(folder/"recon_udf.ply", temp_dir/"recon_udf.ply")

            # Load per-axis normalization parameters if available
            norm_params = None
            norm_params_src = folder/"norm_params.npz"
            if isinstance(folder, S3Path):
                try:
                    download_file(norm_params_src, temp_dir)
                    norm_params = np.load(temp_dir/"norm_params.npz")
                except Exception:
                    pass
            elif norm_params_src.exists():
                norm_params = np.load(norm_params_src)

            mesh = trimesh.load(temp_dir / "recon_sdf.ply")
            # Keep only the largest connected component; filter face_udf to match.
            _adj = mesh.face_adjacency  # (M, 2) pairs of adjacent faces
            _, _comp = connected_components(
                csr_matrix((np.ones(len(_adj)), (_adj[:, 0], _adj[:, 1])),
                           shape=(len(mesh.faces),) * 2),
                directed=False)
            _face_mask = _comp == np.bincount(_comp).argmax()
            mesh = mesh.submesh([np.where(_face_mask)[0]], append=True)
            face_udf = np.load(temp_dir / "udf_g.npy")[_face_mask]

            if mode == "coarse":
                num_clusters, labels = coarse_segmentation(mesh, face_udf, threshold=threshold)
            elif mode == "hierarchical":
                num_clusters, labels = hierarchical_segmentation(mesh, face_udf, threshold1=threshold1, threshold2=threshold2)
                if is_merge_small:
                    num_clusters, labels = merge_small_plane_clusters(mesh, labels)
            elif mode == "udf_mesh":
                num_clusters, labels = udf_mesh_segmentation(mesh, udf_mesh)
            elif mode == "advanced":
                num_clusters, labels = advanced_segmentation(mesh, face_udf)
            else:
                raise ValueError(f"Unknown mode: {mode}")
            if num_clusters == 0:
                print(f"  Warning: 0 clusters produced — assigning whole mesh as cluster 0")
                labels = np.zeros(len(mesh.faces), dtype=np.int32)
                num_clusters = 1
            out_mesh = colorize_mesh(mesh, labels, num_clusters)
            out_mesh.face_attributes["label"] = labels

            if norm_params is not None:
                # Save the normalized (inference-space) cluster mesh
                out_mesh.export(temp_dir/"cluster_normalized.ply")

                # Denormalize: v_orig = v_norm * scale + center
                # norm_scale is the effective per-axis divisor stored by the dataset.
                center = norm_params["center"]  # (3,)
                scale = norm_params["scale"]    # (3,) effective per-axis divisors
                denorm_mesh = out_mesh.copy()
                denorm_mesh.face_attributes = copy.deepcopy(out_mesh.face_attributes)
                denorm_mesh.vertices = denorm_mesh.vertices * scale + center
                denorm_mesh.export(temp_dir/"cluster.ply")
            else:
                out_mesh.export(temp_dir/"cluster.ply")

            if (temp_dir/"cluster.ply").exists():
                if isinstance(folder, S3Path):
                    (folder/"cluster.ply").upload_from(temp_dir/"cluster.ply")
                    if (temp_dir/"cluster_normalized.ply").exists():
                        (folder/"cluster_normalized.ply").upload_from(temp_dir/"cluster_normalized.ply")
                else:
                    shutil.copy(temp_dir/"cluster.ply", folder/"cluster.ply")
                    if (temp_dir/"cluster_normalized.ply").exists():
                        shutil.copy(temp_dir/"cluster_normalized.ply", folder/"cluster_normalized.ply")
                return True
            else:
                return False
    except Exception as e:
        print(f"Error processing {folder}: {e}")
        return False


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Segment reconstructed meshes into BRep faces. Each immediate "
                    "subdirectory of ROOT must contain recon_sdf.ply and udf_g.npy.")
    ap.add_argument("root", help="directory containing per-shape result folders")
    args = ap.parse_args()

    data_root = Path(args.root)
    folders = sorted(item for item in data_root.glob("*") if item.is_dir())
    print(f"Total folders: {len(folders)}")
    task_states = [process_item(folder) for folder in tqdm(folders)]
    task_states = np.array(task_states)
    print(f"Total: {len(task_states)}   Succeed: {(task_states == 1).sum()}   "
          f"Failed: {(task_states == 0).sum()}")