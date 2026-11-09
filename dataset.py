import os
import json
import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch.utils.data import Dataset


print("dataset")
# ============================================================
# Motif category mapping (raw motif id -> 6 collapsed classes)
# ============================================================
# MOTIF_MAP = {
#     0: 0,   # bulge
#     1: 1,   # exterior loop
#     2: 2,   # hairpin
#     3: 3,   # stem (open)
#     7: 3,   # stem (close)
#     5: 4,   # junction
#     9: 5    # internal loop
# }

# ============================================================
# Atomic / chemical lookup table per nucleotide base:
# [purine flag, H-bond donors, H-bond acceptors, ring count, MW]
# ============================================================
# ATOMIC_TABLE = {
#     "A": [1, 1, 1, 2, 135.13],
#     "G": [1, 1, 2, 2, 151.13],
#     "C": [0, 0, 1, 1, 111.10],
#     "U": [0, 0, 2, 1, 112.09]
# }
# MW_MAX = 151.13

# ============================================================
# Spatial neighborhood cutoff (Angstroms), shared by the
# local-geometry features and the spatial graph edges.
# ============================================================
SPATIAL_CUTOFF = 8.0

# ============================================================
# Handcrafted feature group dimensions (removes magic numbers
# from the assembly code below).
# ============================================================
# BASE_DIM = 5
# MOTIF_DIM = 6
# SS_DIM = 3
SASA_DIM = 2
GEOMETRY_DIM = 5   # dist_centroid, neighbor_count, pos_norm, mean_nbr_dist, min_nbr_dist
# ATOMIC_DIM = 5
# WINDOW_DIM = 5
# TOPOLOGY_DIM = 2
GRAPH_DIM = 2      # degree, clustering coefficient

# Total handcrafted dimension. RNABERT_DIM is not fixed here because
# it depends on the embedding file provided at runtime; it is simply
# appended after the handcrafted block (see feature order below).
# HANDCRAFTED_DIM = (
#      MOTIF_DIM + SS_DIM +  GEOMETRY_DIM
#     + ATOMIC_DIM + WINDOW_DIM  + GRAPH_DIM
# )  # = 26

HANDCRAFTED_DIM = (
      SASA_DIM +  GEOMETRY_DIM
      + GRAPH_DIM
)  # = 9

# ------------------------------------------------------------
# NOTE on a deliberate deviation from the literal spec:
#
# When computing mean/min spatial-neighbor distance, we exclude the
# immediate sequence neighbors (i-1, i+1) from the 8A neighborhood.
# Those are backbone-connected and are almost always within a few A
# of every residue, regardless of tertiary structure -- including
# them makes "min neighbor distance" mostly measure "am I an interior
# residue" rather than anything about 3D packing / pocket geometry,
# which is what actually matters for ligand-binding prediction.
# `neighbor_count` itself is left exactly as specified (all spatial
# neighbors, including backbone-adjacent ones).
# Set this to False to reproduce the literal, un-filtered behavior.
# ------------------------------------------------------------
EXCLUDE_BACKBONE_NEIGHBORS_IN_GEOM = True


class RNALigandJSONDataset(Dataset):
    """
    Dataset V2 for RNA-small-molecule binding-site prediction.

    Reads per-structure JSON files (unchanged format) describing an RNA
    graph -- nodes with base identity, motif, secondary structure, SASA,
    3D coordinates and a per-node binding label -- and builds a PyG
    `Data` object with:

        x          : [num_nodes, HANDCRAFTED_DIM (+ RNABERT_DIM)]
        edge_index : backbone + base-pair + spatial edges (unchanged)
        edge_type  : 0 = backbone, 1 = base-pair, 2 = spatial
        edge_attr  : one-hot edge type (+ normalized distance for spatial)
        y          : per-node label

    Feature order (never mixed): Base -> Motif -> Secondary Structure ->
    SASA -> Local Geometry -> Chemical -> Sequential Context ->
    Topology -> Graph -> RNABERT.
    """

    def __init__(
        self,
        json_folder,
        rnabert_dir=None,
        ernie_dir=None,
        rnafm_dir=None,

        # use_base=True,
        # use_motif=True,
        # use_ss=True,
        use_sasa=True,
        use_geometry=True,
        # use_atomic=True,
        # use_window=True,
        # use_topology=True,
        use_graph=True,
        debug=False
    ):
        self.files = sorted([
            os.path.join(json_folder, f)
            for f in os.listdir(json_folder)
            if f.endswith(".json")
        ])

        self.rnabert_dir = rnabert_dir
        self.ernie_dir = ernie_dir
        self.rnafm_dir = rnafm_dir

        # self.use_base = use_base
        # self.use_motif = use_motif
        # self.use_ss = use_ss
        self.use_sasa = use_sasa
        self.use_geometry = use_geometry
        # self.use_atomic = use_atomic
        # self.use_window = use_window
        # self.use_topology = use_topology
        self.use_graph = use_graph

        self.debug = debug

    def __len__(self):
        return len(self.files)

    # ------------------------------------------------------------
    # Topology helper: stem-loop segment length + distance to the
    # nearest segment boundary, per node. Unchanged logic from v1,
    # just pulled out into its own function for readability.
    # ------------------------------------------------------------
    # @staticmethod
    # def _compute_topology_arrays(nodes):
    #     num_nodes = len(nodes)
    #     motif_classes = [
    #         MOTIF_MAP.get(int(n.get("motif", -1)), -1)
    #         for n in nodes
    #     ]

    #     segment_length = [1] * num_nodes
    #     dist_boundary = [0] * num_nodes

    #     seg_start = 0
    #     while seg_start < num_nodes:
    #         seg_end = seg_start
    #         while (
    #             seg_end + 1 < num_nodes
    #             and motif_classes[seg_end + 1] == motif_classes[seg_start]
    #         ):
    #             seg_end += 1

    #         seg_len = seg_end - seg_start + 1
    #         for k in range(seg_start, seg_end + 1):
    #             pos_in_seg = k - seg_start
    #             segment_length[k] = seg_len
    #             dist_boundary[k] = min(pos_in_seg, seg_len - 1 - pos_in_seg)

    #         seg_start = seg_end + 1

    #     return segment_length, dist_boundary

    # ------------------------------------------------------------
    # Geometry helper: compute the full pairwise-distance matrix
    # ONCE per structure, and derive neighbor_count / mean_neighbor_
    # distance / min_neighbor_distance from it for every node.
    # This replaces the old per-node recomputation of distances.
    # ------------------------------------------------------------
    @staticmethod
    def _compute_geometry_arrays(coords):
        num_nodes = coords.shape[0]

        diff = coords[:, None, :] - coords[None, :, :]
        dist_matrix = np.linalg.norm(diff, axis=-1)

        neighbor_mask = dist_matrix < SPATIAL_CUTOFF
        np.fill_diagonal(neighbor_mask, False)

        neighbor_counts = neighbor_mask.sum(axis=1)

        # Mask used only for mean/min distance stats: optionally drop
        # the trivial backbone-adjacent contacts (see module-level note).
        stat_mask = neighbor_mask.copy()
        if EXCLUDE_BACKBONE_NEIGHBORS_IN_GEOM:
            for i in range(num_nodes):
                if i - 1 >= 0:
                    stat_mask[i, i - 1] = False
                if i + 1 < num_nodes:
                    stat_mask[i, i + 1] = False

        mean_neighbor_distance = np.zeros(num_nodes)
        min_neighbor_distance = np.zeros(num_nodes)

        for i in range(num_nodes):
            neighbor_dists = dist_matrix[i, stat_mask[i]]
            if neighbor_dists.size == 0:
                # No non-backbone contacts: fall back to the full
                # (unfiltered) spatial neighborhood.
                neighbor_dists = dist_matrix[i, neighbor_mask[i]]
            if neighbor_dists.size == 0:
                # Truly isolated node -> no spatial neighbors at all.
                mean_neighbor_distance[i] = SPATIAL_CUTOFF
                min_neighbor_distance[i] = SPATIAL_CUTOFF
            else:
                mean_neighbor_distance[i] = neighbor_dists.mean()
                min_neighbor_distance[i] = neighbor_dists.min()

        return dist_matrix, neighbor_counts, mean_neighbor_distance, min_neighbor_distance

    # ------------------------------------------------------------
    # Edge construction: backbone + base-pair + spatial edges.
    # Logic is unchanged from v1; only reuses the precomputed
    # distance matrix instead of recomputing distances.
    # ------------------------------------------------------------
    @staticmethod
    def _build_edges(nodes, dist_matrix):
        num_nodes = len(nodes)
        edge_index = []
        edge_type = []
        edge_attr = []

        # Backbone edges (sequence-adjacent, bidirectional)
        for i in range(num_nodes - 1):
            edge_index.append([i, i + 1])
            edge_index.append([i + 1, i])
            edge_type.append(0)
            edge_type.append(0)
            edge_attr.append([1, 0, 0, 0])
            edge_attr.append([1, 0, 0, 0])

        # Base-pair edges (from dot-bracket secondary structure)
        stack = []
        for i, node in enumerate(nodes):
            if node["dotbracket"] == "(":
                stack.append(i)
            elif node["dotbracket"] == ")" and stack:
                j = stack.pop()
                edge_index.append([i, j])
                edge_index.append([j, i])
                edge_type.append(1)
                edge_type.append(1)
                edge_attr.append([0, 1, 0, 0])
                edge_attr.append([0, 1, 0, 0])

        # Spatial edges (3D proximity within SPATIAL_CUTOFF)
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                dist = dist_matrix[i, j]
                if dist < SPATIAL_CUTOFF:
                    dist_norm = dist / SPATIAL_CUTOFF
                    edge_index.append([i, j])
                    edge_index.append([j, i])
                    edge_type.append(2)
                    edge_type.append(2)
                    edge_attr.append([0, 0, 1, dist_norm])
                    edge_attr.append([0, 0, 1, dist_norm])

        return edge_index, edge_type, edge_attr

    # ------------------------------------------------------------
    # Graph-level structural features: node degree and clustering
    # coefficient, computed on the FULL graph (backbone + base-pair
    # + spatial edges combined, treated as one undirected simple
    # graph). Intentionally limited to just these two metrics.
    # ------------------------------------------------------------
    @staticmethod
    def _compute_graph_features(num_nodes, edge_index):
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        for (u, v) in edge_index:
            if u != v:
                G.add_edge(u, v)

        degree = dict(G.degree())
        clustering = nx.clustering(G)

        degree_arr = np.array([degree[i] for i in range(num_nodes)], dtype=np.float32)
        clustering_arr = np.array([clustering[i] for i in range(num_nodes)], dtype=np.float32)

        return degree_arr, clustering_arr

    def __getitem__(self, idx):

        with open(self.files[idx]) as f:
            data = json.load(f)

        nodes = data["rna"]["nodes"]
        num_nodes = len(nodes)

        # --------------------------------------------------------
        # RNABERT embedding loading (unchanged)
        # --------------------------------------------------------
        rnabert_emb = None
        ernie_emb = None
        rnafm_emb = None
        if self.rnabert_dir is not None:
            fname = os.path.basename(self.files[idx]).replace(".json", ".npy")
            rnabert_emb = np.load(os.path.join(self.rnabert_dir, fname))
            assert rnabert_emb.shape[0] == num_nodes, \
                f"{fname}: RNABERT len {rnabert_emb.shape[0]} vs nodes {num_nodes}"
            
        if self.ernie_dir is not None:
            fname = os.path.basename(self.files[idx]).replace(".json", ".npy")
            ernie_emb = np.load(os.path.join(self.ernie_dir, fname))
            assert ernie_emb.shape[0] == num_nodes, \
                f"{fname}: ERNIE_BERT len {ernie_emb.shape[0]} vs nodes {num_nodes}"
            
        if self.rnafm_dir is not None:
            fname = os.path.basename(self.files[idx]).replace(".json", ".npy")
            rnafm_emb = np.load(os.path.join(self.rnafm_dir, fname))
            assert rnafm_emb.shape[0] == num_nodes, \
                f"{fname}: RNAFM len {rnafm_emb.shape[0]} vs nodes {num_nodes}"

        # --------------------------------------------------------
        # Structure-level globals: coordinates / centroid / SASA stats
        # --------------------------------------------------------
        coords = np.array([node["coord"] for node in nodes])
        centroid = coords.mean(axis=0)

        sasa_vals = np.array([node.get("sasa", 0.0) for node in nodes])
        sasa_mean = sasa_vals.mean()
        sasa_std = sasa_vals.std() if sasa_vals.std() > 0 else 1.0
        # Epsilon guard: avoids a divide-by-zero on structures where
        # every SASA value happens to be 0.
        sasa_max = sasa_vals.max() if sasa_vals.max() > 0 else 1.0

        # --------------------------------------------------------
        # Stem-loop topology arrays (segment length / boundary dist)
        # --------------------------------------------------------
        # segment_length, dist_boundary = self._compute_topology_arrays(nodes)

        # --------------------------------------------------------
        # Vectorized geometry arrays: one pairwise-distance pass,
        # reused for both node features and spatial edges below.
        # --------------------------------------------------------
        (dist_matrix, neighbor_counts,
         mean_neighbor_distance, min_neighbor_distance) = self._compute_geometry_arrays(coords)

        # --------------------------------------------------------
        # Graph construction: backbone + base-pair + spatial edges
        # --------------------------------------------------------
        edge_index, edge_type, edge_attr = self._build_edges(nodes, dist_matrix)

        degree_arr, clustering_arr = None, None
        if self.use_graph:
            degree_arr, clustering_arr = self._compute_graph_features(num_nodes, edge_index)

        if self.debug:
            print(f"[DEBUG] {self.files[idx]}: {num_nodes} nodes, "
                  f"{len(edge_index)} directed edge entries")

        # --------------------------------------------------------
        # Per-node feature assembly
        # --------------------------------------------------------
        x = []
        y = []

        for i, node in enumerate(nodes):
            base = node["base"]

            # ----------------------------------------------------
            # Base Identity Features
            # ----------------------------------------------------
            # base_onehot = [
            #     int(base == "A"),
            #     int(base == "U"),
            #     int(base == "G"),
            #     int(base == "C"),
            #     int(base not in ["A", "U", "G", "C"])
            # ]

            # ----------------------------------------------------
            # Motif Features
            # ----------------------------------------------------
            # motif_raw = int(node.get("motif", -1))
            # motif_class = MOTIF_MAP.get(motif_raw, None)
            # motif_onehot = [0] * MOTIF_DIM
            # if motif_class is not None:
            #     motif_onehot[motif_class] = 1

            # ----------------------------------------------------
            # Secondary Structure Features
            # ----------------------------------------------------
            # ss = node.get("dotbracket", ".")
            # ss_onehot = [
            #     int(ss == "."),
            #     int(ss == "("),
            #     int(ss == ")")
            # ]

            # ----------------------------------------------------
            # Surface Accessibility Features
            #   sasa_norm : SASA / max(SASA) for this structure
            #   sasa_rel  : (SASA - mean) / std for this structure
            # No tanh squashing; both are epsilon-guarded above.
            # ----------------------------------------------------
            sasa = float(node.get("sasa", 0.0))
            sasa_norm = sasa / sasa_max
            sasa_rel = (sasa - sasa_mean) / sasa_std

            # ----------------------------------------------------
            # Local Geometry Features
            #   distance_from_centroid, neighbor_count, position_norm,
            #   mean_neighbor_distance, minimum_neighbor_distance
            # neighbor_count/mean/min come from the precomputed arrays
            # (no per-node distance recomputation). Left un-normalized
            # as specified.
            # ----------------------------------------------------
            coord = np.array(node["coord"])
            dist_centroid = np.linalg.norm(coord - centroid)
            neighbor_count = float(neighbor_counts[i])
            pos_norm = i / max(1, num_nodes - 1)
            mean_nbr_dist = float(mean_neighbor_distance[i])
            min_nbr_dist = float(min_neighbor_distance[i])

            # ----------------------------------------------------
            # Chemical Features (unchanged; no normalization except
            # molecular weight, which was already normalized in v1)
            # ----------------------------------------------------
            # purine, donors, acceptors, rings, mw = ATOMIC_TABLE.get(
            #     base, [0, 0, 0, 0, 0.0]
            # )
            # mw_norm = mw / MW_MAX

            # ----------------------------------------------------
            # Sequential Context Features (base fractions in a
            # +/-3 window around position i)
            # ----------------------------------------------------
            # start = max(0, i - 3)
            # end = min(num_nodes, i + 4)
            # window_bases = [nodes[j]["base"] for j in range(start, end)]
            # wlen = len(window_bases)

            # win_frac_A = window_bases.count("A") / wlen
            # win_frac_U = window_bases.count("U") / wlen
            # win_frac_G = window_bases.count("G") / wlen
            # win_frac_C = window_bases.count("C") / wlen
            # win_gc_frac = (window_bases.count("G") + window_bases.count("C")) / wlen

            # ----------------------------------------------------
            # Topology Features (unchanged, un-normalized)
            # ----------------------------------------------------
            # segment_length_log = float(np.log1p(segment_length[i]))
            # dist_boundary_clipped = float(min(dist_boundary[i], 5))

            # ----------------------------------------------------
            # Graph Features: degree + clustering coefficient from
            # the full backbone + base-pair + spatial graph
            # ----------------------------------------------------
            if self.use_graph:
                node_degree = float(degree_arr[i])
                node_clustering = float(clustering_arr[i])
            else:
                node_degree = 0.0
                node_clustering = 0.0

            # ----------------------------------------------------
            # Final Feature Assembly
            # Every group is computed above; here we just gate each
            # group on its use_* flag and concatenate in the fixed
            # order: base -> motif -> ss -> sasa -> geometry ->
            # chemical -> window -> topology -> graph -> (rnabert)
            # ----------------------------------------------------
            # base_feat = base_onehot if self.use_base else [0.0] * BASE_DIM
            # motif_feat = motif_onehot if self.use_motif else [0.0] * MOTIF_DIM
            # ss_feat = ss_onehot if self.use_ss else [0.0] * SS_DIM
            sasa_feat = [sasa_norm, sasa_rel] if self.use_sasa else [0.0] * SASA_DIM
            geometry_feat = (
                [dist_centroid, neighbor_count, pos_norm, mean_nbr_dist, min_nbr_dist]
                if self.use_geometry else [0.0] * GEOMETRY_DIM
            )
            # chemical_feat = (
            #     [purine, donors, acceptors, rings, mw_norm]
            #     if self.use_atomic else [0.0] * ATOMIC_DIM
            # )
            # window_feat = (
            #     [win_frac_A, win_frac_U, win_frac_G, win_frac_C, win_gc_frac]
            #     if self.use_window else [0.0] * WINDOW_DIM
            # )
            # topology_feat = (
            #     [segment_length_log, dist_boundary_clipped]
            #     if self.use_topology else [0.0] * TOPOLOGY_DIM
            # )
            graph_feat = (
                [node_degree, node_clustering]
                if self.use_graph else [0.0] * GRAPH_DIM
            )

            # feat = (
            #      motif_feat + ss_feat +  geometry_feat
            #     + chemical_feat + window_feat + graph_feat
            # )
            feat = (
                 sasa_feat +  geometry_feat
                 + graph_feat
            )

            assert len(feat) == HANDCRAFTED_DIM, \
                f"Expected {HANDCRAFTED_DIM} handcrafted features, got {len(feat)}"

            if self.debug and i == 0:
                print("[DEBUG] Feature group dims (node 0):")
                # print(f"motif={len(motif_feat)} ss={len(ss_feat)} "
                #       f" geometry={len(geometry_feat)} "
                #       f"chemical={len(chemical_feat)} window={len(window_feat)} "
                #       f"graph={len(graph_feat)}")
                print(f"[DEBUG] handcrafted feature vector (node 0): {feat}")
                print(f"[DEBUG] handcrafted dim: {len(feat)}")

            # RNABERT embedding, appended last per the feature order.
            # if rnabert_emb is not None:
            #     feat = feat + rnabert_emb[i].astype(np.float32).tolist()

            if rnabert_emb is not None:
                feat += rnabert_emb[i].astype(np.float32).tolist()

            if ernie_emb is not None:
                feat += ernie_emb[i].astype(np.float32).tolist()

            if rnafm_emb is not None:
                feat += rnafm_emb[i].astype(np.float32).tolist()

            if self.debug and i == 0 and rnabert_emb is not None:
                print(f"[DEBUG] final node feature dim (with RNABERT): {len(feat)}")

            x.append(feat)
            y.append(int(node["label"]))

        # --------------------------------------------------------
        # Tensor conversion
        # --------------------------------------------------------
        x_tensor = torch.tensor(x, dtype=torch.float)
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_type_tensor = torch.tensor(edge_type, dtype=torch.long)
        edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float)
        y_tensor = torch.tensor(y, dtype=torch.float)

        if self.debug:
            print(f"[DEBUG] final x shape: {tuple(x_tensor.shape)}")

        # print("Final feature dimension:", x_tensor.shape[1])
        return Data(
            x=x_tensor,
            edge_index=edge_index_tensor,
            edge_type=edge_type_tensor,
            edge_attr=edge_attr_tensor,
            y=y_tensor
        )