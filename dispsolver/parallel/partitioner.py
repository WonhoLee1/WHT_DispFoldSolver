"""
partitioner.py
==============
Domain Decomposition and Mesh Partitioning for MPI Parallel FEA in WHT_DispFoldSolver.

Partitions 2D (Mesh2D) and 3D (Mesh3D) finite element meshes into P non-overlapping
element sets (subdomains), classifying nodes into:
  1. Internal nodes: Nodes belonging strictly to elements within this subdomain.
  2. Interface nodes: Nodes shared across boundary interfaces between subdomains.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Any, Optional, Union
import numpy as np

from dispsolver.mesh2d.mesh2d import Mesh2D, Node2D, Element2D
from dispsolver.mesh3d.mesh3d import Mesh3D, Node3D, Element3D


@dataclass
class Subdomain:
    """Represents a local subdomain partition for an MPI rank."""
    rank: int
    num_ranks: int
    elem_ids: List[int]
    node_ids: List[int]
    internal_node_ids: List[int]
    interface_node_ids: List[int]
    neighbor_ranks: List[int]
    shared_nodes_with_rank: Dict[int, List[int]]
    global_to_local_node: Dict[int, int]
    local_to_global_node: List[int]
    sub_mesh: Union[Mesh2D, Mesh3D]

    @property
    def num_elements(self) -> int:
        return len(self.elem_ids)

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def num_internal_nodes(self) -> int:
        return len(self.internal_node_ids)

    @property
    def num_interface_nodes(self) -> int:
        return len(self.interface_node_ids)


class DomainPartitioner:
    """Geometric & Graph-inspired Domain Decomposition Partitioner for 2D & 3D Meshes."""

    @staticmethod
    def compute_element_centroids(mesh: Union[Mesh2D, Mesh3D]) -> Dict[int, np.ndarray]:
        """Computes the geometric center of each element in the mesh."""
        centroids = {}
        is_3d = isinstance(mesh, Mesh3D)

        for eid, elem in mesh.elements.items():
            pts = []
            for nid in elem.node_ids:
                node = mesh.nodes[nid]
                if is_3d:
                    pts.append([node.x, node.y, node.z])
                else:
                    pts.append([node.x, node.y])
            centroids[eid] = np.mean(pts, axis=0)

        return centroids

    @classmethod
    def partition(
        cls,
        mesh: Union[Mesh2D, Mesh3D],
        num_partitions: int,
        axis: str = "auto"
    ) -> List[Subdomain]:
        """Partitions the given mesh into P subdomains.

        Parameters
        ----------
        mesh : Union[Mesh2D, Mesh3D]
            Global FE mesh to partition.
        num_partitions : int
            Number of target subdomains (typically equal to MPI comm.size).
        axis : str
            Partition axis: "auto", "x", "y", or "z" (for 3D). If "auto", picks
            the coordinate axis with the largest bounding box span.

        Returns
        -------
        List[Subdomain]
            List of Subdomain objects, indexed by rank 0 .. num_partitions - 1.
        """
        if num_partitions < 1:
            raise ValueError(f"num_partitions must be >= 1, got {num_partitions}")

        is_3d = isinstance(mesh, Mesh3D)
        centroids = cls.compute_element_centroids(mesh)
        elem_ids = list(centroids.keys())

        # If 1 partition, trivial identity mapping
        if num_partitions == 1:
            all_nids = list(mesh.nodes.keys())
            g2l = {nid: i for i, nid in enumerate(all_nids)}
            sub = Subdomain(
                rank=0,
                num_ranks=1,
                elem_ids=elem_ids,
                node_ids=all_nids,
                internal_node_ids=all_nids,
                interface_node_ids=[],
                neighbor_ranks=[],
                shared_nodes_with_rank={},
                global_to_local_node=g2l,
                local_to_global_node=all_nids,
                sub_mesh=mesh
            )
            return [sub]

        # Determine partition coordinate axis
        all_centers = np.array([centroids[eid] for eid in elem_ids])
        bbox_min = np.min(all_centers, axis=0)
        bbox_max = np.max(all_centers, axis=0)
        spans = bbox_max - bbox_min

        axis_map = {"x": 0, "y": 1, "z": 2}
        if axis.lower() == "auto":
            axis_idx = int(np.argmax(spans))
        else:
            axis_idx = axis_map.get(axis.lower(), 0)
            if axis_idx >= all_centers.shape[1]:
                axis_idx = 0

        # Sort element IDs by coordinate along the partition axis
        sorted_indices = np.argsort(all_centers[:, axis_idx])
        sorted_elem_ids = [elem_ids[idx] for idx in sorted_indices]

        # Divide elements into P contiguous chunks
        n_elems = len(sorted_elem_ids)
        chunk_size = n_elems // num_partitions
        remainder = n_elems % num_partitions

        partition_elems: List[List[int]] = []
        cur_idx = 0
        for p in range(num_partitions):
            size_p = chunk_size + (1 if p < remainder else 0)
            p_eids = sorted_elem_ids[cur_idx:cur_idx + size_p]
            partition_elems.append(p_eids)
            cur_idx += size_p

        # Node-to-partition membership tracking
        node_to_parts: Dict[int, Set[int]] = {nid: set() for nid in mesh.nodes.keys()}
        for p, p_eids in enumerate(partition_elems):
            for eid in p_eids:
                elem = mesh.elements[eid]
                for nid in elem.node_ids:
                    node_to_parts[nid].add(p)

        # Build Subdomain objects
        subdomains: List[Subdomain] = []

        for p in range(num_partitions):
            p_eids = partition_elems[p]
            p_nids_set: Set[int] = set()
            for eid in p_eids:
                elem = mesh.elements[eid]
                p_nids_set.update(elem.node_ids)

            p_nids = sorted(list(p_nids_set))
            internal_nids = []
            interface_nids = []
            neighbor_ranks_set: Set[int] = set()
            shared_nodes: Dict[int, List[int]] = {}

            for nid in p_nids:
                parts = node_to_parts[nid]
                if len(parts) == 1:
                    internal_nids.append(nid)
                else:
                    interface_nids.append(nid)
                    for other_p in parts:
                        if other_p != p:
                            neighbor_ranks_set.add(other_p)
                            if other_p not in shared_nodes:
                                shared_nodes[other_p] = []
                            shared_nodes[other_p].append(nid)

            neighbor_ranks = sorted(list(neighbor_ranks_set))
            g2l = {nid: idx for idx, nid in enumerate(p_nids)}

            # Build sub-mesh for this subdomain
            if is_3d:
                sub_mesh = Mesh3D()
                for nid in p_nids:
                    node = mesh.nodes[nid]
                    sub_mesh.add_node(nid, node.x, node.y, node.z)
                for eid in p_eids:
                    elem = mesh.elements[eid]
                    sub_mesh.add_element(eid, elem.node_ids, elem.elem_type, elem.pid)
            else:
                sub_mesh = Mesh2D()
                for nid in p_nids:
                    node = mesh.nodes[nid]
                    sub_mesh.add_node(nid, node.x, node.y)
                for eid in p_eids:
                    elem = mesh.elements[eid]
                    sub_mesh.add_element(eid, elem.node_ids, elem.elem_type, elem.pid)

            sub = Subdomain(
                rank=p,
                num_ranks=num_partitions,
                elem_ids=p_eids,
                node_ids=p_nids,
                internal_node_ids=internal_nids,
                interface_node_ids=interface_nids,
                neighbor_ranks=neighbor_ranks,
                shared_nodes_with_rank=shared_nodes,
                global_to_local_node=g2l,
                local_to_global_node=p_nids,
                sub_mesh=sub_mesh
            )
            subdomains.append(sub)

        return subdomains
