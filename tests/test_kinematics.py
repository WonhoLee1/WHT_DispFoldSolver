"""
test_kinematics.py
==================
Patch tests for Q4 (SRI B-bar) and T3 elements.

AC-2: Patch test (Q4, T3) — uniform stress error < 1e-10

Test methodology
----------------
1. Build a 2×2 Q4 (or T3) mesh of a unit square [0,1]×[0,1]
2. Apply consistent boundary displacements:
     ux = α·x,  uy = -ν·α·y  (plane strain uniaxial tension in x)
3. Compute element strains via B-matrix
4. Verify: all elements produce identical strain components within |ε - ε_exact| < 1e-12
   (stricter than the AC-2 requirement of < 1e-10)
5. Compute element stiffness and verify symmetry and positive semi-definiteness
"""

from __future__ import annotations

import numpy as np
import pytest

from dispsolver.mesh import Mesh
from dispsolver.element.q4 import (
    compute_K_elem,
    compute_M_elem,
    compute_strains,
    compute_stress,
    plane_strain_D,
    jacobian,
    B_matrix,
    B_bar_matrix,
)
from dispsolver.element.t3 import (
    compute_K_elem as compute_K_elem_t3,
    compute_M_elem as compute_M_elem_t3,
    compute_strains as compute_strains_t3,
    compute_stress as compute_stress_t3,
)

# ------------------------------------------------------------------
# Test parameters
# ------------------------------------------------------------------
E = 1000.0
NU = 0.3
ALPHA = 1e-3  # applied strain
RHO = 1.0


def _exact_stress() -> np.ndarray:
    """Exact stress for plane strain: σ_xx = E·α, σ_yy = 0, τ_xy = 0."""
    D = plane_strain_D(E, NU)
    eps_exact = np.array([ALPHA, -NU * ALPHA, 0.0])
    return D @ eps_exact


def _build_q4_patch_mesh() -> Mesh:
    """Build a 2×2 Q4 patch mesh with a center node and mid-edge nodes.

    Node layout:
        4(0,1)───7(0.5,1)───3(1,1)
         |          |          |
        8(0,0.5)──5(0.5,0.5)──6(1,0.5)
         |          |          |
        1(0,0)────2(0.5,0)────2(1,0) → actually node 2 and 2

    Actually let me use a simpler 4-element mesh:
    Nodes:
      1: (0,0),  2: (0.5,0),  3: (1,0)
      4: (0,0.5), 5: (0.5,0.5), 6: (1,0.5)
      7: (0,1),   8: (0.5,1),   9: (1,1)
    Elements:
      E1: 1-2-5-4  (bottom-left)
      E2: 2-3-6-5  (bottom-right)
      E3: 4-5-8-7  (top-left)
      E4: 5-6-9-8  (top-right)
    """
    mesh = Mesh()
    # Nodes
    mesh.add_node(1, 0.0, 0.0)
    mesh.add_node(2, 0.5, 0.0)
    mesh.add_node(3, 1.0, 0.0)
    mesh.add_node(4, 0.0, 0.5)
    mesh.add_node(5, 0.5, 0.5)
    mesh.add_node(6, 1.0, 0.5)
    mesh.add_node(7, 0.0, 1.0)
    mesh.add_node(8, 0.5, 1.0)
    mesh.add_node(9, 1.0, 1.0)

    # Elements (Q4, CCW)
    mesh.add_element(1, [1, 2, 5, 4], "QUAD4")
    mesh.add_element(2, [2, 3, 6, 5], "QUAD4")
    mesh.add_element(3, [4, 5, 8, 7], "QUAD4")
    mesh.add_element(4, [5, 6, 9, 8], "QUAD4")
    return mesh


def _build_t3_patch_mesh() -> Mesh:
    """Build a 4-triangle patch mesh covering [0,1]×[0,1].

    Node layout same as Q4 (topological)::

    7---8---9
    |&#47;|&#47;|
    |&#47;  |&#47;  |
    4---5---6
    |&#47;|&#47;|
    |&#47;  |&#47;  |
    1---2---3

    Elements:
      E1: 1-5-4  (center→split)
      E2: 1-2-5
      E3: 2-3-6
      E4: 2-6-5
      E5: 4-5-8
      E6: 4-8-7
      E7: 5-9-8
      E8: 5-6-9
    """
    mesh = Mesh()
    # Same nodes
    mesh.add_node(1, 0.0, 0.0)
    mesh.add_node(2, 0.5, 0.0)
    mesh.add_node(3, 1.0, 0.0)
    mesh.add_node(4, 0.0, 0.5)
    mesh.add_node(5, 0.5, 0.5)
    mesh.add_node(6, 1.0, 0.5)
    mesh.add_node(7, 0.0, 1.0)
    mesh.add_node(8, 0.5, 1.0)
    mesh.add_node(9, 1.0, 1.0)

    # 8 triangles
    mesh.add_element(1, [1, 5, 4], "TRIA3")
    mesh.add_element(2, [1, 2, 5], "TRIA3")
    mesh.add_element(3, [2, 3, 6], "TRIA3")
    mesh.add_element(4, [2, 6, 5], "TRIA3")
    mesh.add_element(5, [4, 5, 8], "TRIA3")
    mesh.add_element(6, [4, 8, 7], "TRIA3")
    mesh.add_element(7, [5, 9, 8], "TRIA3")
    mesh.add_element(8, [5, 6, 9], "TRIA3")
    return mesh


# ------------------------------------------------------------------
# Q4 Patch Test
# ------------------------------------------------------------------

class TestQ4PatchTest:
    """Q4 element patch test — constant strain reproduction."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.mesh = _build_q4_patch_mesh()
        self.nid_to_idx = self.mesh.node_id_to_index()
        self.exact_stress = _exact_stress()
        self.exact_strain = np.array([ALPHA, -NU * ALPHA, 0.0])

    def test_q4_constant_strain_patch(self):
        """All Q4 elements reproduce identical ε_xx, ε_yy, γ_xy under linear displacement."""
        conn, _, _, _ = self.mesh.connectivity_array("QUAD4")
        node_coords = self.mesh.nodes_array()

        eps_list = []
        for elem_nodes in conn:
            coords = node_coords[elem_nodes]
            # Build element nodal displacements: ux = α·x, uy = -ν·α·y
            u_elem = np.zeros(8, dtype=np.float64)
            for i in range(4):
                u_elem[2 * i] = ALPHA * coords[i, 0]
                u_elem[2 * i + 1] = -NU * ALPHA * coords[i, 1]
            # Compute strains at element center
            eps = compute_strains(coords, u_elem, xi=0.0, eta=0.0)
            eps_list.append(eps)

        eps_arr = np.array(eps_list)
        # Each strain component should be identical across all elements
        for comp in range(3):
            diff = np.max(np.abs(eps_arr[:, comp] - self.exact_strain[comp]))
            assert diff < 1e-12, (
                f"Q4 patch: ε component {comp} max error {diff:.2e} > 1e-12"
            )

    def test_q4_constant_stress_patch(self):
        """All Q4 elements reproduce identical stress under linear displacement."""
        conn, _, _, _ = self.mesh.connectivity_array("QUAD4")
        node_coords = self.mesh.nodes_array()

        sigma_list = []
        for elem_nodes in conn:
            coords = node_coords[elem_nodes]
            u_elem = np.zeros(8)
            for i in range(4):
                u_elem[2 * i] = ALPHA * coords[i, 0]
                u_elem[2 * i + 1] = -NU * ALPHA * coords[i, 1]
            sigma = compute_stress(coords, u_elem, E, NU, xi=0.0, eta=0.0)
            sigma_list.append(sigma)

        sigma_arr = np.array(sigma_list)
        for comp in range(3):
            diff = np.max(np.abs(sigma_arr[:, comp] - self.exact_stress[comp]))
            assert diff < 1e-12, (
                f"Q4 patch: σ component {comp} max error {diff:.2e} > 1e-12"
            )

    def test_q4_stiffness_symmetry(self):
        """Q4 element stiffness matrix is symmetric and positive semi-definite."""
        node_coords = self.mesh.nodes_array()
        conn, _, _, _ = self.mesh.connectivity_array("QUAD4")

        for elem_nodes in conn:
            coords = node_coords[elem_nodes]
            K = compute_K_elem(coords, E, NU)
            # Symmetry
            assert np.allclose(K, K.T, atol=1e-14), "Q4 K not symmetric"
            # Eigenvalues ≥ 0 (within numerical tolerance)
            eigvals = np.linalg.eigvalsh(K)
            assert eigvals[0] > -1e-10, (
                f"Q4 K has negative eigenvalue: {eigvals[0]:.2e}"
            )

    def test_q4_mass_symmetry(self):
        """Q4 consistent mass matrix is symmetric and positive definite."""
        node_coords = self.mesh.nodes_array()
        conn, _, _, _ = self.mesh.connectivity_array("QUAD4")

        for elem_nodes in conn:
            coords = node_coords[elem_nodes]
            M = compute_M_elem(coords, RHO)
            assert np.allclose(M, M.T, atol=1e-14), "Q4 M not symmetric"
            eigvals = np.linalg.eigvalsh(M)
            assert eigvals[0] > -1e-10, (
                f"Q4 M has negative eigenvalue: {eigvals[0]:.2e}"
            )
            # Mass should be positive
            assert np.trace(M) > 0, "Q4 M trace <= 0"

    def test_q4_jacobian_positive(self):
        """Q4 Jacobian determinant is positive at all integration points."""
        node_coords = self.mesh.nodes_array()
        conn, _, _, _ = self.mesh.connectivity_array("QUAD4")

        for elem_nodes in conn:
            coords = node_coords[elem_nodes]
            for xi, eta in [(-1, -1), (1, -1), (1, 1), (-1, 1), (0, 0)]:
                xi_f = float(xi) / np.sqrt(3) if abs(xi) == 1 else float(xi)
                eta_f = float(eta) / np.sqrt(3) if abs(eta) == 1 else float(eta)
                J, detJ, invJ = jacobian(xi_f, eta_f, coords)
                assert detJ > 0, (
                    f"Q4 negative detJ={detJ:.2e} at (ξ={xi_f}, η={eta_f})"
                )


# ------------------------------------------------------------------
# T3 Patch Test
# ------------------------------------------------------------------

class TestT3PatchTest:
    """T3 element patch test — constant strain reproduction."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.mesh = _build_t3_patch_mesh()
        self.exact_stress = _exact_stress()
        self.exact_strain = np.array([ALPHA, -NU * ALPHA, 0.0])

    def test_t3_constant_strain_patch(self):
        """All T3 elements reproduce identical constant strain."""
        conn, _, _, _ = self.mesh.connectivity_array("TRIA3")
        node_coords = self.mesh.nodes_array()

        eps_list = []
        for elem_nodes in conn:
            coords = node_coords[elem_nodes]
            u_elem = np.zeros(6)
            for i in range(3):
                u_elem[2 * i] = ALPHA * coords[i, 0]
                u_elem[2 * i + 1] = -NU * ALPHA * coords[i, 1]
            eps = compute_strains_t3(coords, u_elem)
            eps_list.append(eps)

        eps_arr = np.array(eps_list)
        for comp in range(3):
            diff = np.max(np.abs(eps_arr[:, comp] - self.exact_strain[comp]))
            assert diff < 1e-12, (
                f"T3 patch: ε component {comp} max error {diff:.2e} > 1e-12"
            )

    def test_t3_constant_stress_patch(self):
        """All T3 elements reproduce identical constant stress."""
        conn, _, _, _ = self.mesh.connectivity_array("TRIA3")
        node_coords = self.mesh.nodes_array()

        sigma_list = []
        for elem_nodes in conn:
            coords = node_coords[elem_nodes]
            u_elem = np.zeros(6)
            for i in range(3):
                u_elem[2 * i] = ALPHA * coords[i, 0]
                u_elem[2 * i + 1] = -NU * ALPHA * coords[i, 1]
            sigma = compute_stress_t3(coords, u_elem, E, NU)
            sigma_list.append(sigma)

        sigma_arr = np.array(sigma_list)
        for comp in range(3):
            diff = np.max(np.abs(sigma_arr[:, comp] - self.exact_stress[comp]))
            assert diff < 1e-12, (
                f"T3 patch: σ component {comp} max error {diff:.2e} > 1e-12"
            )

    def test_t3_stiffness_symmetry(self):
        """T3 element stiffness matrix is symmetric."""
        conn, _, _, _ = self.mesh.connectivity_array("TRIA3")
        node_coords = self.mesh.nodes_array()

        for elem_nodes in conn:
            coords = node_coords[elem_nodes]
            K = compute_K_elem_t3(coords, E, NU)
            assert np.allclose(K, K.T, atol=1e-14), "T3 K not symmetric"
            eigvals = np.linalg.eigvalsh(K)
            assert eigvals[0] > -1e-10, (
                f"T3 K has negative eigenvalue: {eigvals[0]:.2e}"
            )

    def test_t3_mass_symmetry(self):
        """T3 mass matrix is symmetric and positive."""
        conn, _, _, _ = self.mesh.connectivity_array("TRIA3")
        node_coords = self.mesh.nodes_array()

        for elem_nodes in conn:
            coords = node_coords[elem_nodes]
            M = compute_M_elem_t3(coords, RHO)
            assert np.allclose(M, M.T, atol=1e-14), "T3 M not symmetric"
            eigvals = np.linalg.eigvalsh(M)
            assert eigvals[0] > -1e-10, (
                f"T3 M has negative eigenvalue: {eigvals[0]:.2e}"
            )
            assert np.trace(M) > 0, "T3 M trace <= 0"

    def test_t3_jacobian_positive(self):
        """T3 Jacobian determinant is positive for valid elements."""
        from dispsolver.element.t3 import jacobian as t3_jacobian
        conn, _, _, _ = self.mesh.connectivity_array("TRIA3")
        node_coords = self.mesh.nodes_array()

        for elem_nodes in conn:
            coords = node_coords[elem_nodes]
            J, detJ, invJ = t3_jacobian(coords)
            assert detJ > 0, f"T3 negative detJ={detJ:.2e}"


# ------------------------------------------------------------------
# Mixed verification: global assembly sanity check
# ------------------------------------------------------------------

class TestGlobalAssembly:
    """Verify that elements can be assembled into a global system."""

    def test_global_stiffness_assembly_q4(self):
        """Assemble global K from Q4 elements; verify symmetry and size."""
        mesh = _build_q4_patch_mesh()
        conn, nid_to_idx, sorted_nids, _ = mesh.connectivity_array("QUAD4")
        node_coords = mesh.nodes_array()
        ndof = len(sorted_nids) * 2

        from scipy.sparse import coo_matrix

        rows, cols, vals = [], [], []
        for elem_nodes in conn:
            coords = node_coords[elem_nodes]
            K_elem = compute_K_elem(coords, E, NU)
            dof_indices = np.zeros(8, dtype=np.int32)
            for i, nid_idx in enumerate(elem_nodes):
                dof_indices[2 * i] = 2 * nid_idx
                dof_indices[2 * i + 1] = 2 * nid_idx + 1
            ii = np.repeat(dof_indices, 8)
            jj = np.tile(dof_indices, 8)
            rows.extend(ii)
            cols.extend(jj)
            vals.extend(K_elem.ravel())

        K_global = coo_matrix(
            (vals, (rows, cols)), shape=(ndof, ndof)
        ).tocsr()

        assert K_global.shape == (ndof, ndof)
        # Symmetry
        diff = np.abs(K_global - K_global.T).max()
        assert diff < 1e-10, f"Global K symmetry error: {diff:.2e}"

        # Solve: K·u = f for the patch test
        # Apply BC: left edge ux=0, bottom edge uy=0
        # Load: right edge ux = ALPHA
        bc_dofs = []
        prescribed = {}
        for nid, idx in nid_to_idx.items():
            node = mesh.get_node(nid)
            if abs(node.x) < 1e-12:  # left edge: ux=0
                bc_dofs.append(2 * idx)
                prescribed[2 * idx] = 0.0
            if abs(node.y) < 1e-12:  # bottom edge: uy=0
                bc_dofs.append(2 * idx + 1)
                prescribed[2 * idx + 1] = 0.0
            if abs(node.x - 1.0) < 1e-12:  # right edge: ux = ALPHA
                prescribed[2 * idx] = ALPHA * node.x
            if abs(node.y - 1.0) < 1e-12:  # top edge: uy = -NU*ALPHA
                prescribed[2 * idx + 1] = -NU * ALPHA * node.y

        # Partition: free vs prescribed
        all_dofs = set(range(ndof))
        prescribed_set = set(prescribed.keys())
        free_dofs = sorted(all_dofs - prescribed_set)
        bc_dofs = sorted(prescribed_set)

        Kff = K_global[np.ix_(free_dofs, free_dofs)].toarray()
        Kfb = K_global[np.ix_(free_dofs, bc_dofs)].toarray()

        # f = 0 (no external loads — pure displacement BC)
        ub = np.array([prescribed[d] for d in bc_dofs])
        uf = -np.linalg.solve(Kff, Kfb @ ub)

        # Full displacement vector
        u_full = np.zeros(ndof)
        for i, d in enumerate(free_dofs):
            u_full[d] = uf[i]
        for i, d in enumerate(bc_dofs):
            u_full[d] = ub[i]

        # Verify: internal nodes have correct displacement
        for nid, idx in nid_to_idx.items():
            node = mesh.get_node(nid)
            ux = u_full[2 * idx]
            uy = u_full[2 * idx + 1]
            assert abs(ux - ALPHA * node.x) < 1e-10, (
                f"Node {nid}: ux error = {abs(ux - ALPHA * node.x):.2e}"
            )
            assert abs(uy + NU * ALPHA * node.y) < 1e-10, (
                f"Node {nid}: uy error = {abs(uy + NU * ALPHA * node.y):.2e}"
            )
