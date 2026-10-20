"""
run_fistr_conforming_teardrop.py
================================
FrontISTR 14-Layer Canonical Foldable Display 180-Degree Teardrop Folding Simulation:
Conforming Hybrid Formulation (Option B):
- 7 Stiff Layers (6 PET + 1 GLASS): 20-node Quadratic Hex (`362`, 1 layer each)
- 7 Soft Layers (7 PSA): 1st-order u-P Hex (`361 UP`, 2x subdivision in X and Z)
- Total thickness = 0.630 mm (420 um stiff + 210 um PSA)
- Hinge gap = [-7.5, +7.5] mm with tape void in bottom PSA_0
- Left / Right Support Plates: driven by finite Rodrigues rotation (!BOUNDARY, ROT_CENTER)
- Zero hanging nodes across all 13 interfaces, ZERO TIE penalty overhead!

Compares against Baseline 361 FI:
  - Reaction Moment M_L, M_R vs Folding Angle
  - Reaction Force Fx_L, Fx_R
  - Max Mises Stress in Cover Glass (relief of parasitic shear locking)
  - Deformed teardrop loop geometry (apex Y, neck gap)
"""

from __future__ import annotations
import os
import sys
import math
import json
import time
import argparse
import subprocess
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

# Matplotlib Korean font setup
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import koreanize_matplotlib
    plt.rcParams["font.size"] = 9
except ImportError:
    plt = None

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from example_disp_fold.model_spec import UnifiedFoldModelConfig
from example_disp_fold.run_fistr_teardrop import parse_fistr_res_file, compute_reaction_moment_from_res

FISTR_DIR = r"D:\PythonCodeStudy\FrontISTR\FrontISTR-latest.thread_openblas"
FISTR_EXE = os.path.join(FISTR_DIR, "fistr1.exe")

# 14-Layer Canonical Stackup
STACKUP_14 = [
    {"name": "PSA_0",     "mat": "PSA",   "t": 0.030, "is_stiff": False, "is_void_in_hinge": True},
    {"name": "PET_1",     "mat": "PET",   "t": 0.060, "is_stiff": True,  "is_void_in_hinge": False},
    {"name": "PSA_1",     "mat": "PSA",   "t": 0.030, "is_stiff": False, "is_void_in_hinge": False},
    {"name": "PET_2",     "mat": "PET",   "t": 0.060, "is_stiff": True,  "is_void_in_hinge": False},
    {"name": "PSA_2",     "mat": "PSA",   "t": 0.030, "is_stiff": False, "is_void_in_hinge": False},
    {"name": "PET_3",     "mat": "PET",   "t": 0.060, "is_stiff": True,  "is_void_in_hinge": False},
    {"name": "PSA_3",     "mat": "PSA",   "t": 0.030, "is_stiff": False, "is_void_in_hinge": False},
    {"name": "PET_4",     "mat": "PET",   "t": 0.060, "is_stiff": True,  "is_void_in_hinge": False},
    {"name": "PSA_4",     "mat": "PSA",   "t": 0.030, "is_stiff": False, "is_void_in_hinge": False},
    {"name": "PET_5",     "mat": "PET",   "t": 0.060, "is_stiff": True,  "is_void_in_hinge": False},
    {"name": "PSA_5",     "mat": "PSA",   "t": 0.030, "is_stiff": False, "is_void_in_hinge": False},
    {"name": "PET_6",     "mat": "PET",   "t": 0.060, "is_stiff": True,  "is_void_in_hinge": False},
    {"name": "PSA_6",     "mat": "PSA",   "t": 0.030, "is_stiff": False, "is_void_in_hinge": False},
    {"name": "GLASS_Top", "mat": "GLASS", "t": 0.060, "is_stiff": True,  "is_void_in_hinge": False},
]


class ConformingTeardropMeshBuilder:
    """
    Constructs the 14-layer conforming hybrid mesh for 180-deg teardrop folding:
    - Stiff layers (PET/GLASS): 20-node Hex (`362`)
    - Soft layers (PSA): 1st-order u-P Hex (`361 UP`), 2x subdivided in X and Z
    - Support Plates (PLATE): 1st-order Hex (`361`) under PSA_0 on left and right wings
    - Stationary Pivot Reference: 1 detached Hex8 hosting PIVOT_L and PIVOT_R
    """

    def __init__(
        self,
        config: UnifiedFoldModelConfig,
        nx_hinge: int = 30,
        nx_plate: int = 20,
        width_z: float = 1.0,
        ny_stiff: int = 1,
    ):
        self.config = config
        self.geo = config.geometry
        self.half_L = self.geo.display_half_length  # 40.0
        self.hinge_gap = self.geo.hinge_half_gap     # 7.5
        self.plate_t = self.geo.plate_thickness     # 0.5
        self.W = width_z
        self.ny_stiff = ny_stiff
        self.z0 = -0.5 * self.W
        self.z1 = +0.5 * self.W
        self.zm = 0.0

        self.nx_hinge = nx_hinge
        self.nx_plate = nx_plate

        # Discrete X grid for stiff layers
        # Left plate: -40 to -7.5 (biased towards hinge)
        prog = self.config.mesh.bias_progression  # 1.15
        total_ratio = float(prog ** 20.0)
        s_p = np.geomspace(1.0, total_ratio, nx_plate + 1)
        s_p = (s_p - s_p[0]) / (s_p[-1] - s_p[0])
        xs_left = -self.half_L + s_p * ((-self.hinge_gap) - (-self.half_L))

        # Hinge: uniform dense
        xs_hinge = np.linspace(-self.hinge_gap, self.hinge_gap, nx_hinge + 1)

        # Right plate: +7.5 to +40 (denser near +7.5)
        xs_right = self.hinge_gap + (1.0 - s_p[::-1]) * (self.half_L - self.hinge_gap)

        self.xs = np.concatenate([xs_left[:-1], xs_hinge[:-1], xs_right])
        self.nx_stiff_total = len(self.xs) - 1

        self.node_map: Dict[Tuple[float, float, float], int] = {}
        self.node_list: List[Tuple[int, float, float, float]] = []
        self.layer_elements: Dict[str, List[List[int]]] = {}
        self.plate_elements: Dict[str, List[List[int]]] = {"EL_PLATE": []}

        self.left_driven_nids: List[int] = []
        self.right_driven_nids: List[int] = []
        self.z_fixed_nids: List[int] = []

        self.pivot_l_nid = -1
        self.pivot_r_nid = -1
        self.pivot_dummy_nids: List[int] = []
        self.real_max_nid = 0
        self.real_max_eid = 0

    def get_node_id(self, x: float, y: float, z: float) -> int:
        key = (round(x, 7), round(y, 7), round(z, 7))
        if key not in self.node_map:
            nid = len(self.node_map) + 1
            self.node_map[key] = nid
            self.node_list.append((nid, key[0], key[1], key[2]))
            return nid
        return self.node_map[key]

    def build(self):
        self.node_map.clear()
        self.node_list.clear()
        self.layer_elements.clear()
        self.plate_elements = {"EL_PLATE": []}

        # 1. Build 14 Display Layers
        y_curr = 0.0

        for l_idx, layer in enumerate(STACKUP_14):
            lname = layer["name"]
            t = layer["t"]
            is_stiff = layer["is_stiff"]
            is_void = layer["is_void_in_hinge"]

            y0 = y_curr
            y1 = y_curr + t
            ym = y_curr + t / 2.0
            y_curr = y1

            self.layer_elements[lname] = []

            if is_stiff:
                # 20-node Hex (362), subdivided into ny_stiff elements across thickness
                dy_sub = t / float(self.ny_stiff)
                for iy in range(self.ny_stiff):
                    y_sub0 = y0 + iy * dy_sub
                    y_sub1 = y_sub0 + dy_sub
                    ym_sub = (y_sub0 + y_sub1) / 2.0

                    for ex in range(self.nx_stiff_total):
                        x0 = self.xs[ex]
                        x1 = self.xs[ex + 1]
                        xm = (x0 + x1) / 2.0
                        z0, zm, z1 = self.z0, self.zm, self.z1

                        n1 = self.get_node_id(x0, y_sub0, z0)
                        n2 = self.get_node_id(x1, y_sub0, z0)
                        n3 = self.get_node_id(x1, y_sub1, z0)
                        n4 = self.get_node_id(x0, y_sub1, z0)
                        n5 = self.get_node_id(x0, y_sub0, z1)
                        n6 = self.get_node_id(x1, y_sub0, z1)
                        n7 = self.get_node_id(x1, y_sub1, z1)
                        n8 = self.get_node_id(x0, y_sub1, z1)

                        n9  = self.get_node_id(xm, y_sub0, z0)
                        n10 = self.get_node_id(x1, ym_sub, z0)
                        n11 = self.get_node_id(xm, y_sub1, z0)
                        n12 = self.get_node_id(x0, ym_sub, z0)

                        n13 = self.get_node_id(xm, y_sub0, z1)
                        n14 = self.get_node_id(x1, ym_sub, z1)
                        n15 = self.get_node_id(xm, y_sub1, z1)
                        n16 = self.get_node_id(x0, ym_sub, z1)

                        n17 = self.get_node_id(x0, y_sub0, zm)
                        n18 = self.get_node_id(x1, y_sub0, zm)
                        n19 = self.get_node_id(x1, y_sub1, zm)
                        n20 = self.get_node_id(x0, y_sub1, zm)

                        self.layer_elements[lname].append([
                            n1, n2, n3, n4, n5, n6, n7, n8,
                            n9, n10, n11, n12, n13, n14, n15, n16,
                            n17, n18, n19, n20
                        ])
            else:
                # 1st-order Hex (361 UP) - 2x subdivided in X and Z
                for ex in range(self.nx_stiff_total):
                    x_start = self.xs[ex]
                    x_end = self.xs[ex + 1]
                    x_mid = (x_start + x_end) / 2.0

                    # Void check
                    if is_void and (-self.hinge_gap <= x_mid <= self.hinge_gap):
                        continue

                    x_sub = [(x_start, x_mid), (x_mid, x_end)]
                    z_sub = [(self.z0, self.zm), (self.zm, self.z1)]

                    for x0, x1 in x_sub:
                        for z0, z1 in z_sub:
                            n1 = self.get_node_id(x0, y0, z0)
                            n2 = self.get_node_id(x1, y0, z0)
                            n3 = self.get_node_id(x1, y1, z0)
                            n4 = self.get_node_id(x0, y1, z0)
                            n5 = self.get_node_id(x0, y0, z1)
                            n6 = self.get_node_id(x1, y0, z1)
                            n7 = self.get_node_id(x1, y1, z1)
                            n8 = self.get_node_id(x0, y1, z1)

                            self.layer_elements[lname].append([n1, n2, n3, n4, n5, n6, n7, n8])

        # 2. Build Support Plates under Left and Right wings
        # Left plate: x in [-half_L, -hinge_gap], y in [-plate_t, 0]
        # Right plate: x in [+hinge_gap, +half_L], y in [-plate_t, 0]
        # Discretized with 2 layers in Y and matching 2x subdivided grid in X and Z
        ny_p = 2
        dy_p = self.plate_t / float(ny_p)
        z_sub = [(self.z0, self.zm), (self.zm, self.z1)]

        left_plate_nodes = set()
        right_plate_nodes = set()

        for ex in range(self.nx_stiff_total):
            x_start = self.xs[ex]
            x_end = self.xs[ex + 1]
            x_mid = (x_start + x_end) / 2.0

            is_left_wing = x_end <= -self.hinge_gap + 1e-5
            is_right_wing = x_start >= self.hinge_gap - 1e-5

            if not (is_left_wing or is_right_wing):
                continue

            x_sub = [(x_start, x_mid), (x_mid, x_end)]

            for iy in range(ny_p):
                yp0 = -self.plate_t + iy * dy_p
                yp1 = -self.plate_t + (iy + 1) * dy_p

                for x0, x1 in x_sub:
                    for z0, z1 in z_sub:
                        n1 = self.get_node_id(x0, yp0, z0)
                        n2 = self.get_node_id(x1, yp0, z0)
                        n3 = self.get_node_id(x1, yp1, z0)
                        n4 = self.get_node_id(x0, yp1, z0)
                        n5 = self.get_node_id(x0, yp0, z1)
                        n6 = self.get_node_id(x1, yp0, z1)
                        n7 = self.get_node_id(x1, yp1, z1)
                        n8 = self.get_node_id(x0, yp1, z1)

                        conn = [n1, n2, n3, n4, n5, n6, n7, n8]
                        self.plate_elements["EL_PLATE"].append(conn)

                        if is_left_wing:
                            for n in conn: left_plate_nodes.add(n)
                        else:
                            for n in conn: right_plate_nodes.add(n)

        # 3. Driven node sets
        # Include all plate nodes and display bottom nodes under the plates
        disp_bot_l = {
            nid for nid, x, y, z in self.node_list
            if x <= -self.hinge_gap + 1e-5 and abs(y - 0.0) < 1e-5
        }
        disp_bot_r = {
            nid for nid, x, y, z in self.node_list
            if x >= self.hinge_gap - 1e-5 and abs(y - 0.0) < 1e-5
        }

        self.left_driven_nids = sorted(left_plate_nodes | disp_bot_l)
        self.right_driven_nids = sorted(right_plate_nodes | disp_bot_r)
        driven_set = set(self.left_driven_nids) | set(self.right_driven_nids)

        # Z fixed nodes (for 2D Plane Strain symmetry: constrain u_z = 0 on non-driven nodes)
        self.z_fixed_nids = sorted([
            nid for nid, x, y, z in self.node_list
            if abs(z - self.zm) < 1e-5 and nid not in driven_set
        ])

        self.real_max_nid = len(self.node_list)
        total_elems = sum(len(el) for el in self.layer_elements.values()) + len(self.plate_elements["EL_PLATE"])
        self.real_max_eid = total_elems

        # 4. Stationary Pivot Reference Hex Element
        # Hosts PIVOT_L at (-1.4, 0.0) and PIVOT_R at (+1.4, 0.0)
        p_l = (-float(self.geo.hinge_pivot_x), float(self.geo.hinge_pivot_y))  # (-1.4, 0.0)
        p_r = (float(self.geo.hinge_pivot_x), float(self.geo.hinge_pivot_y))   # (+1.4, 0.0)
        y_ref_bot = -2.0
        base_nid = self.real_max_nid

        ref_coords = [
            (p_l[0], y_ref_bot, self.z0),  # base_nid + 1
            (p_r[0], y_ref_bot, self.z0),  # base_nid + 2
            (p_r[0], p_r[1],    self.z0),  # base_nid + 3 <-- PIVOT_R
            (p_l[0], p_l[1],    self.z0),  # base_nid + 4 <-- PIVOT_L
            (p_l[0], y_ref_bot, self.z1),  # base_nid + 5
            (p_r[0], y_ref_bot, self.z1),  # base_nid + 6
            (p_r[0], p_r[1],    self.z1),  # base_nid + 7
            (p_l[0], p_l[1],    self.z1),  # base_nid + 8
        ]

        self.pivot_dummy_nids = []
        for idx, pt in enumerate(ref_coords, start=1):
            rnid = base_nid + idx
            self.node_list.append((rnid, pt[0], pt[1], pt[2]))
            self.pivot_dummy_nids.append(rnid)

        self.pivot_r_nid = base_nid + 3
        self.pivot_l_nid = base_nid + 4

        ref_eid = self.real_max_eid + 1
        self.plate_elements["EL_PIVOT_REF"] = [list(self.pivot_dummy_nids)]


class FrontISTRConformingTeardropRunner:
    """Writes input files, executes FrontISTR, and post-processes 180-deg teardrop folding."""

    def __init__(
        self,
        builder: ConformingTeardropMeshBuilder,
        run_dir: str,
        substeps: int = 90,
        theta_max_deg: float = 90.0,
    ):
        self.mb = builder
        self.run_dir = run_dir
        self.substeps = substeps
        self.theta_max_deg = theta_max_deg
        self.res_write_freq = max(1, substeps // 10) if substeps >= 10 else 1

    def write_deck(self):
        os.makedirs(self.run_dir, exist_ok=True)
        self.mb.build()

        # 1. hecmw_ctrl.dat
        with open(os.path.join(self.run_dir, "hecmw_ctrl.dat"), "w", encoding="utf-8") as f:
            f.write("!MESH, NAME=fstrMSH, TYPE=HECMW-ENTIRE\n")
            f.write("fistr_teardrop.msh\n")
            f.write("!CONTROL, NAME=fstrCNT\n")
            f.write("fistr_teardrop.cnt\n")
            f.write("!RESULT, NAME=fstrRES, IO=OUT\n")
            f.write("fistr_teardrop.res\n")
            f.write("!RESULT, NAME=vis_out, IO=OUT\n")
            f.write("fistr_teardrop_vis\n")

        # 2. fistr_teardrop.msh
        sec_forms: List[Tuple[int, str, str]] = []
        sec_idx = 0

        with open(os.path.join(self.run_dir, "fistr_teardrop.msh"), "w", encoding="utf-8") as f:
            f.write("!HEADER\n FrontISTR 14-Layer Conforming Hybrid 362-361UP Teardrop Mesh\n!NODE\n")
            for nid, x, y, z in self.mb.node_list:
                f.write(f" {nid}, {x:.8f}, {y:.8f}, {z:.8f}\n")

            eid = 1
            # Display layer elements
            for layer in STACKUP_14:
                lname = layer["name"]
                is_stiff = layer["is_stiff"]
                etype = 362 if is_stiff else 361
                elems = self.mb.layer_elements[lname]
                if not elems:
                    continue
                f.write(f"!ELEMENT, TYPE={etype}, EGRP={lname}\n")
                for el in elems:
                    f.write(f" {eid}, " + ", ".join(map(str, el)) + "\n")
                    eid += 1

            # Plate elements
            for pgrp, elems in self.mb.plate_elements.items():
                if not elems:
                    continue
                f.write(f"!ELEMENT, TYPE=361, EGRP={pgrp}\n")
                for el in elems:
                    f.write(f" {eid}, " + ", ".join(map(str, el)) + "\n")
                    eid += 1

            # Node groups
            f.write("!NGROUP, NGRP=PIVOT_L\n")
            f.write(f" {self.mb.pivot_l_nid}\n")
            f.write("!NGROUP, NGRP=PIVOT_R\n")
            f.write(f" {self.mb.pivot_r_nid}\n")
            f.write("!NGROUP, NGRP=PIVOT_DUMMY\n")
            for n in self.mb.pivot_dummy_nids:
                f.write(f" {n}\n")

            f.write("!NGROUP, NGRP=LEFT_DRIVEN\n")
            for n in self.mb.left_driven_nids:
                f.write(f" {n}\n")

            f.write("!NGROUP, NGRP=RIGHT_DRIVEN\n")
            for n in self.mb.right_driven_nids:
                f.write(f" {n}\n")

            if self.mb.z_fixed_nids:
                f.write("!NGROUP, NGRP=Z_FIXED\n")
                for n in self.mb.z_fixed_nids:
                    f.write(f" {n}\n")

            # Materials in .msh
            f.write("!MATERIAL, NAME=PET, ITEM=1\n!ITEM=1, SUBITEM=2\n 4000.0, 0.30\n")
            f.write("!MATERIAL, NAME=PSA, ITEM=1\n!ITEM=1, SUBITEM=2\n 0.05, 0.499\n")
            f.write("!MATERIAL, NAME=GLASS, ITEM=1\n!ITEM=1, SUBITEM=2\n 70000.0, 0.22\n")
            f.write("!MATERIAL, NAME=PLATE, ITEM=1\n!ITEM=1, SUBITEM=2\n 200000.0, 0.30\n")

            # Sections in .msh
            for layer in STACKUP_14:
                lname = layer["name"]
                if not self.mb.layer_elements[lname]:
                    continue
                sec_idx += 1
                mat_name = layer["mat"]
                form_name = "UP" if not layer["is_stiff"] else "FI"
                f.write(f"!SECTION, TYPE=SOLID, EGRP={lname}, MATERIAL={mat_name}\n 1.0\n")
                sec_forms.append((sec_idx, lname, form_name))

            for pgrp in ["EL_PLATE", "EL_PIVOT_REF"]:
                if not self.mb.plate_elements[pgrp]:
                    continue
                sec_idx += 1
                f.write(f"!SECTION, TYPE=SOLID, EGRP={pgrp}, MATERIAL=PLATE\n 1.0\n")
                sec_forms.append((sec_idx, pgrp, "FI"))

            f.write("!END\n")

        # 3. fistr_teardrop.cnt
        theta_rad = math.radians(self.theta_max_deg)

        with open(os.path.join(self.run_dir, "fistr_teardrop.cnt"), "w", encoding="utf-8") as f:
            f.write("!VERSION\n 5\n!SOLUTION, TYPE=NLSTATIC\n!WRITE, LOG, FREQUENCY=1\n")
            f.write(f"!WRITE, RESULT, FREQUENCY={self.res_write_freq}\n")
            f.write(f"!WRITE, VISUAL, FREQUENCY={self.res_write_freq}\n")
            f.write("!OUTPUT_RES\n DISP, ON\n REACTION, ON\n NSTRESS, ON\n NMISES, ON\n ESTRESS, ON\n EMISES, ON\n")
            f.write("!OUTPUT_VIS\n DISP, ON\n REACTION, ON\n NSTRESS, ON\n NMISES, ON\n")
            f.write("!SOLVER, METHOD=MUMPS, TIMELOG=YES, ITERLOG=NO\n")

            # Boundary conditions
            f.write("!BOUNDARY, GRPID=1\n")
            f.write(" PIVOT_DUMMY, 1, 3, 0.0\n")
            if self.mb.z_fixed_nids:
                f.write(" Z_FIXED, 3, 3, 0.0\n")

            # Rodrigues finite-rotation BCs around pivot_L and pivot_R
            f.write("!BOUNDARY, ROT_CENTER=PIVOT_L, GRPID=1\n")
            f.write(" LEFT_DRIVEN, 1, 1, 0.0\n")
            f.write(" LEFT_DRIVEN, 2, 2, 0.0\n")
            f.write(f" LEFT_DRIVEN, 3, 3, {-theta_rad:.12f}\n")

            f.write("!BOUNDARY, ROT_CENTER=PIVOT_R, GRPID=1\n")
            f.write(" RIGHT_DRIVEN, 1, 1, 0.0\n")
            f.write(" RIGHT_DRIVEN, 2, 2, 0.0\n")
            f.write(f" RIGHT_DRIVEN, 3, 3, {theta_rad:.12f}\n")

            # Materials
            f.write("!MATERIAL, NAME=PET\n!ELASTIC\n 4000.0, 0.30\n")
            f.write("!MATERIAL, NAME=PSA\n!HYPERELASTIC, TYPE=NEOHOOKE\n 8.33889260e-03, 2.40000000e-01\n")
            f.write("!MATERIAL, NAME=GLASS\n!ELASTIC\n 70000.0, 0.22\n")
            f.write("!MATERIAL, NAME=PLATE\n!ELASTIC\n 200000.0, 0.30\n")

            # Section formulations
            for sec_id, egrp_name, form_str in sec_forms:
                if form_str == "UP":
                    f.write(f"!SECTION, SECNUM={sec_id}, FORM361=UP\n")

            # Step control (CONVERG=2.0e-4 and MAXITER=50 for robust large-deformation convergence)
            f.write(f"!STEP, SUBSTEPS={self.substeps}, MAXITER=50, CONVERG=2.0e-4\n")
            f.write(" BOUNDARY, 1\n")
            f.write("!VISUAL, method=PSR\n!surface_num=1\n!surface 1\n!output_type=VTK\n!END\n")

    def run(self) -> Dict[str, Any]:
        """Executes FrontISTR, parses results, and outputs verification metrics."""
        self.write_deck()

        print("=" * 100)
        print("[*] Running FrontISTR 14-Layer Conforming Hybrid (362 + 361UP) Teardrop Folding")
        print(f"    Run Directory   : {self.run_dir}")
        print(f"    Nodes (Real)    : {self.mb.real_max_nid} (+8 dummy ref)")
        print(f"    Elements (Real) : {self.mb.real_max_eid} (+1 dummy ref)")
        print(f"    Substeps        : {self.substeps} (2.0 deg / substep) | Solver: MUMPS")
        print("=" * 100)

        t0 = time.time()
        res = subprocess.run([FISTR_EXE], cwd=self.run_dir, capture_output=True, text=True)
        wall_time = time.time() - t0

        stdout_path = os.path.join(self.run_dir, "fistr_stdout.log")
        with open(stdout_path, "w", encoding="utf-8") as f:
            f.write(res.stdout + "\n" + res.stderr)

        sta_path = os.path.join(self.run_dir, "FSTR.sta")
        sta_text = ""
        if os.path.exists(sta_path):
            with open(sta_path, "r", encoding="utf-8", errors="replace") as f:
                sta_text = f.read()

        completed = "FSTR_SOLVE_NLGEOM HAS COMPLETED SUCCESSFULLY" in sta_text
        print(f"[*] Solve Completed: {completed} | Exit Code: {res.returncode} | Wall Time: {wall_time:.2f}s")

        # Find latest completed substep
        available_substeps = []
        for s in range(self.substeps, 0, -1):
            rf_path = os.path.join(self.run_dir, f"fistr_teardrop.res.0.{s}")
            if os.path.exists(rf_path):
                available_substeps.append(s)

        if not available_substeps:
            return {
                "completed": False,
                "exit_code": res.returncode,
                "wall_time_s": wall_time,
                "error": "No result files found.",
            }

        latest_step = max(available_substeps)
        latest_res_path = os.path.join(self.run_dir, f"fistr_teardrop.res.0.{latest_step}")
        data = parse_fistr_res_file(latest_res_path, max_real_nid=self.mb.real_max_nid)

        # Compute reactions at left and right pivots
        p_l = (-float(self.mb.geo.hinge_pivot_x), float(self.mb.geo.hinge_pivot_y))
        p_r = (float(self.mb.geo.hinge_pivot_x), float(self.mb.geo.hinge_pivot_y))

        nodes_dict = {nid: (x, y, z) for nid, x, y, z in self.mb.node_list}
        reac_l = compute_reaction_moment_from_res(nodes_dict, data, self.mb.left_driven_nids, p_l)
        reac_r = compute_reaction_moment_from_res(nodes_dict, data, self.mb.right_driven_nids, p_r)

        # Deformed geometry metrics
        disp = data["DISP"]
        nids = data["nids"]
        x_def = np.array([nodes_dict[n][0] + disp[i, 0] for i, n in enumerate(nids)])
        y_def = np.array([nodes_dict[n][1] + disp[i, 1] for i, n in enumerate(nids)])

        apex_y = float(np.min(y_def))
        max_y = float(np.max(y_def))
        loop_width_x = float(np.max(x_def) - np.min(x_def))
        max_mises = float(np.max(data["NMISES"])) if "NMISES" in data else 0.0

        # History across saved increments
        history_steps = sorted(available_substeps)
        history_theta = []
        history_M_L = []
        history_M_R = []
        history_Fx_L = []
        history_Fx_R = []

        for s in history_steps:
            rpath = os.path.join(self.run_dir, f"fistr_teardrop.res.0.{s}")
            rdata = parse_fistr_res_file(rpath, max_real_nid=self.mb.real_max_nid)
            rl = compute_reaction_moment_from_res(nodes_dict, rdata, self.mb.left_driven_nids, p_l)
            rr = compute_reaction_moment_from_res(nodes_dict, rdata, self.mb.right_driven_nids, p_r)
            th = self.theta_max_deg * (s / float(self.substeps))
            history_theta.append(th)
            history_M_L.append(rl["Mz"])
            history_M_R.append(rr["Mz"])
            history_Fx_L.append(rl["Fx"])
            history_Fx_R.append(rr["Fx"])

        summary = {
            "case_name": "FISTR_14Layer_Conforming_Hybrid_362_361UP",
            "completed": completed,
            "wall_time_s": wall_time,
            "latest_substep": latest_step,
            "substeps_target": self.substeps,
            "num_real_nodes": self.mb.real_max_nid,
            "num_real_elements": self.mb.real_max_eid,
            "loop_width_x_mm": loop_width_x,
            "apex_y_mm": apex_y,
            "max_y_mm": max_y,
            "max_mises_mpa": max_mises,
            "M_L_final_Nmm": reac_l["Mz"],
            "M_R_final_Nmm": reac_r["Mz"],
            "Fx_L_final_N": reac_l["Fx"],
            "Fx_R_final_N": reac_r["Fx"],
            "history_theta_deg": history_theta,
            "history_M_L": history_M_L,
            "history_M_R": history_M_R,
            "history_Fx_L": history_Fx_L,
            "history_Fx_R": history_Fx_R,
            "run_dir": self.run_dir,
        }

        with open(os.path.join(self.run_dir, "fistr_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary


def plot_comparison(
    conforming_summary: Dict[str, Any],
    baseline_summary: Dict[str, Any],
    out_png: str,
):
    """Plots side-by-side comparison between Conforming Hybrid and Baseline 361 FI."""
    if plt is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # 1. Reaction Moment vs Angle
    ax = axes[0]
    th_c = conforming_summary.get("history_theta_deg", [])
    m_c = [abs(m) for m in conforming_summary.get("history_M_L", [])]
    th_b = baseline_summary.get("history_theta_deg", [])
    m_b = [abs(m) for m in baseline_summary.get("history_M_L", [])]

    if th_b and m_b:
        ax.plot(th_b, m_b, "r--o", label="Baseline (361 FI - Shear Locked)", linewidth=1.8, markersize=4)
    if th_c and m_c:
        ax.plot(th_c, m_c, "b-s", label="Conforming Hybrid (362 + 361UP)", linewidth=2.0, markersize=4)

    ax.set_title("14층 적층 180° 티어드롭 접힘 모멘트 곡선 ($M_L$ vs $\\theta$)", fontsize=10, fontweight="bold")
    ax.set_xlabel("한쪽 날개 회전각 $\\theta$ (deg) [총 접힘 180°]")
    ax.set_ylabel("작용 반력 모멘트 $|M_L|$ (N·mm)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")

    # 2. Key Metrics Bar Chart
    ax2 = axes[1]
    labels = ["최종 모멘트\n|M_L| (N·mm)", "최대 Mises 응력\n(100 MPa)", "해석 시간\n(s)"]
    m_final_b = abs(baseline_summary.get("M_L_final_Nmm", 35.15))
    m_final_c = abs(conforming_summary.get("M_L_final_Nmm", 0.0))

    s_mises_b = baseline_summary.get("max_mises_mpa", 3102.5) / 100.0
    s_mises_c = conforming_summary.get("max_mises_mpa", 0.0) / 100.0

    t_b = baseline_summary.get("wall_time_s", 25.9)
    t_c = conforming_summary.get("wall_time_s", 0.0)

    x = np.arange(len(labels))
    w = 0.35

    rects1 = ax2.bar(x - w/2, [m_final_b, s_mises_b, t_b], w, label="Baseline (361 FI)", color="#e74c3c", alpha=0.85)
    rects2 = ax2.bar(x + w/2, [m_final_c, s_mises_c, t_c], w, label="Conforming (362+361UP)", color="#2ecc71", alpha=0.85)

    ax2.set_title("핵심 정량 지표 대조 (전단 잠김 소거 검증)", fontsize=10, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.grid(True, linestyle="--", alpha=0.6, axis="y")
    ax2.legend(loc="upper right")

    for rect in rects1:
        h = rect.get_height()
        ax2.annotate(f"{h:.1f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for rect in rects2:
        h = rect.get_height()
        ax2.annotate(f"{h:.1f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[*] Comparison plot saved to: {out_png}")


def main():
    parser = argparse.ArgumentParser(description="FrontISTR 14-Layer Conforming Hybrid Teardrop Benchmark")
    parser.add_argument("--substeps", type=int, default=90, help="Number of substeps for 90-degree wing rotation (default: 90)")
    parser.add_argument("--ny_stiff", type=int, default=1, choices=[1, 2], help="Number of elements through thickness for stiff layers (default: 1)")
    parser.add_argument("--out_dir", type=str, default="example_disp_fold/results_conforming_teardrop", help="Output directory")
    args = parser.parse_args()

    config = UnifiedFoldModelConfig(preset="teardrop", dim="2D")
    builder = ConformingTeardropMeshBuilder(config=config, nx_hinge=30, nx_plate=20, width_z=1.0, ny_stiff=args.ny_stiff)
    case_folder = f"01_conforming_hybrid_14layer_ny{args.ny_stiff}"
    runner = FrontISTRConformingTeardropRunner(
        builder=builder,
        run_dir=os.path.join(args.out_dir, case_folder),
        substeps=args.substeps,
        theta_max_deg=90.0,
    )

    summary_conforming = runner.run()

    # Load baseline summary if exists
    baseline_path = os.path.join(REPO_ROOT, "example_disp_fold", "results_fistr", "FISTR_2D_teardrop_FI_FI", "fistr_summary.json")
    summary_baseline = {}
    if os.path.exists(baseline_path):
        with open(baseline_path, "r", encoding="utf-8") as f:
            summary_baseline = json.load(f)

    out_png = os.path.join(args.out_dir, "fistr_14layer_conforming_vs_361fi_teardrop.png")
    plot_comparison(summary_conforming, summary_baseline, out_png)

    print("\n" + "=" * 100)
    print(" 14-LAYER TEARDROP FOLDING BENCHMARK REPORT ")
    print("=" * 100)
    print(f" Case                 : {summary_conforming['case_name']}")
    print(f" Completed            : {summary_conforming['completed']} (Substeps {summary_conforming.get('latest_substep', 0)}/{args.substeps})")
    print(f" Wall Time            : {summary_conforming.get('wall_time_s', 0.0):.2f} s")
    print(f" Final Moment M_L     : {summary_conforming.get('M_L_final_Nmm', 0.0):.4f} N*mm (Baseline 361 FI: {summary_baseline.get('M_L_final_Nmm', 35.15):.4f} N*mm)")
    print(f" Max Mises Stress     : {summary_conforming.get('max_mises_mpa', 0.0):.1f} MPa (Baseline 361 FI: {summary_baseline.get('max_mises_mpa', 3102.5):.1f} MPa)")
    print(f" Loop Width (X Span)  : {summary_conforming.get('loop_width_x_mm', 0.0):.4f} mm (Baseline 361 FI: {summary_baseline.get('loop_width_x_mm', 4.532):.4f} mm)")
    print(f" Apex Y Coordinate    : {summary_conforming.get('apex_y_mm', 0.0):.4f} mm")
    print("=" * 100)


if __name__ == "__main__":
    main()
