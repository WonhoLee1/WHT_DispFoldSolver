"""Abaqus .inp keyword parser — transforms tokenized blocks into AbaqusModel.

Each keyword handler parses the data lines of an AbaqusKeywordBlock
and populates the corresponding fields of the AbaqusModel.

Supports hierarchical material definitions via *MATERIAL:
    *MATERIAL, NAME=PET
    *ELASTIC
    4000.0, 0.3
    *PLASTIC
    80.0, 0.0
    ...

Material properties accumulate into a single AbaqusMaterial entry
when *MATERIAL scoping is active.
"""

import warnings
from typing import List, Optional, Set

from dispsolver.io.abaqus_model import (
    AbaqusModel, AbaqusNode, AbaqusElement, AbaqusMaterial, AbaqusSection,
    AbaqusBoundary, AbaqusLoad, AbaqusStep, AbaqusMpc, AbaqusTie,
    AbaqusAmplitude, AbaqusDload, AbaqusSurface, AbaqusContactPair,
    AbaqusContactProperty, AbaqusPart, AbaqusInstance, AbaqusRigidBody,
)
from dispsolver.io.abaqus_lexer import AbaqusKeywordBlock

# Keywords that are material sub-properties (continue material context)
_MATERIAL_PROPS: Set[str] = frozenset({
    "ELASTIC", "PLASTIC", "HYPERELASTIC", "VISCOELASTIC", "TRS", "DENSITY",
})


def _parse_data_lines(data_lines: List[str], ncols: int, dtype=float) -> List[List[float]]:
    """Parse comma/whitespace-separated data lines into a table of floats."""
    rows = []
    for line in data_lines:
        if not line.strip():
            continue
        # Split by comma or whitespace
        parts = [p for p in line.replace(",", " ").split() if p.strip()]
        if not parts:
            continue
        row = []
        for p in parts:
            try:
                row.append(dtype(p))
            except ValueError:
                row.append(p)  # keep as string
        rows.append(row)
    return rows


class AbaqusParser:
    """Parses a sequence of AbaqusKeywordBlock into an AbaqusModel."""

    def __init__(self):
        self.model = AbaqusModel()
        self._current_part: Optional[str] = None  # name of current part being defined
        self._current_instance: Optional[AbaqusInstance] = None
        self._in_assembly = False
        self._current_step: Optional[AbaqusStep] = None
        self._current_material: Optional[str] = None  # active *MATERIAL name

        # Build handler dispatch table
        self._handlers = {
            "NODE": self._parse_node,
            "ELEMENT": self._parse_element,
            "NSET": self._parse_nset,
            "ELSET": self._parse_elset,
            "MATERIAL": self._parse_material,
            "ELASTIC": self._parse_elastic,
            "PLASTIC": self._parse_plastic,
            "HYPERELASTIC": self._parse_hyperelastic,
            "VISCOELASTIC": self._parse_viscoelastic,
            "TRS": self._parse_trs,
            "DENSITY": self._parse_density,
            "SOLID SECTION": self._parse_section,
            "BOUNDARY": self._parse_boundary,
            "CLOAD": self._parse_cload,
            "DLOAD": self._parse_dload,
            "DSLOAD": self._parse_dload,
            "AMPLITUDE": self._parse_amplitude,
            "STEP": self._parse_step,
            "DYNAMIC": self._parse_dynamic,
            "STATIC": self._parse_static,
            "END STEP": self._parse_end_step,
            "MPC": self._parse_mpc,
            "TIE": self._parse_tie,
            "INITIAL CONDITIONS": self._parse_initial_conditions,
            "RIGID BODY": self._parse_rigid_body,
            "SURFACE": self._parse_surface,
            "CONTACT PAIR": self._parse_contact_pair,
            "SURFACE INTERACTION": self._parse_contact_property,
            "PART": self._parse_part,
            "END PART": self._parse_end_part,
            "ASSEMBLY": self._parse_assembly,
            "END ASSEMBLY": self._parse_end_assembly,
            "INSTANCE": self._parse_instance,
            "END INSTANCE": self._parse_end_instance,
        }

    def parse(self, blocks: List[AbaqusKeywordBlock]) -> AbaqusModel:
        """Parse a list of keyword blocks into an AbaqusModel."""
        self.model = AbaqusModel()
        for block in blocks:
            handler = self._handlers.get(block.keyword)
            if handler:
                handler(block)
            else:
                warnings.warn(f"Unsupported keyword: *{block.keyword} — skipping")
            # Clear material context on non-material-property keywords
            if block.keyword != "MATERIAL" and block.keyword not in _MATERIAL_PROPS:
                self._current_material = None
        return self.model

    # ----- Node/Element -----

    def _parse_node(self, block: AbaqusKeywordBlock):
        nset = block.params.get("nset", "")
        rows = _parse_data_lines(block.data_lines, 3)
        for r in rows:
            node = AbaqusNode(id=int(r[0]), x=float(r[1]), y=float(r[2]))
            if self._current_part:
                self.model.parts[self._current_part].nodes.append(node)
            else:
                self.model.nodes.append(node)

    def _parse_element(self, block: AbaqusKeywordBlock):
        etype = block.params.get("type", "CPE4").upper()
        if etype in ("C3D8", "C3D20", "C3D8R", "C3D20R"):
            raise NotImplementedError(f"3D elements ({etype}) are not supported — 2D only")
        elset = block.params.get("elset", "")
        rows = _parse_data_lines(block.data_lines, 5, dtype=int)
        elem_ids = []
        for r in rows:
            elem = AbaqusElement(eid=int(r[0]), etype=etype, node_ids=r[1:])
            elem_ids.append(elem.eid)
            if self._current_part:
                self.model.parts[self._current_part].elements.append(elem)
            else:
                self.model.elements.append(elem)
        # Register elementset from ELSET= parameter, if specified
        if elset:
            target_elsets = self.model.parts[self._current_part].elsets if self._current_part else self.model.elsets
            if elset in target_elsets:
                target_elsets[elset].update(elem_ids)
            else:
                target_elsets[elset] = set(elem_ids)

    # ----- Set definitions -----

    def _parse_nset(self, block: AbaqusKeywordBlock):
        name = block.params.get("nset", "")
        generate = block.params.get("generate", "").lower() == "yes"
        if generate and block.data_lines:
            parts = block.data_lines[0].replace(",", " ").split()
            if len(parts) >= 3:
                start, end, step = int(parts[0]), int(parts[1]), int(parts[2])
                ids = list(range(start, end + 1, step))
            else:
                return
        else:
            ids = [int(p.replace(",", "")) for p in
                   " ".join(block.data_lines).split() if p.strip().replace(",", "").isdigit()]
        target = self.model.parts[self._current_part].nsets if self._current_part else self.model.nsets
        target[name] = ids

    def _parse_elset(self, block: AbaqusKeywordBlock):
        name = block.params.get("elset", "")
        ids = [int(p.replace(",", "")) for p in " ".join(block.data_lines).split()
               if p.strip().replace(",", "").isdigit()]
        target = self.model.parts[self._current_part].elsets if self._current_part else self.model.elsets
        target[name] = ids

    # ----- Material -----

    def _parse_material(self, block: AbaqusKeywordBlock):
        """*MATERIAL — start a material scope; subsequent property keywords
        accumulate into this material entry."""
        name = block.params.get("name", f"MAT-{len(self.model.materials)}")
        self._current_material = name
        if name not in self.model.materials:
            self.model.materials[name] = AbaqusMaterial(name=name)

    def _parse_elastic(self, block: AbaqusKeywordBlock):
        rows = _parse_data_lines(block.data_lines, 2)
        if not rows:
            return
        if self._current_material:
            mat = self.model.materials[self._current_material]
            mat.elastic = (rows[0][0], rows[0][1])
        else:
            name = block.params.get("name", f"MAT-{len(self.model.materials)}")
            mat = AbaqusMaterial(name=name, mat_type="ELASTIC",
                                 params={"E": rows[0][0], "nu": rows[0][1]})
            target = self.model.parts[self._current_part].materials if self._current_part else self.model.materials
            target[name] = mat

    def _parse_plastic(self, block: AbaqusKeywordBlock):
        rows = _parse_data_lines(block.data_lines, 2)
        harden_table = [(r[0], r[1]) for r in rows]
        if self._current_material:
            mat = self.model.materials[self._current_material]
            mat.plastic_table = harden_table
        else:
            mat_name = block.params.get("name", f"MAT-PLASTIC-{len(self.model.materials)}")
            mat = AbaqusMaterial(name=mat_name, mat_type="PLASTIC",
                                 params={"hardening_table": harden_table})
            target = self.model.parts[self._current_part].materials if self._current_part else self.model.materials
            target[mat_name] = mat

    def _parse_hyperelastic(self, block: AbaqusKeywordBlock):
        model_type = block.params.get("neo hooke", block.params.get("type", "NEO HOOKE")).upper()
        rows = _parse_data_lines(block.data_lines, 2)
        if not rows:
            return
        hyper_data: dict = {"model": model_type}
        if model_type in ("NEO HOOKE",):
            hyper_data["C10"] = rows[0][0]
            hyper_data["D1"] = rows[0][1] if len(rows[0]) >= 2 else 0.0
        elif model_type == "YEOH":
            hyper_data["C10"] = rows[0][0]
            hyper_data["C20"] = rows[0][1] if len(rows[0]) >= 2 else 0.0
            hyper_data["C30"] = rows[0][2] if len(rows[0]) >= 3 else 0.0
            hyper_data["D1"] = rows[0][3] if len(rows[0]) >= 4 else 0.0
        elif model_type == "ARRUDA-BOYCE":
            hyper_data["mu"] = rows[0][0]
            hyper_data["lambda_m"] = rows[0][1] if len(rows[0]) >= 2 else 3.0
            hyper_data["D"] = rows[0][2] if len(rows[0]) >= 3 else 0.0
        if self._current_material:
            mat = self.model.materials[self._current_material]
            mat.hyperelastic = hyper_data
        else:
            mat_name = block.params.get("name", f"MAT-HYPER-{len(self.model.materials)}")
            mat = AbaqusMaterial(name=mat_name, mat_type=f"HYPERELASTIC_{model_type.replace(' ', '_')}",
                                 params=hyper_data)
            target = self.model.parts[self._current_part].materials if self._current_part else self.model.materials
            target[mat_name] = mat

    def _parse_viscoelastic(self, block: AbaqusKeywordBlock):
        rows = _parse_data_lines(block.data_lines, 3)
        prony_data = [(r[0], r[1], r[2]) for r in rows]
        if self._current_material:
            mat = self.model.materials[self._current_material]
            mat.viscoelastic_prony = prony_data
        else:
            mat_name = block.params.get("name", f"MAT-VISCO-{len(self.model.materials)}")
            mat = AbaqusMaterial(name=mat_name, mat_type="VISCOELASTIC",
                                 params={"prony": prony_data})
            target = self.model.parts[self._current_part].materials if self._current_part else self.model.materials
            target[mat_name] = mat

    def _parse_trs(self, block: AbaqusKeywordBlock):
        """Time-temperature superposition (WLF or user)."""
        trs_def = block.params.get("definition", "").upper()
        params = {}
        if trs_def == "WLF" and block.data_lines:
            parts = block.data_lines[0].replace(",", " ").split()
            if len(parts) >= 3:
                params = {"T_ref": float(parts[0]), "C1": float(parts[1]), "C2": float(parts[2])}
        params["definition"] = trs_def if trs_def else "WLF"
        if self._current_material:
            mat = self.model.materials[self._current_material]
            mat.trs = params
        else:
            mat_name = self._current_part or "GLOBAL"
            mat = AbaqusMaterial(name=f"TRS_{mat_name}", mat_type="TRS", params=params)
            self.model.materials[f"TRS_{mat_name}"] = mat

    def _parse_density(self, block: AbaqusKeywordBlock):
        rows = _parse_data_lines(block.data_lines, 1)
        if rows:
            rho = rows[0][0]
            if self._current_material:
                mat = self.model.materials[self._current_material]
                mat.density = rho
            self.model.density = rho

    # ----- Section -----

    def _parse_section(self, block: AbaqusKeywordBlock):
        elset = block.params.get("elset", "")
        material = block.params.get("material", "")
        rows = _parse_data_lines(block.data_lines, 1) if block.data_lines else []
        thickness = rows[0][0] if rows else 1.0
        sec = AbaqusSection(name=f"SECT_{elset}", material=material, elset=elset, thickness=thickness)
        target = self.model.parts[self._current_part].sections if self._current_part else self.model.sections
        target[sec.name] = sec

    # ----- Boundary / Load -----

    def _parse_boundary(self, block: AbaqusKeywordBlock):
        rows = _parse_data_lines(block.data_lines, 4) if block.data_lines else []
        for r in rows:
            bc = AbaqusBoundary(nset=block.params.get("nset", ""), dof1=int(r[0]), dof2=int(r[1]), value=r[2])
            self.model.boundaries.append(bc)

    def _parse_cload(self, block: AbaqusKeywordBlock):
        name = block.params.get("name", f"LOAD-{len(self.model.loads)}")
        rows = _parse_data_lines(block.data_lines, 2)
        for r in rows:
            load = AbaqusLoad(name=name, nset_or_elset=block.params.get("nset", block.params.get("elset", "")),
                              magnitude=r[1])
            self.model.loads.append(load)

    def _parse_dload(self, block: AbaqusKeywordBlock):
        rows = _parse_data_lines(block.data_lines, 3)
        for r in rows:
            face_str = str(r[1]) if not isinstance(r[1], (int, float)) else str(int(r[1]))
            dload = AbaqusDload(elset=str(r[0]), face=face_str, magnitude=float(r[2]))
            self.model.dloads.append(dload)

    def _parse_amplitude(self, block: AbaqusKeywordBlock):
        name = block.params.get("name", "")
        time_type = block.params.get("time", "STEP").upper()
        rows = _parse_data_lines(block.data_lines, 2)
        data = [(r[0], r[1]) for r in rows]
        amp = AbaqusAmplitude(name=name, time_type=time_type, data=data)
        self.model.amplitudes[name] = amp

    # ----- Step -----

    def _parse_step(self, block: AbaqusKeywordBlock):
        name = block.params.get("name", f"Step-{len(self.model.steps)}")
        self._current_step = AbaqusStep(name=name, procedure="")

    def _parse_dynamic(self, block: AbaqusKeywordBlock):
        if self._current_step is None:
            return
        self._current_step.procedure = "DYNAMIC"
        rows = _parse_data_lines(block.data_lines, 4)
        if rows:
            self._current_step.bounds = {
                "dt_init": rows[0][0], "t_total": rows[0][1],
                "dt_min": rows[0][2], "dt_max": rows[0][3] if len(rows[0]) >= 4 else rows[0][0]
            }

    def _parse_static(self, block: AbaqusKeywordBlock):
        if self._current_step is None:
            return
        self._current_step.procedure = "STATIC"
        rows = _parse_data_lines(block.data_lines, 4)
        if rows:
            self._current_step.bounds = {
                "dt_init": rows[0][0], "t_total": rows[0][1],
                "dt_min": rows[0][2] if len(rows[0]) >= 3 else rows[0][0],
                "dt_max": rows[0][3] if len(rows[0]) >= 4 else rows[0][0]
            }

    def _parse_end_step(self, block: AbaqusKeywordBlock):
        if self._current_step is not None:
            self.model.steps.append(self._current_step)
            self._current_step = None

    # ----- Constraint -----

    def _parse_mpc(self, block: AbaqusKeywordBlock):
        mpc_type = block.params.get("type", "BEAM").upper()
        rows = _parse_data_lines(block.data_lines, 1, dtype=int)
        nodes = [r[0] for r in rows]
        mpc = AbaqusMpc(mpc_type=mpc_type, nodes=nodes)
        self.model.mpcs.append(mpc)

    def _parse_tie(self, block: AbaqusKeywordBlock):
        slave = block.params.get("slave", "")
        master = block.params.get("master", "")
        rows = _parse_data_lines(block.data_lines, 1)
        pos_tol = rows[0][0] if rows else 0.0
        tie = AbaqusTie(slave=slave, master=master, position_tolerance=pos_tol)
        self.model.ties.append(tie)

    def _parse_rigid_body(self, block: AbaqusKeywordBlock):
        elset = block.params.get("elset", "")
        nset = block.params.get("nset", "")
        ref_node = int(block.params.get("ref node", "0"))
        if ref_node == 0:
            return
        # Resolve node IDs from the set
        if nset:
            node_ids = self.model.nsets.get(nset, [])
        elif elset:
            # Elementset: collect all nodes from the elements in that set
            elem_ids = self.model.elsets.get(elset, set())
            node_ids = []
            seen = set()
            for e in self.model.elements:
                if e.eid in elem_ids:
                    for nid in e.node_ids:
                        if nid not in seen:
                            seen.add(nid)
                            node_ids.append(nid)
        else:
            return
        rb = AbaqusRigidBody(ref_node=ref_node, node_ids=node_ids, name=nset or elset)
        self.model.rigid_bodies.append(rb)

    def _parse_initial_conditions(self, block: AbaqusKeywordBlock):
        ic_type = block.params.get("type", "TEMPERATURE").upper()
        rows = _parse_data_lines(block.data_lines, 2)
        params = {"type": ic_type, "data": rows}
        mat = AbaqusMaterial(name=f"IC_{ic_type}", mat_type="INITIAL_CONDITIONS", params=params)
        self.model.materials[f"IC_{ic_type}"] = mat

    # ----- Contact -----

    def _parse_surface(self, block: AbaqusKeywordBlock):
        name = block.params.get("name", "")
        surf_type = block.params.get("type", "ELEMENT").upper()
        definitions = []
        for line in block.data_lines:
            parts = line.replace(",", " ").split()
            if len(parts) >= 2:
                definitions.append((parts[0], parts[1].upper()))
        surface = AbaqusSurface(name=name, surface_type=surf_type, definitions=definitions)
        self.model.surfaces[name] = surface

    def _parse_contact_pair(self, block: AbaqusKeywordBlock):
        name = block.params.get("name", f"CP-{len(self.model.contact_pairs)}")
        interaction = block.params.get("interaction", "")
        if block.data_lines:
            parts = block.data_lines[0].replace(",", " ").split()
            if len(parts) >= 2:
                slave, master = parts[0], parts[1]
                pair = AbaqusContactPair(name=name, slave_surface=slave,
                                         master_surface=master, interaction=interaction)
                self.model.contact_pairs.append(pair)

    def _parse_contact_property(self, block: AbaqusKeywordBlock):
        name = block.params.get("name", f"CPROP-{len(self.model.contact_properties)}")
        rows = _parse_data_lines(block.data_lines, 2)
        eps = rows[0][0] if rows else 1e6
        mu = rows[0][1] if len(rows) > 0 and len(rows[0]) > 1 else 0.0
        prop = AbaqusContactProperty(name=name, penalty_stiffness=eps, friction_coeff=mu)
        self.model.contact_properties[name] = prop

    # ----- Part / Assembly -----

    def _parse_part(self, block: AbaqusKeywordBlock):
        name = block.params.get("name", f"PART-{len(self.model.parts)}")
        self._current_part = name
        self.model.parts[name] = AbaqusPart(name=name)

    def _parse_end_part(self, block: AbaqusKeywordBlock):
        self._current_part = None

    def _parse_assembly(self, block: AbaqusKeywordBlock):
        self._in_assembly = True

    def _parse_end_assembly(self, block: AbaqusKeywordBlock):
        self._in_assembly = False

    def _parse_instance(self, block: AbaqusKeywordBlock):
        part_name = block.params.get("part", "")
        name = block.params.get("name", part_name)
        trans = None
        rot = None
        if "translation" in block.params:
            t_parts = block.params["translation"].split()
            if len(t_parts) >= 2:
                trans = (float(t_parts[0]), float(t_parts[1]))
        data_lines_text = " ".join(block.data_lines)
        if "*TRANSLATION" in [l.strip().upper() for l in block.data_lines]:
            idx = next((i for i, l in enumerate(block.data_lines) if "TRANSLATION" in l.upper()), None)
            if idx is not None and idx + 1 < len(block.data_lines):
                t_parts = block.data_lines[idx + 1].replace(",", " ").split()
                if len(t_parts) >= 2:
                    trans = (float(t_parts[0]), float(t_parts[1]))
        instance = AbaqusInstance(part_name=part_name, instance_name=name,
                                  translation=trans, rotation=rot)
        self.model.instances.append(instance)

    def _parse_end_instance(self, block: AbaqusKeywordBlock):
        pass
