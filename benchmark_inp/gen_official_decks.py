"""Generate official-benchmark-equivalent .inp decks for reader testing.

The SIMULIA doc pages in benchmark_element/reference_abaqus_docs/ carry no
inp text, so these continuum analogues are built from the documented
problem descriptions. Reader coverage is the goal, not benchmark numbers.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def write(name, text):
    path = os.path.join(HERE, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def cook_2d(nx=4, ny=4):
    """Tapered panel: left 44 tall, right 16 tall starting 16 up, 48 apart."""
    xs = [48.0 * i / nx for i in range(nx + 1)]
    lines = ["*HEADING", "Cook membrane 2D coarse analogue (CPE4H, distributing coupling)",
             "*NODE"]
    nid = {}
    n = 0
    for j in range(ny + 1):
        for i in range(nx + 1):
            n += 1
            x = xs[i]
            y_lo = 16.0 * (x / 48.0)
            y_hi = 44.0 + (32.0 - 44.0) * (x / 48.0)
            y = y_lo + (y_hi - y_lo) * (j / ny)
            nid[(i, j)] = n
            lines.append(f"{n}, {x:.6f}, {y:.6f}")
    lines.append("*ELEMENT, TYPE=CPE4H, ELSET=PANEL")
    e = 0
    for j in range(ny):
        for i in range(nx):
            e += 1
            lines.append(f"{e}, {nid[(i, j)]}, {nid[(i + 1, j)]}, "
                         f"{nid[(i + 1, j + 1)]}, {nid[(i, j + 1)]}")
    left = sorted(nid[(0, j)] for j in range(ny + 1))
    right = sorted(nid[(nx, j)] for j in range(ny + 1))
    lines += ["*NSET, NSET=CLAMP",
              ", ".join(map(str, left)),
              "*NSET, NSET=TIP_EDGE",
              ", ".join(map(str, right)),
              "*MATERIAL, NAME=RUBBER",
              "*HYPERELASTIC, NEO HOOKE",
              "0.4, 2.5e-4",
              "*SOLID SECTION, ELSET=PANEL, MATERIAL=RUBBER",
              "*COUPLING, CONSTRAINT NAME=CR, REF NODE=1000, SURFACE=TIP_EDGE",
              "*DISTRIBUTING",
              "*STEP, NAME=SHEAR, NLGEOM=YES",
              "*STATIC",
              "0.1, 1.0",
              "*BOUNDARY",
              "CLAMP, 1, 2, 0.0",
              "*CLOAD",
              "1000, 2, -1.0",
              "*END STEP"]
    return "\n".join(lines) + "\n"


def cantilever_2d(nx=10, ny=1):
    lines = ["*HEADING", "Cantilever continuum analogue (CPE4, tip load)",
             "*NODE"]
    nid = {}
    n = 0
    for j in range(ny + 1):
        for i in range(nx + 1):
            n += 1
            nid[(i, j)] = n
            lines.append(f"{n}, {float(i):.6f}, {float(j):.6f}")
    lines.append("*ELEMENT, TYPE=CPE4, ELSET=BEAM")
    e = 0
    for j in range(ny):
        for i in range(nx):
            e += 1
            lines.append(f"{e}, {nid[(i, j)]}, {nid[(i + 1, j)]}, "
                         f"{nid[(i + 1, j + 1)]}, {nid[(i, j + 1)]}")
    root = sorted(nid[(0, j)] for j in range(ny + 1))
    tip = sorted(nid[(nx, j)] for j in range(ny + 1))
    lines += ["*NSET, NSET=ROOT",
              ", ".join(map(str, root)),
              "*NSET, NSET=TIP",
              ", ".join(map(str, tip)),
              "*MATERIAL, NAME=STEEL",
              "*ELASTIC",
              "100.0, 0.3",
              "*DENSITY",
              "1.0",
              "*SOLID SECTION, ELSET=BEAM, MATERIAL=STEEL",
              "*STEP, NAME=BEND, NLGEOM=YES",
              "*STATIC",
              "0.1, 1.0",
              "*BOUNDARY",
              "ROOT, 1, 2, 0.0",
              "*CLOAD",
              "TIP, 2, -1.0",
              "*END STEP"]
    return "\n".join(lines) + "\n"


def cube_c3d8():
    return """*HEADING
Single C3D8 cube sanity deck
*NODE
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 1.0, 1.0, 0.0
4, 0.0, 1.0, 0.0
5, 0.0, 0.0, 1.0
6, 1.0, 0.0, 1.0
7, 1.0, 1.0, 1.0
8, 0.0, 1.0, 1.0
*ELEMENT, TYPE=C3D8, ELSET=CUBE
1, 1, 2, 3, 4, 5, 6, 7, 8
*NSET, NSET=BASE
1, 4, 5, 8
*MATERIAL, NAME=M
*ELASTIC
1000.0, 0.0
*SOLID SECTION, ELSET=CUBE, MATERIAL=M
*STEP, NAME=PULL, NLGEOM=NO
*STATIC
1.0, 1.0
*BOUNDARY
BASE, 1, 3, 0.0
*END STEP
"""


def patch_2d(nx=2, ny=2):
    """2D patch: unit square, linear field u=x/10, v=y/20 on all edges."""
    lines = ["*HEADING", "Patch analogue (CPE4, linear boundary field)",
             "*NODE"]
    nid = {}
    n = 0
    for j in range(ny + 1):
        for i in range(nx + 1):
            n += 1
            nid[(i, j)] = n
            lines.append(f"{n}, {i / nx:.6f}, {j / ny:.6f}")
    lines.append("*ELEMENT, TYPE=CPE4, ELSET=PATCH")
    e = 0
    for j in range(ny):
        for i in range(nx):
            e += 1
            lines.append(f"{e}, {nid[(i, j)]}, {nid[(i + 1, j)]}, "
                         f"{nid[(i + 1, j + 1)]}, {nid[(i, j + 1)]}")
    edge = sorted(set(
        [nid[(i, 0)] for i in range(nx + 1)] + [nid[(i, ny)] for i in range(nx + 1)]
        + [nid[(0, j)] for j in range(ny + 1)] + [nid[(nx, j)] for j in range(ny + 1)]))
    lines += ["*NSET, NSET=EDGE",
              ", ".join(map(str, edge)),
              "*MATERIAL, NAME=M",
              "*ELASTIC",
              "1000.0, 0.25",
              "*SOLID SECTION, ELSET=PATCH, MATERIAL=M",
              "*STEP, NAME=PATCHLOAD, NLGEOM=NO",
              "*STATIC",
              "1.0, 1.0",
              "*BOUNDARY"]
    for n_ in edge:
        x = (n_ - 1) % (nx + 1) / nx
        y = (n_ - 1) // (nx + 1) / ny
        lines.append(f"{n_}, 1, 1, {x / 10:.6f}")
        lines.append(f"{n_}, 2, 2, {y / 20:.6f}")
    lines.append("*END STEP")
    return "\n".join(lines) + "\n"


def cantilever_moment_2d(nx=10, ny=1):
    """Cantilever with end couple: opposite CLOADs at tip corners."""
    base = cantilever_2d(nx=nx, ny=ny)
    base = base.replace("*CLOAD\nTIP, 2, -1.0",
                        "*CLOAD\nTIP, 2, -1.0\n*CLOAD\nTIP, 1, 0.5")
    base = base.replace("Cantilever continuum analogue (CPE4, tip load)",
                        "Cantilever moment analogue (CPE4, tip couple)")
    return base


def thick_cylinder_2d(nt=8, nr=2, r_in=1.0, r_out=2.0):
    """Quarter ring under internal pressure (CPE4)."""
    import math
    lines = ["*HEADING", "Thick cylinder quarter analogue (CPE4, internal pressure)",
             "*NODE"]
    nid = {}
    n = 0
    for j in range(nr + 1):
        r = r_in + (r_out - r_in) * j / nr
        for i in range(nt + 1):
            n += 1
            a = (math.pi / 2) * i / nt
            nid[(i, j)] = n
            lines.append(f"{n}, {r * math.cos(a):.6f}, {r * math.sin(a):.6f}")
    lines.append("*ELEMENT, TYPE=CPE4, ELSET=RING")
    e = 0
    for j in range(nr):
        for i in range(nt):
            e += 1
            lines.append(f"{e}, {nid[(i, j)]}, {nid[(i + 1, j)]}, "
                         f"{nid[(i + 1, j + 1)]}, {nid[(i, j + 1)]}")
    sym_x = sorted(nid[(i, 0)] for i in range(nt + 1))
    sym_y = sorted(nid[(0, j)] for j in range(nr + 1))
    lines += ["*NSET, NSET=SYMX",
              ", ".join(map(str, sym_x)),
              "*NSET, NSET=SYMY",
              ", ".join(map(str, sym_y)),
              "*ELSET, ELSET=INNER",
              ", ".join(str(e + 1) for e in range(nt)),
              "*MATERIAL, NAME=M",
              "*ELASTIC",
              "1000.0, 0.3",
              "*SOLID SECTION, ELSET=RING, MATERIAL=M",
              "*STEP, NAME=PRESS, NLGEOM=NO",
              "*STATIC",
              "0.1, 1.0",
              "*BOUNDARY",
              "SYMX, 2, 2, 0.0",
              "SYMY, 1, 1, 0.0",
              "*DLOAD",
              "INNER, P1, 10.0",
              "*END STEP"]
    return "\n".join(lines) + "\n"


def tensile_bar_3d(nx=2, ny=1, nz=1):
    """Uniaxial tension bar (C3D8)."""
    lines = ["*HEADING", "Tensile bar analogue (C3D8)",
             "*NODE"]
    nid = {}
    n = 0
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                n += 1
                nid[(i, j, k)] = n
                lines.append(f"{n}, {float(i):.6f}, {float(j):.6f}, {float(k):.6f}")
    lines.append("*ELEMENT, TYPE=C3D8, ELSET=BAR")
    e = 0
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                e += 1
                lines.append(f"{e}, {nid[(i, j, k)]}, {nid[(i + 1, j, k)]}, "
                             f"{nid[(i + 1, j + 1, k)]}, {nid[(i, j + 1, k)]}, "
                             f"{nid[(i, j, k + 1)]}, {nid[(i + 1, j, k + 1)]}, "
                             f"{nid[(i + 1, j + 1, k + 1)]}, {nid[(i, j + 1, k + 1)]}")
    fix = sorted(nid[(0, j, k)] for j in range(ny + 1) for k in range(nz + 1))
    pull = sorted(nid[(nx, j, k)] for j in range(ny + 1) for k in range(nz + 1))
    lines += ["*NSET, NSET=FIX",
              ", ".join(map(str, fix)),
              "*NSET, NSET=PULL",
              ", ".join(map(str, pull)),
              "*MATERIAL, NAME=M",
              "*ELASTIC",
              "1000.0, 0.0",
              "*SOLID SECTION, ELSET=BAR, MATERIAL=M",
              "*STEP, NAME=TENSION, NLGEOM=NO",
              "*STATIC",
              "1.0, 1.0",
              "*BOUNDARY",
              "FIX, 1, 3, 0.0",
              "PULL, 1, 1, 0.1",
              "*END STEP"]
    return "\n".join(lines) + "\n"


def shear_2d():
    """Single-element shear (CPE4R)."""
    return """*HEADING
Single-element shear analogue (CPE4R)
*NODE
1, 0.0, 0.0
2, 1.0, 0.0
3, 1.0, 1.0
4, 0.0, 1.0
*ELEMENT, TYPE=CPE4R, ELSET=ONE
1, 1, 2, 3, 4
*NSET, NSET=BOT
1, 2
*NSET, NSET=TOP
3, 4
*MATERIAL, NAME=M
*ELASTIC
1000.0, 0.3
*SOLID SECTION, ELSET=ONE, MATERIAL=M
*STEP, NAME=SHEAR, NLGEOM=NO
*STATIC
1.0, 1.0
*BOUNDARY
BOT, 1, 2, 0.0
TOP, 1, 1, 0.1
*END STEP
"""


def cook_3d(nx=2, ny=2, nz=1):
    """Cook panel extruded 5 thick (C3D8)."""
    lines = ["*HEADING", "Cook 3D coarse analogue (C3D8)",
             "*NODE"]
    nid = {}
    n = 0
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                n += 1
                x = 48.0 * i / nx
                y_lo = 16.0 * (x / 48.0)
                y_hi = 44.0 + (32.0 - 44.0) * (x / 48.0)
                y = y_lo + (y_hi - y_lo) * (j / ny)
                nid[(i, j, k)] = n
                lines.append(f"{n}, {x:.6f}, {y:.6f}, {5.0 * k / nz:.6f}")
    lines.append("*ELEMENT, TYPE=C3D8, ELSET=PANEL")
    e = 0
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                e += 1
                lines.append(f"{e}, {nid[(i, j, k)]}, {nid[(i + 1, j, k)]}, "
                             f"{nid[(i + 1, j + 1, k)]}, {nid[(i, j + 1, k)]}, "
                             f"{nid[(i, j, k + 1)]}, {nid[(i + 1, j, k + 1)]}, "
                             f"{nid[(i + 1, j + 1, k + 1)]}, {nid[(i, j + 1, k + 1)]}")
    clamp = sorted(nid[(0, j, k)] for j in range(ny + 1) for k in range(nz + 1))
    lines += ["*NSET, NSET=CLAMP",
              ", ".join(map(str, clamp)),
              "*MATERIAL, NAME=M",
              "*ELASTIC",
              "1000.0, 0.3",
              "*SOLID SECTION, ELSET=PANEL, MATERIAL=M",
              "*STEP, NAME=S, NLGEOM=NO",
              "*STATIC",
              "0.1, 1.0",
              "*BOUNDARY",
              "CLAMP, 1, 3, 0.0",
              "*END STEP"]
    return "\n".join(lines) + "\n"


def contact_2block_2d():
    """Two-block contact pair with surfaces (parse coverage)."""
    return """*HEADING
Two-block contact analogue (CPE4, CONTACT PAIR)
*PART, NAME=TOP
*NODE
1, 0.0, 1.0
2, 1.0, 1.0
3, 1.0, 2.0
4, 0.0, 2.0
*ELEMENT, TYPE=CPE4, ELSET=E
1, 1, 2, 3, 4
*SURFACE, NAME=SLAVE, TYPE=ELEMENT
E, S1
*END PART
*PART, NAME=BOT
*NODE
1, 0.0, 0.0
2, 1.0, 0.0
3, 1.0, 1.0
4, 0.0, 1.0
*ELEMENT, TYPE=CPE4, ELSET=E
1, 1, 2, 3, 4
*SURFACE, NAME=MASTER, TYPE=ELEMENT
E, S3
*END PART
*MATERIAL, NAME=M
*ELASTIC
1000.0, 0.3
*SOLID SECTION, ELSET=E, MATERIAL=M
*ASSEMBLY, NAME=A
*INSTANCE, NAME=I_TOP, PART=TOP
*END INSTANCE
*INSTANCE, NAME=I_BOT, PART=BOT
*END INSTANCE
*END ASSEMBLY
*SURFACE INTERACTION, NAME=FR
*FRICTION
0.2
*CONTACT PAIR, INTERACTION=FR
I_BOT.MASTER, I_TOP.SLAVE
*STEP, NAME=S, NLGEOM=NO
*STATIC
0.1, 1.0
*END STEP
"""


def assembly_rotated_2d():
    """Single part, rotated instance (hierarchy coverage)."""
    return """*HEADING
Rotated-instance analogue
*PART, NAME=P
*NODE
1, 1.0, 0.0
2, 2.0, 0.0
3, 2.0, 1.0
4, 1.0, 1.0
*ELEMENT, TYPE=CPE4, ELSET=E
1, 1, 2, 3, 4
*NSET, NSET=EDGE
1, 2
*END PART
*MATERIAL, NAME=M
*ELASTIC
1000.0, 0.3
*SOLID SECTION, ELSET=E, MATERIAL=M
*ASSEMBLY, NAME=A
*INSTANCE, NAME=I, PART=P
1.0, 0.0, 0.0
0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 90.0
*NSET, NSET=ILOCAL
1
*END INSTANCE
*END ASSEMBLY
*STEP, NAME=S, NLGEOM=NO
*STATIC
0.1, 1.0
*END STEP
"""


if __name__ == "__main__":
    decks = [("cook_2d_coarse.inp", cook_2d()),
             ("cantilever_2d.inp", cantilever_2d()),
             ("cube_c3d8.inp", cube_c3d8()),
             ("patch_2d.inp", patch_2d()),
             ("cantilever_moment_2d.inp", cantilever_moment_2d()),
             ("thick_cylinder_2d.inp", thick_cylinder_2d()),
             ("tensile_bar_3d.inp", tensile_bar_3d()),
             ("shear_2d.inp", shear_2d()),
             ("cook_3d_coarse.inp", cook_3d()),
             ("contact_2block_2d.inp", contact_2block_2d()),
             ("assembly_rotated_2d.inp", assembly_rotated_2d())]
    for name, text in decks:
        print("wrote", write(name, text))
