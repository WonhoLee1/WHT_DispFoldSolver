"""Inventory: read every official deck, classify status. Scratch report."""
import glob
import os
import traceback
import warnings

from dispsolver.io.abaqus_reader import _parse_source, _AbortParse

D = os.path.dirname(os.path.abspath(__file__))


def inventory(dim):
    rows = []
    for path in sorted(glob.glob(os.path.join(D, "*.inp"))):
        name = os.path.basename(path)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                reader, model = _parse_source(path, dim=dim)
            n_nodes = sum(len(p.nodes) for p in model.parts.values())
            n_elems = sum(len(p.elements) for p in model.parts.values())
            rows.append((name, "OK", n_nodes, n_elems,
                         sorted(model.steps.keys()),
                         sorted(set(reader.unsupported_keywords)),
                         len(reader.warnings)))
        except _AbortParse as e:
            rows.append((name, "ABORT", 0, 0, [], [], str(e)[:80]))
        except Exception as e:  # noqa: BLE001 - inventory must not stop
            rows.append((name, "ERROR", 0, 0, [], [],
                         (str(e)[:80] + traceback.format_exc(limit=3)[-200:])))
    return rows


if __name__ == "__main__":
    for dim in (2, 3):
        print(f"===== dim={dim} =====")
        for r in inventory(dim):
            print(r[0], "|", r[1], "| nodes:", r[2], "| elems:", r[3],
                  "| steps:", r[4], "| unsupported:", r[5],
                  ("| warns:" + str(r[6]) if r[1] == "OK" else "| " + str(r[6])))
