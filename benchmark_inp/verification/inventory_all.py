"""Full inventory over benchmark_inp/verification/*.inp (1,859 decks)."""
import glob
import os
import warnings
from collections import Counter

from dispsolver.io.abaqus_reader import _parse_source, _AbortParse

D = os.path.dirname(os.path.abspath(__file__))

ok = abort = err = 0
unsup = Counter()
abort_why = Counter()
err_why = Counter()
ok_names = []

for path in sorted(glob.glob(os.path.join(D, "*.inp"))):
    name = os.path.basename(path)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reader, model = _parse_source(path, dim=3)
        ok += 1
        ok_names.append(name)
        for k in reader.unsupported_keywords:
            unsup[k] += 1
    except _AbortParse as e:
        abort += 1
        abort_why[str(e)[:60]] += 1
    except Exception as e:  # noqa: BLE001 - inventory must not stop
        err += 1
        err_why[type(e).__name__ + ": " + str(e)[:70]] += 1

print(f"OK={ok} ABORT={abort} ERROR={err}")
print("--- top unsupported ---")
for k, v in unsup.most_common(25):
    print(f"{v:5d}  {k}")
print("--- abort reasons ---")
for k, v in abort_why.most_common(10):
    print(f"{v:5d}  {k}")
print("--- errors ---")
for k, v in err_why.most_common(10):
    print(f"{v:5d}  {k}")
with open(os.path.join(D, "inventory_ok.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(ok_names) + "\n")
