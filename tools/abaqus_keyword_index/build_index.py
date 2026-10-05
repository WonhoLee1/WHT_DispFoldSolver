"""Build dispsolver/io/abaqus_keyword_index.json from the Abaqus Keywords Guide.

The Keywords Guide pages (SIMACAEKEYRefMap/simakey-r-*.htm) carry semantic
markup -- product list, data type (model / history), allowed levels
(part / assembly / model / step), per-usage required and optional parameter
lists with their values and defaults, and data-line layouts. This tool
extracts that STRUCTURE (names, levels, parameter names/values/defaults,
data-line item counts and short item labels) into a JSON index; it stores no
descriptive prose.

Usage:
    python -m tools.abaqus_keyword_index.build_index \
        --base http://desktop-whl:4040/English \
        --cache <dir for downloaded pages> \
        --out dispsolver/io/abaqus_keyword_index.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from bs4 import BeautifulSoup

BROWSER = "SIMACAECAERefMap/simacae-c-gen-kwbrowser.htm"
KEYDIR = "SIMACAEKEYRefMap"


def fetch(base: str, rel: str, cache: str) -> str:
    path = os.path.join(cache, rel.replace("/", "__"))
    if os.path.exists(path):
        return open(path, encoding="utf-8").read()
    with urllib.request.urlopen(f"{base}/{rel}", timeout=30) as r:
        text = r.read().decode("utf-8", errors="replace")
    os.makedirs(cache, exist_ok=True)
    open(path, "w", encoding="utf-8").write(text)
    return text


def _txt(el) -> str:
    return re.sub(r"\s+", " ", el.get_text(" ", strip=True)) if el is not None else ""


def _short(s: str, n: int = 90) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    s = re.split(r"(?<=[.;])\s", s, maxsplit=1)[0]
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def _param_entry(dt, dd) -> dict:
    """Name, allowed values and default of one parameter, read from the
    markup: values are <abqparamvalue> spans that follow '<abqparameter>NAME
    </abqparameter>=' inside the description; '(default)' right after a value
    marks the default."""
    name_el = dt.find(class_="abqparameter")
    name = (_txt(name_el) if name_el else _txt(dt)).split("=")[0].strip().upper()
    values, default = [], None
    for pv in dd.find_all(class_="abqparamvalue"):
        prev = pv.previous_sibling
        prev_txt = prev if isinstance(prev, str) else (prev.get_text() if prev is not None else "")
        if not prev_txt.rstrip().endswith("="):
            continue
        pname = pv.find_previous(class_="abqparameter")
        if pname is None or _txt(pname).strip().upper() != name:
            continue
        v = _txt(pv).strip().upper()
        if not v or re.fullmatch(r"[0-9.+\-E]+", v):
            continue
        if v not in values:
            values.append(v)
        nxt = pv.next_sibling
        nxt_txt = nxt if isinstance(nxt, str) else (nxt.get_text() if nxt is not None else "")
        if default is None and nxt_txt.lstrip().startswith("(default)"):
            default = v
    text = _txt(dd)
    flag = bool(re.search(r"\bInclude this parameter\b", text, re.I))
    out = {"name": name}
    if values:
        out["values"] = values
    if default is not None:
        out["default"] = default
    if flag:
        out["flag"] = True
    first = _short(text, 140)
    if re.search(r"\b(only|if|when|unless)\b", first, re.I):
        out["applies"] = first           # usage restriction stated in the first sentence
    return out


def _params(section) -> list:
    out = []
    for dl in section.find_all("dl", class_="paramlist"):
        for dt in dl.find_all("dt", class_="paramhead"):
            dd = dt.find_next_sibling("dd")
            if dd is not None:
                out.append(_param_entry(dt, dd))
    return out


def _datalines(section) -> dict:
    title = _txt(section.find(class_="datadesctitle"))
    lines = []
    for dl in section.find_all("dl", class_="datalines"):
        for dt in dl.find_all("dt", class_="datalinetitle"):
            dd = dt.find_next_sibling("dd")
            items = [_short(_txt(li)) for li in dd.find_all("li", class_="dataitem")] if dd else []
            lines.append({"line": _txt(dt), "items": items})
    after = _txt(section)
    return {"title": _short(title, 140), "lines": lines,
            "repeat": bool(re.search(r"Repeat this data line|Repeat this set of data lines", after, re.I))}


_PLAIN_TITLES = re.compile(r"^(required|optional) parameters?$", re.I)


def _usage(node, title: str) -> dict:
    """Parameters of one usage. A parameter listed under a plainly titled
    "Required parameter(s)" section is required; under a qualified title
    ("Required parameter if DEFINITION=EQUALLY SPACED", "... (history data
    only)") it carries that title as its condition."""
    u = {"title": _short(title, 140), "required": [], "optional": [], "data": []}
    for sec in node.find_all("div", class_="section"):
        cls = " ".join(sec.get("class", []))
        st = _txt(sec.find(class_="sectiontitle"))
        if any(c in cls for c in ("paramsrequired", "paramsoptional", "paramscustom")):
            kind = ("required" if "paramsrequired" in cls else "optional" if "paramsoptional" in cls
                    else ("required" if re.search(r"\brequired\b", st, re.I) else "optional"))
            cond = None if (not st or _PLAIN_TITLES.match(st.strip())) else _short(st, 140)
            for prm in _params(sec):
                if cond:
                    prm["condition"] = cond
                elif kind == "required" and (prm.get("flag") or prm.get("applies")):
                    # a flag or a restricted parameter under "Required parameters"
                    # is one of alternative forms, not unconditionally required
                    prm["condition"] = prm.get("applies") or "alternative form (flag)"
                u[kind].append(prm)
        elif "datadesc" in cls:
            u["data"].append(_datalines(sec))
    return u


def _standard(text: str) -> str:
    """The Abaqus/Standard part of 'X in Abaqus/Standard; Y in Abaqus/Explicit'."""
    t = text.lower().strip()
    if ";" in t:
        for seg in t.split(";"):
            if "abaqus/standard" in seg:
                return seg.replace("in abaqus/standard", "").strip()
        return t.split(";")[0].strip()
    return t


def parse_page(slug: str, html: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    title = _txt(soup.find("title")) or _txt(soup.find("h1"))
    name = re.sub(r"^\*\s*", "", title).strip().upper()
    head = soup.find(class_="abqkeysectionbody") or soup
    raw_type = _txt(head.select_one("p.kwtype span.key")) or ""
    t_std = _standard(raw_type)
    levels, assembly_ok = [], True
    for sp in head.select("p.kwlevel span.key"):
        lv = _txt(sp).lower()
        if "not supported in a model defined in terms of an assembly" in lv:
            assembly_ok = False
            continue
        for part in re.split(r"\s+or\s+|,", _standard(lv)):
            part = part.strip()
            if part and part not in levels:
                levels.append(part)
    entry = {"keyword": name, "slug": slug,
             "products": [_txt(s) for s in head.select("p.abqproductname span.key")],
             "data_type": ("history" if t_std.startswith("history") else
                           "both" if "model or history" in t_std else
                           "model" if t_std.startswith("model") else t_std),
             "levels": levels,
             "assembly_supported": assembly_ok}
    subs = soup.find_all(class_="abqkeysubsection")
    usages = []
    if subs:
        for sub in subs:
            usages.append(_usage(sub, _txt(sub.find(["h2", "h3"]))))
    else:
        usages.append(_usage(soup.find(class_="abqkeysectionbody") or soup, ""))
    entry["usages"] = usages
    return entry


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://desktop-whl:4040/English")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default=os.path.join("dispsolver", "io", "abaqus_keyword_index.json"))
    a = ap.parse_args(argv)
    browser = fetch(a.base, BROWSER, a.cache)
    slugs = sorted(set(re.findall(r"simakey-r-[a-z0-9_\-]+\.htm", browser)))
    with ThreadPoolExecutor(8) as ex:
        pages = list(ex.map(lambda s: (s, fetch(a.base, f"{KEYDIR}/{s}", a.cache)), slugs))
    index, problems = {}, []
    for slug, html in pages:
        try:
            e = parse_page(slug, html)
        except Exception as exc:     # report, do not guess
            problems.append(f"{slug}: {type(exc).__name__}: {exc}")
            continue
        if not e["keyword"]:
            problems.append(f"{slug}: no keyword name")
            continue
        index[e["keyword"]] = e
    doc = {"source": "Abaqus Keywords Guide (SIMACAEKEYRefMap), structure only",
           "count": len(index), "keywords": dict(sorted(index.items()))}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False)
    print(f"{len(index)} keywords -> {a.out}; {len(problems)} problems")
    for p in problems[:20]:
        print("  ", p)
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
