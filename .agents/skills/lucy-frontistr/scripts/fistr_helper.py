# -*- coding: utf-8 -*-
"""
lucy-frontistr 헬퍼 스크립트 (fistr_helper.py).
FrontISTR SMP/MPI 실행, 완료 무결성 검증('FrontISTR Completed !!'),
0.log / FSTR.sta 요약 파싱 및 한국어 공식 매뉴얼(docs_ko) 고속 검색을 제공합니다.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


class FrontISTRHelper:
    """FrontISTR 실행, 결과 파싱 및 한국어 매뉴얼 검색을 수행하는 클래스."""

    ROOT_DIR = Path(r"D:\PythonCodeStudy\FrontISTR")
    DOCS_KO_DIR = ROOT_DIR / "docs_ko"
    FABRICA_DIR = ROOT_DIR / "frontistr-fabrica-main"
    MASTER_DIR = ROOT_DIR / "FrontISTR-master"
    SMP_BIN_DIR = ROOT_DIR / "FrontISTR-latest.thread_openblas"
    MPI_BIN_DIR = ROOT_DIR / "FrontISTR-latest.hybrid_msmpi_openblas"

    def __init__(self) -> None:
        """FrontISTRHelper 초기화 및 경로 검증."""
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
        self.smp_fistr = self.SMP_BIN_DIR / "fistr1.exe"
        self.mpi_fistr = self.MPI_BIN_DIR / "fistr1.exe"
        self.mpi_part = self.MPI_BIN_DIR / "hecmw_part1.exe"
        self.mpiexec = self.MPI_BIN_DIR / "mpiexec.exe"

    def check_environment(self) -> Dict[str, Any]:
        """FrontISTR 주요 폴더 및 실행 파일의 존재 여부를 확인합니다.

        Returns:
            Dict[str, Any]: 각 경로의 존재 여부 및 상태 정보.
        """
        paths = {
            "root": self.ROOT_DIR,
            "docs_ko": self.DOCS_KO_DIR,
            "fabrica": self.FABRICA_DIR,
            "source_master": self.MASTER_DIR,
            "smp_fistr1": self.smp_fistr,
            "mpi_fistr1": self.mpi_fistr,
            "mpi_part1": self.mpi_part,
            "mpiexec": self.mpiexec,
        }
        status = {k: {"path": str(v), "exists": v.exists()} for k, v in paths.items()}
        return status

    def search_docs(self, query: str, max_results: int = 15) -> List[Dict[str, Any]]:
        """한국어 매뉴얼(docs_ko)에서 키워드를 검색합니다.

        Args:
            query (str): 검색할 키워드 또는 정규식 문자열.
            max_results (int): 최대 반환 문서 수.

        Returns:
            List[Dict[str, Any]]: 매칭된 문서 경로와 스니펫 목록.
        """
        results: List[Dict[str, Any]] = []
        if not self.DOCS_KO_DIR.exists():
            return results

        pattern = re.compile(re.escape(query), re.IGNORECASE)
        for root, dirs, files in os.walk(self.DOCS_KO_DIR):
            if "consolidated" in Path(root).parts:
                continue
            for fname in sorted(files):
                if not fname.endswith(".md") or fname == "README.md":
                    continue
                fpath = Path(root) / fname
                rel_path = fpath.relative_to(self.DOCS_KO_DIR).as_posix()
                try:
                    content = fpath.read_text(encoding="utf-8")
                except Exception:
                    continue

                matches = []
                lines = content.splitlines()
                for idx, line in enumerate(lines):
                    if pattern.search(line):
                        start = max(0, idx - 1)
                        end = min(len(lines), idx + 2)
                        snippet = " / ".join(l.strip() for l in lines[start:end] if l.strip())
                        matches.append(f"L{idx + 1}: {snippet[:180]}")
                        if len(matches) >= 3:
                            break

                if matches:
                    results.append(
                        {
                            "file": rel_path,
                            "abs_path": str(fpath),
                            "matches": matches,
                        }
                    )
                    if len(results) >= max_results:
                        return results
        return results

    def parse_summary(self, work_dir: Path) -> Dict[str, Any]:
        """작업 디렉토리의 out_fistr.log, 0.log, FSTR.sta를 파싱하여 결과 요약을 반환합니다.

        Args:
            work_dir (Path): 해석이 실행된 디렉토리 경로.

        Returns:
            Dict[str, Any]: 완료 여부, Global Summary 물리량 최대/최소, 스텝 수렴 상태.
        """
        summary: Dict[str, Any] = {
            "work_dir": str(work_dir),
            "completed": False,
            "node_summary": [],
            "element_summary": [],
            "sta_tail": [],
        }

        out_log = work_dir / "out_fistr.log"
        if out_log.exists():
            txt = out_log.read_text(encoding="utf-8", errors="replace")
            if "FrontISTR Completed !!" in txt:
                summary["completed"] = True

        zero_log = work_dir / "0.log"
        if zero_log.exists():
            txt0 = zero_log.read_text(encoding="utf-8", errors="replace")
            if "FrontISTR Completed !!" in txt0:
                summary["completed"] = True

            lines = txt0.splitlines()
            mode = None
            for line in lines:
                if "Global Summary @Node" in line:
                    mode = "node"
                    summary["node_summary"].append(line.strip())
                    continue
                elif "Global Summary @Element" in line:
                    mode = "elem"
                    summary["element_summary"].append(line.strip())
                    continue
                elif line.strip().startswith("//"):
                    if mode == "node":
                        summary["node_summary"].append(line.strip())
                    elif mode == "elem":
                        summary["element_summary"].append(line.strip())
                elif mode and not line.strip().startswith(("Max", "Min", "//")):
                    mode = None

        fstr_sta = work_dir / "FSTR.sta"
        if fstr_sta.exists():
            sta_lines = [
                l.strip()
                for l in fstr_sta.read_text(encoding="utf-8", errors="replace").splitlines()
                if l.strip()
            ]
            summary["sta_tail"] = sta_lines[-10:]

        return summary

    def run_solver(
        self,
        work_dir: Path,
        mode: str = "smp",
        np_procs: int = 2,
        threads: int = 2,
    ) -> Dict[str, Any]:
        """FrontISTR 솔버를 SMP 또는 MPI 모드로 실행하고 결과를 검증합니다.

        Args:
            work_dir (Path): hecmw_ctrl.dat가 위치한 작업 폴더.
            mode (str): 'smp' 또는 'mpi'.
            np_procs (int): MPI 프로세스 수 (mode='mpi'일 때 사용).
            threads (int): OpenMP 스레드 수 (-t 옵션).

        Returns:
            Dict[str, Any]: 실행 코드, 완료 여부 및 요약 데이터.
        """
        work_dir = work_dir.resolve()
        if not (work_dir / "hecmw_ctrl.dat").exists():
            raise FileNotFoundError(f"hecmw_ctrl.dat not found in {work_dir}")

        out_log = work_dir / "out_fistr.log"

        if mode.lower() == "mpi":
            part_ctrl = work_dir / "hecmw_part_ctrl.dat"
            if part_ctrl.exists():
                part_log = work_dir / "out_part.log"
                with open(part_log, "w", encoding="utf-8") as pf:
                    subprocess.run(
                        [str(self.mpi_part)],
                        cwd=str(work_dir),
                        stdout=pf,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
            cmd = [
                str(self.mpiexec),
                "-np",
                str(np_procs),
                str(self.mpi_fistr),
                "-t",
                str(threads),
            ]
        else:
            cmd = [str(self.smp_fistr), "-t", str(threads)]

        with open(out_log, "w", encoding="utf-8") as lf:
            proc = subprocess.run(
                cmd,
                cwd=str(work_dir),
                stdout=lf,
                stderr=subprocess.STDOUT,
                check=False,
            )

        res = self.parse_summary(work_dir)
        res["exit_code"] = proc.returncode
        res["command"] = " ".join(cmd)
        return res


def test_environment_and_search() -> None:
    """환경 경로 검증 및 한국어 매뉴얼 검색 테스트 함수."""
    helper = FrontISTRHelper()
    env_info = helper.check_environment()
    print("=== [Test 1] Environment Check ===")
    print(json.dumps(env_info, indent=2, ensure_ascii=False))
    for k, v in env_info.items():
        assert v["exists"], f"Missing required path for {k}: {v['path']}"

    print("\n=== [Test 2] Korean Manual Search ('FORM361') ===")
    hits = helper.search_docs("FORM361", max_results=5)
    for h in hits:
        print(f" - {h['file']}: {h['matches'][0]}")
    assert len(hits) > 0, "Expected at least 1 search hit for FORM361"


def main() -> None:
    """CLI 엔트리 포인트."""
    parser = argparse.ArgumentParser(description="lucy-frontistr Helper CLI")
    parser.add_argument("--check-env", action="store_true", help="환경 경로 및 바이너리 상태 점검")
    parser.add_argument("--search", type=str, help="한국어 매뉴얼(docs_ko) 키워드 검색")
    parser.add_argument("--run", type=str, help="지정한 작업 디렉토리에서 FrontISTR 실행")
    parser.add_argument("--mode", type=str, default="smp", choices=["smp", "mpi"], help="실행 모드 (smp | mpi)")
    parser.add_argument("--np", type=int, default=2, help="MPI 프로세스 수")
    parser.add_argument("--threads", type=int, default=4, help="OpenMP 스레드 수")
    parser.add_argument("--summary", type=str, help="지정한 작업 디렉토리의 0.log / FSTR.sta 요약 출력")
    parser.add_argument("--self-test", action="store_true", help="내장 자체 검증 테스트 실행")

    args = parser.parse_args()
    helper = FrontISTRHelper()

    if args.self_test:
        test_environment_and_search()
        return

    if args.check_env:
        print(json.dumps(helper.check_environment(), indent=2, ensure_ascii=False))
        return

    if args.search:
        hits = helper.search_docs(args.search)
        print(json.dumps(hits, indent=2, ensure_ascii=False))
        return

    if args.run:
        res = helper.run_solver(
            Path(args.run), mode=args.mode, np_procs=args.np, threads=args.threads
        )
        print(json.dumps(res, indent=2, ensure_ascii=False))
        if not res["completed"]:
            sys.exit(1)
        return

    if args.summary:
        res = helper.parse_summary(Path(args.summary))
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
