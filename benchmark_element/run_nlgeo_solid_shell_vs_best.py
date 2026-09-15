"""
run_nlgeo_solid_shell_vs_best.py
=================================
SOLID_SHELL(ANS+EAS Solid-Shell) 대변형(Large Deflection) 캔틸레버 벤치마크.
기존 최강자 C3D8I_CR, C3D8I와 Bisshopp & Drucker(1945) 탄성 곡선 정밀 이론해 비교.

실행:
    python benchmark_element/run_nlgeo_solid_shell_vs_best.py

결과:
    콘솔 출력: 각 요소의 최종 팁 변위(uy, ux)와 이론 대비 오차(%)
    저장 그래프: dev_log/figures/nlgeo_solid_shell_vs_best.png
"""

import sys
from pathlib import Path
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from benchmark_element.benchmark_nlgeo_cantilever import (
    run_3d_cantilever_transverse,
    BisshoppDruckerElastica,
    E_MOD, I_SECTION, L_BEAM, P_TIP
)


def main():
    # ------------------------------------------------------------------
    # 1. Bisshopp & Drucker(1945) 정밀 이론해 계산
    # ------------------------------------------------------------------
    elastica = BisshoppDruckerElastica(E_MOD, I_SECTION, L_BEAM, P_TIP)
    theta_tip, ux_exact, uy_exact = elastica.solve_at_load(P_TIP)

    print("=" * 65)
    print("  Bisshopp & Drucker (1945) 탄성 곡선 이론해")
    print("=" * 65)
    print(f"  P_TIP   = {P_TIP:.4f} N")
    print(f"  EI      = {E_MOD * I_SECTION:.6e} N·m²")
    print(f"  theta_tip = {np.degrees(theta_tip):.4f}°")
    print(f"  uy (수직 처짐) = {uy_exact:.6f} m")
    print(f"  ux (축 단축)   = {ux_exact:.6f} m")
    print()

    # ------------------------------------------------------------------
    # 2. 비교 요소 목록 (최강자 vs SOLID_SHELL)
    # ------------------------------------------------------------------
    candidates = [
        ("C3D8I_CR",    "EAS+Corotational (현 최강자)"),
        ("C3D8I",       "EAS-9 Total Lagrangian"),
        ("C3D8_CR",     "Co-rotational (기준)"),
        ("SOLID_SHELL", "ANS+EAS Solid-Shell (신규)"),
    ]

    results = {}
    print("=" * 65)
    print("  대변형 캔틸레버 벤치마크 실행 중 (coarse mesh)...")
    print("=" * 65)

    for etype, label in candidates:
        print(f"\n  [{etype}] {label} 실행 중...", end="", flush=True)
        try:
            res = run_3d_cantilever_transverse(etype, mesh_density="coarse")
            final_uy = res["final_uy"]
            final_ux = res["final_ux"]
            cutbacks  = res["cutbacks"]
            err_uy = abs(final_uy - uy_exact) / abs(uy_exact) * 100.0
            err_ux = abs(final_ux - ux_exact) / abs(ux_exact) * 100.0 if abs(ux_exact) > 1e-12 else 0.0
            results[etype] = {
                "label":    label,
                "final_uy": final_uy,
                "final_ux": final_ux,
                "err_uy":   err_uy,
                "err_ux":   err_ux,
                "cutbacks": cutbacks,
                "p_history":  res["p_history"],
                "uy_history": res["uy_history"],
                "profile_x":  res["profile_x"],
                "profile_y":  res["profile_y"],
            }
            print(f"  완료 (cutbacks={cutbacks})")
        except Exception as e:
            print(f"  ERROR: {e}")
            results[etype] = None

    # ------------------------------------------------------------------
    # 3. 결과 표 출력
    # ------------------------------------------------------------------
    print()
    print("=" * 65)
    print("  대변형 캔틸레버 결과 비교 (이론 대비 오차)")
    print("=" * 65)
    hdr = f"  {'요소':16s} {'uy [m]':>10s} {'err_uy%':>9s} {'ux [m]':>10s} {'err_ux%':>9s} {'cutbacks':>8s}"
    print(hdr)
    print("  " + "-" * 63)
    print(f"  {'[이론해]':16s} {uy_exact:>10.6f} {'':>9s} {ux_exact:>10.6f} {'':>9s}")
    print("  " + "-" * 63)

    for etype, _ in candidates:
        r = results.get(etype)
        if r is None:
            print(f"  {etype:16s} {'ERROR':>10s}")
            continue
        verdict = "PASS" if r["err_uy"] < 5.0 else ("WARN" if r["err_uy"] < 15.0 else "FAIL")
        print(
            f"  {etype:16s} {r['final_uy']:>10.6f} {r['err_uy']:>8.3f}%"
            f" {r['final_ux']:>10.6f} {r['err_ux']:>8.3f}%"
            f" {r['cutbacks']:>8d}  {verdict}"
        )

    # ------------------------------------------------------------------
    # 4. 변형 형상(Profile) 및 팁 변위 이력 플롯
    # ------------------------------------------------------------------
    try:
        import koreanize_matplotlib  # noqa: F401
    except ImportError:
        pass
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # -- 이론 탄성 곡선 (연속)
    p_vals = np.linspace(0, P_TIP, 80)
    uy_th_list, ux_th_list = [], []
    for p in p_vals[1:]:
        _, ux_p, uy_p = elastica.solve_at_load(p)
        uy_th_list.append(uy_p)
        ux_th_list.append(ux_p)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 팁 변위 이력
    ax = axes[0]
    ax.plot(p_vals[1:], uy_th_list, "k-", lw=2, label="이론해 (Bisshopp-Drucker)")
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for (etype, _), col in zip(candidates, colors):
        r = results.get(etype)
        if r is None:
            continue
        p_hist = r["p_history"] * P_TIP
        uy_hist = r["uy_history"]
        ax.plot(p_hist, uy_hist, "o--", color=col, ms=4, lw=1.2, label=etype)
    ax.set_xlabel("하중 P [N]")
    ax.set_ylabel("팁 수직 처짐 uy [m]")
    ax.set_title("대변형 캔틸레버 — 팁 변위 이력")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4)

    # 변형 형상 (최종)
    ax = axes[1]
    # 이론 탄성 곡선 형상 (연속 s 분포)
    s_arr = np.linspace(0, L_BEAM, 200)
    beta_v = np.sqrt(2.0 * P_TIP / (E_MOD * I_SECTION))
    smax_v = np.sqrt(theta_tip)
    # 상세 탄성 곡선은 세그먼트 분할로 구성
    x_th, y_th = [0.0], [0.0]
    for seg in range(1, 201):
        s_loc = seg / 200 * L_BEAM
        # 해당 s 위치에서 각도 구하기 (역산 적분 — 근사)
        pass  # 상세 탄성 곡선 대신 이론 종점 표시
    ax.scatter([L_BEAM - ux_exact], [uy_exact], marker="*", s=200,
               color="black", zorder=5, label=f"이론 팁점 ({uy_exact:.3f}m)")
    for (etype, _), col in zip(candidates, colors):
        r = results.get(etype)
        if r is None:
            continue
        ax.plot(r["profile_x"], r["profile_y"], "-", color=col, lw=1.5, label=etype)
    ax.set_xlabel("변형 후 X [m]")
    ax.set_ylabel("변형 후 Z [m]")
    ax.set_title("대변형 캔틸레버 — 최종 변형 형상")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4)
    ax.set_aspect("equal")

    plt.tight_layout()
    out_dir = repo_root / "dev_log" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "nlgeo_solid_shell_vs_best.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  그래프 저장: {out_path}")

    # ------------------------------------------------------------------
    # 5. 최종 판정
    # ------------------------------------------------------------------
    print()
    print("=" * 65)
    print("  최종 판정 (uy 오차 < 5%: PASS)")
    print("=" * 65)
    for etype, _ in candidates:
        r = results.get(etype)
        if r is None:
            continue
        status = "PASS" if r["err_uy"] < 5.0 else ("MARGINAL" if r["err_uy"] < 15.0 else "FAIL")
        print(f"  {etype:16s}: uy 오차 {r['err_uy']:.3f}%  → {status}")


if __name__ == "__main__":
    main()
