# Implementation Plan: 3D 솔리드 요소 학술 계보, 개발자 및 문헌 DB 구축

## 1. Goal Description (목표 설명)
`WHT_DispFoldSolver`의 11종 3D 솔리드 요소(`C3D8`, `C3D8I`, `C3D8_FBAR`, `C3D8_CR`, `C3D8H`, `C3D8R`, `C3D4`, `C3D4_ANP`, `C3D10`, `C3D10M`, `C3D6`)에 대하여:
1. **요소별 개발자(Inventors & Key Contributors)**
2. **핵심 원저 문헌(Foundational Papers with Full Citations)**
3. **연관 및 후속 연구문헌과 주요 연구자(Milestone & Extended Literature)**
4. **상용 CAE(Abaqus, LS-DYNA, ANSYS) 정식화 대응 관계 및 수학적 메커니즘**
을 체계적으로 정리하여 [`dev_log/benchmark_3d_elements_20260912.md`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/dev_log/benchmark_3d_elements_20260912.md) 및 [`verification/benchmark_3d_elements.py`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/verification/benchmark_3d_elements.py), 그리고 [`AGENTS.md`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/AGENTS.md)에 영구 기록 및 관리합니다.

---

## 2. 11종 요소별 학술 계보 및 문헌 매핑 초안

```
[Isoparametric Trilinear]
 └── C3D8 (1968, Ergatoudis, Irons, Zienkiewicz)
      ├── [Reduced Integration + Hourglass]
      │    └── C3D8R (1981, Flanagan & Belytschko) ──> (2000, Puso EAS-HG)
      ├── [Incompatible Modes / EAS]
      │    └── C3D8I (1973, Wilson et al.) ──> (1990, Simo & Rifai) ──> (1992, Simo & Armero)
      ├── [F-bar Volumetric Projection]
      │    └── C3D8_FBAR (1980, Hughes B-bar) ──> (1996, de Souza Neto et al.)
      ├── [Co-rotational Geometric Non-linear]
      │    └── C3D8_CR (1973, Belytschko & Hsieh) ──> (2005, Felippa & Haugen)
      └── [Mixed Hybrid u-p]
           └── C3D8H (1965, Herrmann) ──> (1985, Simo, Taylor & Pister) ──> (1987, Sussman & Bathe)

[Tetrahedral Family]
 ├── C3D4 (1956, Turner, Clough, Martin, Topp - CST)
 │    └── C3D4_ANP (1998, Bonet & Burton 2-Pass ANP) ──> (2009, Gee et al.)
 └── C3D10 (1965, Argyris / 1971, Zienkiewicz - Quadratic Tet)
      └── C3D10M (1999, Abaqus HKS / 2001, Czekanski & Meguid / 2009, Gee et al.)

[Prism / Wedge Family]
 └── C3D6 (1969, Zienkiewicz & Irons / 1996, Bathe)
```

### 상세 요소별 연구자 및 서지 정보 리스트

1. **`C3D8` (Standard 8-Node Full Gauss Trilinear Hex)**:
   - **개발자**: Bruce M. Irons, Olgierd C. Zienkiewicz, J. G. Ergatoudis (Swansea Univ.)
   - **핵심문헌**: Ergatoudis, J. G., Irons, B. M., & Zienkiewicz, O. C. (1968). "Curved, isoparametric, 'quadrilateral' elements for finite element analysis." *International Journal of Solids and Structures*, 4(1), 31–42.
   - **연관연구자 & 문헌**: Zienkiewicz, O. C., & Taylor, R. L. (2000). *The Finite Element Method* (5th ed., Vol. 1). Butterworth-Heinemann.

2. **`C3D8I` (Incompatible Modes / Enhanced Assumed Strain Hex)**:
   - **개발자**: Edward L. Wilson, Robert L. Taylor (UC Berkeley); Juan C. Simo (Stanford)
   - **핵심문헌**:
     - Wilson, E. L., Taylor, R. L., Doherty, W. P., & Ghaboussi, J. (1973). "Incompatible displacement models." In *Numerical and Computer Methods in Structural Mechanics* (pp. 43–57). Academic Press.
     - Simo, J. C., & Rifai, M. S. (1990). "A class of mixed assumed strain methods and the method of incompatible modes." *International Journal for Numerical Methods in Engineering*, 29(8), 1595–1638.
   - **연관연구자 & 문헌**: Simo, J. C., & Armero, F. (1992). "Geometrically non-linear enhanced assumed strain methods and the problem of volumetric locking." *IJNME*, 33(7), 1413–1449.

3. **`C3D8_FBAR` (Multiplicative F-bar Hex)**:
   - **개발자**: Eduardo A. de Souza Neto, Djordje Peric, D. R. J. Owen (Swansea Univ.); Thomas J. R. Hughes (Stanford/UT Austin)
   - **핵심문헌**:
     - Hughes, T. J. R. (1980). "Generalization of selective integration procedures to anisotropic and nonlinear media." *International Journal for Numerical Methods in Engineering*, 15(9), 1413–1418.
     - de Souza Neto, E. A., Peric, D., Dutko, M., & Owen, D. R. J. (1996). "Design of simple low order finite elements for large strain analysis of nearly incompressible solids." *IJSS*, 33(20-22), 3277–3296.
   - **연관연구자 & 문헌**: Moran, B., Ortiz, M., & Shih, C. F. (1990). "Formulation of implicit finite element methods for multiplicative finite strain plasticity." *IJNME*, 29(3), 483–514.

4. **`C3D8_CR` (Co-rotational B-bar Hex + Distortion Control)**:
   - **개발자**: Ted Belytschko (Northwestern Univ.); Carlos A. Felippa (Univ. of Colorado Boulder); Bjørn Haugen (NTNU)
   - **핵심문헌**:
     - Belytschko, T., & Hsieh, B. J. (1973). "Non-linear transient finite element analysis with convected co-ordinates." *IJNME*, 7(3), 255–271.
     - Felippa, C. A., & Haugen, B. (2005). "A unified formulation of small-strain corotational finite elements: I. Theory." *Computer Methods in Applied Mechanics and Engineering*, 194(21-24), 2285–2335.
   - **연관연구자 & 문헌**: Rankin, C. C., & Brogan, F. A. (1986). "An element independent corotational procedure for the treatment of large rotations." *Journal of Pressure Vessel Technology*, 108(2), 165–174.

5. **`C3D8H` (Mixed Hybrid Volumetric Pressure Hex, u-p)**:
   - **개발자**: Leonard R. Herrmann (UC Davis); Juan C. Simo, Robert L. Taylor, Karl S. Pister (UC Berkeley); Klaus-Jürgen Bathe (MIT)
   - **핵심문헌**:
     - Herrmann, L. R. (1965). "Elasticity equations for incompressible and nearly incompressible materials by a variational theorem." *AIAA Journal*, 3(10), 1896–1900.
     - Simo, J. C., Taylor, R. L., & Pister, K. S. (1985). "Variational and projection methods for much nearly incompressible elasticity." *CMAME*, 51(1-3), 177–208.
   - **연관연구자 & 문헌**: Sussman, T., & Bathe, K. J. (1987). "A finite element formulation for nonlinear large strain elastic analysis using mixed interpolation." *Computers & Structures*, 26(1-2), 357–409.

6. **`C3D8R` (1-Point Reduced Hex + Flanagan-Belytschko Hourglass Control)**:
   - **개발자**: Dennis P. Flanagan, Ted Belytschko (Sandia / Northwestern Univ.); Michael A. Puso (LLNL)
   - **핵심문헌**:
     - Flanagan, D. P., & Belytschko, T. (1981). "A uniform strain hexahedron and quadrilateral with orthogonal hourglass control." *IJNME*, 17(5), 679–706.
   - **연관연구자 & 문헌**:
     - Belytschko, T., Ong, J. S. J., Liu, W. K., & Kennedy, J. M. (1984). "Hourglass control in linear and nonlinear problems." *CMAME*, 43(3), 251–276.
     - Puso, M. A. (2000). "A highly efficient enhanced assumed strain physically stabilized hexahedral element." *IJNME*, 49(8), 1029–1064.

7. **`C3D4` (Standard Linear 4-Node Tetrahedron, CST)**:
   - **개발자**: M. Jon Turner, Ray W. Clough, Harold C. Martin, LeRoy J. Topp (Boeing / UC Berkeley)
   - **핵심문헌**: Turner, M. J., Clough, R. W., Martin, H. C., & Topp, L. J. (1956). "Stiffness and deflection analysis of complex structures." *Journal of the Aeronautical Sciences*, 23(9), 805–823.
   - **연관연구자 & 문헌**: Gallagher, R. H., Padlog, J., & Bijlaard, P. P. (1962). "Stress analysis of heated complex shapes." *ARS Journal*, 32(5), 700–707.

8. **`C3D4_ANP` (2-Pass Average Nodal Pressure / F-bar Patch Projection Tet)**:
   - **개발자**: Javier Bonet, Anthony J. Burton (Swansea Univ.); Michael W. Gee (TUM); Clark R. Dohrmann (Sandia)
   - **핵심문헌**:
     - Bonet, J., & Burton, A. J. (1998). "A simple average nodal pressure tetrahedral element for finite strain analysis." *Communications in Numerical Methods in Engineering*, 14(5), 437–449.
   - **연관연구자 & 문헌**:
     - Gee, M. W., Dohrmann, C. R., Key, S. W., & Wall, W. A. (2009). "A constrained domain decomposition method for nearly incompressible finite elasticity with tetrahedral meshes." *CMAME*, 198(5-8), 747–760.
     - Puso, M. A., & Solberg, J. (2006). "A stabilized nodally integrated tetrahedral." *IJNME*, 67(6), 841–867.

9. **`C3D10` (Standard 10-Node Quadratic Isoparametric Tet)**:
   - **개발자**: John H. Argyris (Imperial College / Univ. of Stuttgart); Olgierd C. Zienkiewicz (Swansea Univ.)
   - **핵심문헌**: Argyris, J. H. (1965). "Triangular elements with proportional strain and displacement." *Journal of the Royal Aeronautical Society*, 69(658), 711–713.
   - **연관연구자 & 문헌**: Zienkiewicz, O. C. (1971). *The Finite Element Method in Engineering Science*. McGraw-Hill, London.

10. **`C3D10M` (Modified Quadratic Tet with B-bar, HG Control & Positive Contact Force)**:
    - **개발자**: Hibbitt, Karlsson & Sorensen (HKS / Abaqus Development Team, 1999); A. Czekanski, S. A. Meguid (Univ. of Toronto); Michael W. Gee (TUM)
    - **핵심문헌**:
      - Abaqus Theory Guide (1999–2026), §3.2.6: "Modified tetrahedral elements." Dassault Systèmes.
      - Czekanski, A., & Meguid, S. A. (2001). "Analysis of dynamic contact problems using modified quadratic tetrahedral elements." *Finite Elements in Analysis and Design*, 37(8), 629–641.
    - **연관연구자 & 문헌**:
      - Gee, M. W., Dohrmann, C. R., Key, S. W., & Wall, W. A. (2009). "A modified 10-node tetrahedral element with improved contact and incompressibility behavior." *IJNME*, 80(6-7), 785–815.
      - Joldes, G. R., Wittek, A., & Miller, K. (2009). "Suite of finite element implementations of modified tetrahedral elements for non-linear biomechanics." *Medical Image Analysis*, 13(6), 912–923.

11. **`C3D6` (6-Node Linear Triangular Prism / Wedge)**:
    - **개발자**: Olgierd C. Zienkiewicz, Bruce M. Irons (Swansea Univ.); Klaus-Jürgen Bathe (MIT)
    - **핵심문헌**: Zienkiewicz, O. C., Irons, B. M., Ergatoudis, J., Ahmad, S., & Scott, F. C. (1969). "Iso-parametric and associated element families for two and three dimensional analysis." In *Finite Element Techniques in Structural Mechanics* (pp. 383–432).
    - **연관연구자 & 문헌**: Bathe, K. J. (1996). *Finite Element Procedures*. Prentice Hall, Englewood Cliffs, NJ.

---

## 3. Proposed Changes (변경 계획)

### Component 1: `verification/benchmark_3d_elements.py`
- `save_benchmark_report()` 함수를 확장하여 11개 요소의 **연구자(개발자), 핵심문헌, 연관문헌, 서지정보, 수치적 메커니즘**을 마크다운 표 및 상세 섹션으로 자동 포맷팅하도록 업데이트.
- 향후 벤치마크 재실행 시 항상 최신 문헌 DB가 보고서에 포함되도록 유지.

### Component 2: `dev_log/benchmark_3d_elements_20260912.md`
- 벤치마크 결과표 바로 아래에 `## 3. 요소별 개발자, 핵심문헌 및 연관 연구문헌 (Literature Database)` 섹션을 신설.
- 11종 전체에 대한 1) 개발자, 2) 핵심 문헌, 3) 연관 연구문헌 및 연구자, 4) 정식화 특징을 서지 포맷으로 영구 보존.

### Component 3: `AGENTS.md`
- 3D Finite Element Library 섹션에 본 학술 계보 및 핵심 문헌 요약 링크를 추가하여 향후 에이전트들이 이론적 배경을 즉각 참조할 수 있도록 동기화.

---

## 4. Verification Plan (검증 계획)

### Automated Tests
1. `python -u verification/benchmark_3d_elements.py`:
   - 11종 요소 벤치마크가 정상 실행되고, `dev_log/benchmark_3d_elements_20260912.md`에 모든 서지정보 및 개발자 메타데이터가 UTF-8로 깨짐 없이 생성되는지 확인.
2. `git diff` 및 마크다운 렌더링 확인:
   - 인용 문헌 및 DOI 링크, 표 정렬 상태 점검.

### Manual Verification
- `dev_log/benchmark_3d_elements_20260912.md` 파일을 열람하여 모든 11개 요소의 개발자명, 연도, 저널명, 볼륨, 페이지가 학술 규격에 맞게 기재되었는지 점검.
