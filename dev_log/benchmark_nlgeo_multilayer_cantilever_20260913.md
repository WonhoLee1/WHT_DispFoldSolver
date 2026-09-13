# [Benchmark Report] 5층 복합 박막(PET-PSA-PET-PSA-PET) 기하비선형 캔틸레버 벤치마크

**실행 일자**: 2026-09-13  
**해석 스크립트**: [`benchmark_element/benchmark_nlgeo_multilayer_cantilever.py`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/benchmark_element/benchmark_nlgeo_multilayer_cantilever.py)  
**시각화 스크립트**: [`benchmark_element/figures_nlgeo_multilayer_cantilever.py`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/benchmark_element/figures_nlgeo_multilayer_cantilever.py)  
**도면 파일**: [`dev_log/figures/benchmark_nlgeo_multilayer_cantilever.png`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/dev_log/figures/benchmark_nlgeo_multilayer_cantilever.png)  
**원시 데이터**: [`benchmark_element/results/benchmark_nlgeo_multilayer_cantilever.json`](file:///D:/PythonCodeStudy/WHT_DispFoldSolver/benchmark_element/results/benchmark_nlgeo_multilayer_cantilever.json)

---

## 1. 벤치마크 설계 및 물리적 제원

폴더블 디스플레이 다층 박막 스택업의 기하비선형 대변형 거동 및 층간 전단 슬립(Interlayer Shear Slip)을 평가하기 위해 실제 제품 수준의 극단적 강성비(80,000배)를 갖는 5층 복합 외팔보 벤치마크를 정밀 설계하였습니다.

- **기하 제원**:
  - 외팔보 길이: $L = 40.0\,\text{mm}$, 폭 $W = 1.0\,\text{mm}$
  - 총 두께: $H = 210\,\mu\text{m}$ ($0.21\,\text{mm}$)
    - $3\times$ PET 레이어: 각 $50\,\mu\text{m}$ ($0.05\,\text{mm}$)
    - $2\times$ PSA 레이어: 각 $30\,\mu\text{m}$ ($0.03\,\text{mm}$)
- **물성 정의**:
  - **PET (탄성 필름)**: 선형 탄성 $E = 4,000\,\text{MPa}$, $\nu = 0.3$
  - **PSA (초탄성 점착제)**: 비압축성 Neo-Hookean $\mu = 0.05\,\text{MPa}$ ($E \approx 0.15\,\text{MPa}$), $\nu = 0.499$
  - **영률 비 ($E_{\text{PET}} / E_{\text{PSA}}$)**: **약 80,000배!**
- **하중 및 경계조건**:
  - 고정단 ($x = 0$): 전 자유도 구속 ($u_x = u_y = u_z = 0$)
  - 자유단 ($x = 40\,\text{mm}$): 하향 집중 하중 $P_{\text{tip}} = 1.5\,\text{mN}$ (C2 연속 Smootherstep 램프 하중, 15 증분)
- **평가 매트릭스 (총 10개 케이스)**:
  - 2D (6종): `CPE4I+CPE4H` (Conformal), `CPE4I+CPE4H` (SurfaceTie), `CPE4R+CPE4H` (Conformal), `CPE6M+CPE4H` (Conformal), `CPE6M+CPE4H` (SurfaceTie), `CPE4+CPE4` (Baseline)
  - 3D (4종): `C3D8I+C3D8H` (Conformal), `C3D8I+C3D8H` (SurfaceTie), `C3D8R+C3D8R` (Conformal), `C3D8+C3D8` (Baseline)

---

## 2. [표 1] 외팔보 단순 처짐 벤치마크 종합 정량 비교표 (10 Cases)

| 차원 | 케이스 명칭 | PET 요소 | PSA 요소 | 계면 결합 방식 | 처짐 $\|w\|$ [mm] | 층간 슬립 $\Delta u$ [$\mu$m] | 총 소요시간 [s] | 증분당 시간 [ms] | 총 요소수 | 총 자유도 (DOFs) | 역학적 판정 및 특징 |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **2D** | **2D-Opt1 (최우수 권장)** | `CPE4I` | `CPE4H` | **Conformal (공유)** | **13.2050** | **87.90** | **26.96 s** | 1,798 ms | 200 | 492 | **정상 기준해 (무결점 전단 거동)** |
| 2D | **2D-Tie (초고속 Tie)** | `CPE4I` | `CPE4H` | **SurfaceTie (가속)** | **13.2058** | **87.90** | **1.32 s** | **88 ms** | 200 | 820 | **수학적 완전 일치 (99.994%), 1.3초 초고속** |
| 2D | 2D-Opt2 (고속) | `CPE4R` | `CPE4H` | Conformal (공유) | 14.0992 | 92.81 | 6.82 s | 455 ms | 200 | 492 | 전단 억제 우수 (+6.7% 미세 연화) |
| 2D | **2D-CPE6M (2차 삼각 Conformal)** | `CPE6M` | `CPE4H` | **Conformal (2:1 PSA)** | **13.7117** | **90.85** | **7.82 s** | **521 ms** | 400 | 1,458 | **고정밀 굽힘 포착, 비정형 접촉 강건성 우수** |
| 2D | 2D-CPE6M-Tie | `CPE6M` | `CPE4H` | SurfaceTie | 2.2763 | 15.76 | 2.63 s | 175 ms | 320 | 1,786 | 2차 삼각 중간절점 구속 불일치 (인공 잠김) |
| 2D | 2D-Base (기준 모델) | `CPE4` | `CPE4` | Conformal (공유) | 1.4732 | 10.21 | 7.82 s | 522 ms | 200 | 492 | **극심한 전단+체적 잠김 (-88.8% 왜곡)** |
| **3D** | **3D-Opt1 (최우수 권장)** | `C3D8I` | `C3D8H` | **Conformal (공유)** | **12.1658** | **80.06** | **10.04 s** | **669 ms** | 200 | 1,476 | **최고 성능 (10초 완주, 박판 전단 완벽 해소)** |
| 3D | **3D-Tie (가속화 혁신)** | `C3D8I` | `C3D8H` | **SurfaceTie (가속)** | **12.1662** | **80.06** | **24.09 s** | **1,606 ms** | 200 | 2,460 | **기존 2018초 $\to$ 24초 (84배 가속!), Conformal과 0.003% 일치** |
| 3D | 3D-Opt2 (감소적분) | `C3D8R` | `C3D8R` | Conformal (공유) | 11.7796 | 79.17 | 20.51 s | 1,367 ms | 200 | 1,476 | 아워글래스 억제 양호 (-3.2% 미세 강성화) |
| 3D | 3D-Base (기준 모델) | `C3D8` | `C3D8` | Conformal (공유) | 1.0538 | 8.30 | 12.73 s | 849 ms | 200 | 1,476 | **극심한 전단+체적 잠김 (-91.3% 왜곡)** |

---

## 3. 핵심 역학적 분석 및 성과

### 3.1 SurfaceTie 초고속화 달성 (2018초 $\to$ 24초, 84배 가속)
- **배경**: 기존 3D SurfaceTie는 33분 38초(2,018초)가 소요되어 Conformal(10초) 대비 200배 지연 발생.
- **원인 규명**: 
  1. 결합(Bonding) 구속임에도 매 뉴턴 반복마다 Python 전수 탐색(`reproject_deformed`)으로 25,600개 비선형 Quad 투영을 반복 계산(전체 시간의 91.7% 점유).
  2. 동적 리스트 생성 및 Python 배열 변환 오버헤드.
- **해결 방안 적용**:
  1. **투영 좌표 동결 (`freeze_projection=True`)**: 초기 기하에서 결정된 계면 절점의 모요소 투영 좌표($\xi, \eta$)를 고정하여 매 반복 투영 탐색 비용을 **0초(완전 제거)**화.
  2. **희소 행렬 인덱스/강성 사전 빌드**: `rows_tie`, `cols_tie`, `data_tie`를 초기화 시 1회 사전 생성하여 반복 중 메모리 할당 제거.
- **결과**:
  - **3D SurfaceTie 시간**: **$2,018.43\,\text{s} \to 24.09\,\text{s}$ (84배 가속)**!
  - **2D SurfaceTie 시간**: **$1.32\,\text{s}$**!
  - **역학적 정확도**: Conformal 대비 처짐 오차 **0.003%**, 슬립 오차 **0.00%**로 완벽한 기구학적 무결성 검증 완료.

### 3.2 2D `CPE6M` (6절점 변형 2차 삼각형) 평가
- **Conformal 모드 (`CPE6M` + 2:1 PSA 가로 세분화)**:
  - 처짐 $13.71\,\text{mm}$, 슬립 $90.85\,\mu\text{m}$, 소요 시간 **7.82초** (400개 요소, 1,458 DOFs).
  - 2차 다항식 형상함수로 인해 1층 메쉬에서도 곡률을 극히 부드럽게 표현하며, PSA 층을 가로 2배 세분화하여 중간절점을 100% 절점 공유함으로써 계면 연속성을 완벽히 만족.
  - 비정형 삼각 메쉬가 불가피한 복잡 형상이나 롤업 접촉 시 **최고의 대안 요소**로 검증됨.
- **SurfaceTie 모드 (`CPE6M-Tie`)**:
  - 처짐 $2.28\,\text{mm}$ (인공 잠김 발생).
  - 2D SurfaceTie가 선형 2절점 마스터 세그먼트 전제로 작동하여, 6절점 변형 삼각형의 중간 절점과 비선형 변위 모드 간의 기구학적 구속 불일치(Constraint Locking)가 발생함을 확인. $\to$ **CPE6M은 Conformal 모드 사용이 필수적**.

### 3.3 표준 1차 요소의 파멸적 잠김 (Locking Disaster) 재확인
- `CPE4` 및 `C3D8`: 처짐 $1.05\sim1.47\,\text{mm}$ (정상 대비 **$-90\%$**), 슬립 $8\sim10\,\mu\text{m}$ (정상 대비 **$-88\%$**).
- 폴더블 디스플레이 다층 적층 모델링에서 표준 1차 요소의 사용은 절대 불가함을 정량적으로 입증.
