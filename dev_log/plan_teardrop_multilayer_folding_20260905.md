# [구현 계획서] PET-PSA 다층 적층 구조에서의 물방울형(Teardrop-shaped) 폴딩 시뮬레이션 및 검증 계획

---

## 1. 개요 및 배경 (Goal & Background)

사용자 요청:
> **"tear drop folding을 pet-psa- 반복의 적층 구조로 테스트하자. 기존의 정의된 구조를 참고하면 된다."**

앞서 단일 박판 유리(Thin Glass) 기판에서 검증 완료된 **물방울형(Teardrop) 폴딩 기구학**($x_{\text{pivot}}=0.8\text{ mm} \implies d_{\text{neck}}=1.6\text{ mm}$, $x_{\text{gap}}=7.5\text{ mm} \implies L_{\text{free}}=15.0\text{ mm}$)을 코드베이스에 기존 정의되어 있는 **PET-PSA 교대 반복 적층 구조(Laminate Stack)**에 적용하여, 다층 복합체에서의 물방울형 폴딩 거동과 층간 전단 슬립(Interlayer Shear Slip)을 정밀하게 테스트하고 검증하는 실행 계획입니다.

```
       [ 평탄 초기 상태 (t=0) ]
       ──────────────────────────────────────────────────
       PET 층 (3 mesh rows, 0.05 mm, E=4000 MPa, J2 소성)
       PSA 층 (1 mesh row,  0.03 mm, Arruda-Boyce 점탄성)  x N_pairs (3쌍 6층 또는 7쌍 14층)
       ──────────────────────────────────────────────────
       [Plate Left]   |←─── 15.0 mm 자유 힌지부 ───→|   [Plate Right]
                      | (하단 PSA 테이프 Void 처리) |

                                 ▼ 90° 폴딩 (180° 대면 밀착)

       [ 물방울형 폴딩 최종 상태 (t=1.0) ]
        [Plate L]   [Plate R]
        ║  PET-PSA  ║
        ║  (밀착)   ║   d_neck = 1.6 mm (거의 대면 밀착)
        ║           ║
         ╲         ╱    외측 팽창 전이부 (Flared Neck)
          ╲       ╱
           (     )      W_max ≈ 3.7 mm (물방울 바디, R_eff ≈ 1.85 mm)
            ╰───╯       원형/타원형 팁 (Smooth Bend Apex)
```

---

## 2. 기존 정의된 PET-PSA 적층 구조 및 물성 사양

기존 `dispsolver/fold_model_config.py` 및 `examples/gen_ex12_inp.py`에 정의된 구조를 그대로 계승합니다:

### 1) 적층 기하학 (Stackup Geometry)
- **반복 단위(Layer Pattern):**
  - **PSA 층**: 두께 $0.03\text{ mm}$ (30 µm), 1 mesh row
  - **PET 층**: 두께 $0.05\text{ mm}$ (50 µm), 3 mesh rows
  - 단위 쌍당 두께: $0.08\text{ mm}$ (80 µm)
- **적층 옵션:**
  - **기본(3쌍, 6개 물리 계층):** 총 두께 $0.24\text{ mm}$ (240 µm), 두께 방향 12개 요소 분할
  - **확장(7쌍, 14개 물리 계층):** 총 두께 $0.56\text{ mm}$ (560 µm), 두께 방향 28개 요소 분할
- **테이프 절개부 (Bottom Tape Void):**
  - 힌지 자유 구간($|x| < x_{\text{gap}} = 7.5\text{ mm}$)에서 최하단 Layer 1 PSA 테이프는 void(요소 생성 안 함) 처리되어 디스플레이가 힌지부에서 자유롭게 거동함.

### 2) 재료 구성 법칙 (Constitutive Laws)
- **PET (Substrate):**
  - 모델: 대변형 $J_2$ 탄-소성 (`J2Plasticity`)
  - 물성: $E = 4000.0\text{ MPa}$, $\nu = 0.3$, 항복응력 $\sigma_{y0} = 80.0\text{ MPa}$, 경화계수 $H = 400.0\text{ MPa}$
  - 요소: `Q4_COROTATIONAL_SRI` (대변형 전단 잠금 완전 억제)
- **PSA (Pressure Sensitive Adhesive):**
  - 모델: `ArrudaBoyce` 초탄성 + `ViscoelasticMaterial` (Prony 2항 점탄성 이완 + WLF 온도 시프트)
  - 물성: $\mu = 0.016779\text{ MPa}$, $\lambda_m = 3.0$, $K = 0.83333\text{ MPa}$
  - Prony: $g_1 = 0.12, g_2 = 0.08$, $\tau_1 = 0.7\text{ s}, \tau_2 = 7.0\text{ s}$
  - 요소: `Q4_UP` (Flory F-bar 체적 잠금 제어 + Simo 유한변형 점탄성 오버스트레스 적분)

---

## 3. 핵심 수치 안정성 전략 (Critical Numerical Invariant)

> [!IMPORTANT]
> **시간 적분 최대 증분 제한 (`dt_max = 0.01s`) 필수 복원**
>
> 30 µm 두께의 극박 PSA 점착층은 $E \approx 0.05\text{ MPa}$ 수준으로 PET($4000\text{ MPa}$) 대비 80,000배 유연합니다.
> 시간 증분이 $dt \ge 0.05\text{ s}$ ($\Delta\theta \ge 5^\circ$)로 커질 경우, 1회 Newton 시도에서 요소 변형 구배 $\det(F)$가 음수로 전도(`det(F) < 0` $\to$ $\ln(J) \to \text{NaN}$)되어 선형 탐색 실패를 유발합니다.
> `dt_max = 0.01\text{ s}` ($\Delta\theta \le 0.9^\circ$)로 엄격히 제한하면:
> 1. 모든 증분에서 $\det(F) > 0$ 유지
> 2. 증분당 Newton 반복수 $1 \sim 2$회로 극도로 안정된 2차 수렴성 유지
> 3. Cutback 0회로 90도 완전 폴딩 완주 보장 (과거 `ex12` 검증 완료 기록과 일치)

---

## 4. 제안하는 코드 변경 내역 (Proposed Code Changes)

### 1) [MODIFY] `dispsolver/fold_model_config.py`
- `make_teardrop_config` 함수를 확장하여 `substrate: str = "multilayer" | "glass"` 매개변수 지원:
  - `substrate="multilayer"`: 기존 정의된 3쌍/7쌍 PET-PSA 적층 패턴, 테이프 void, F-bar/Corotational 요소 매핑 자동 구성.
  - `substrate="glass"`: 단일 모놀리식 박판 유리 substrate 구성.
- `DriveConfig.dt_max` 기본값을 `0.01`로 안전하게 복원:
  ```python
  @dataclass
  class DriveConfig:
      theta_max_deg: float = 90.0
      t_total: float = 1.0
      dt_init: float = 0.005
      dt_max: float = 0.01   # 0.1 -> 0.01: 점탄성 PSA 층 체적 반전 방지
      dt_min: float = 1e-5
  ```

### 2) [MODIFY] `examples/ex13_unified_model_io.py`
- CLI 인자 연동 강화:
  - `--stack {multilayer, glass}` (기본값: `multilayer`)
  - `--fold_shape {teardrop, u_shape}` (기본값: `teardrop`)
- `--stack multilayer --fold_shape teardrop` 실행 시:
  - $x_{\text{pivot}} = 0.8\text{ mm}$, $x_{\text{gap}} = 7.5\text{ mm}$, $L_{\text{free}} = 15.0\text{ mm}$
  - `layer_void_regions = {1: [(-7.5, 0.0), (0.0, 7.5)]}`
  - `dt_max = 0.01s`, PARDISO SPD 자동 매핑
- 해석 완료 후 콘솔 및 결과 객체에 층간 전단 슬립 통계 및 물방울 형상비 요약 출력.

### 3) [NEW] `examples/verify_teardrop_multilayer.py`
- PET-PSA 다층 적층체의 물방울형 폴딩 전용 분석/시각화 스크립트:
  1. **(a) 전체 및 힌지부 변형 메쉬 시각화**: PET(파랑)와 PSA(빨강)의 층별 변형 형상 및 물방울 루프 표시.
  2. **(b) 층간 전단 슬립(Interlayer Shear Slip) 분포 플롯**: 자유단($x = \pm 40\text{ mm}$) 및 힌지 엣지($x = \pm 7.5\text{ mm}$)에서의 계단식 슬립량(Staircase profile) 정량 측정.
  3. **(c) 검증 요약 표 및 아티팩트 이미지 생성**: `examples/ex13_teardrop_multilayer_shape.png` 저장.

### 4) [NEW] `tests/test_teardrop_multilayer.py`
- pytest 자동화 테스트 케이스:
  - `test_teardrop_multilayer_convergence()`: PET-PSA 다층 모델 90° 완주 및 `reached_target == True` 검증.
  - `test_teardrop_multilayer_kinematics()`: 목 부위 닫힘 갭 $d_{\text{neck}} \le 2.0\text{ mm}$, 외측 루프 폭 $W_{\text{max}} \ge 3.5\text{ mm}$ 검증.
  - `test_teardrop_interlayer_shear()`: 전체 층간 전단 슬립 중 PSA 점착층이 $90\%$ 이상을 분담하는지 물리적 타당성 검증.

---

## 5. 검증 계획 (Verification Plan)

### Automated Tests
1. **신규 적층 물방울 단위 테스트:**
   ```powershell
   pytest tests/test_teardrop_multilayer.py -v
   ```
2. **기존 통합 테스트 회귀 방지:**
   ```powershell
   pytest tests/test_pardiso_options.py tests/test_two_point_bending.py tests/test_convergence_fixes.py tests/test_rigid_plate_tie.py -q
   ```
3. **코어 FEA 벤치마크 (11개 PASS 검증):**
   ```powershell
   python -m verification.run_all
   ```

### Manual Verification
1. **ex13 다층 물방울 폴딩 직접 실행:**
   ```powershell
   python -u examples/ex13_unified_model_io.py --mode build --stack multilayer --fold_shape teardrop
   ```
2. **결과 시각화 확인:**
   - `examples/ex13_teardrop_multilayer_shape.png`에서:
     - 플레이트가 밀착되어 닫히는 형상 확인 ($d_{\text{neck}} \approx 1.6\text{ mm}$).
     - 힌지부 디스플레이가 외측으로 팽창하여 매끄러운 물방울(Teardrop) 루프를 형성하는지 육안 검증.
     - PSA 층이 전단 변형을 집중적으로 흡수하는지 확인.
