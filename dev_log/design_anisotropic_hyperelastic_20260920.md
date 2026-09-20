# 설계 K: 이방성 초탄성 HGO (Wave3 구현용)

## 1. 범위
- Holzapfel-Gasser-Ogden (동맥류 2-섬유족) + GOH 분산 수정. Abaqus `*HYPERELASTIC, ANISOTROPIC`.
- 소성 이방성은 Hill48 완료 (`plasticity_hill.py`) — 본 설계는 초탄성만.

## 2. 현행 (repo)
- 등방 코어: Neo-Hookean/Mooney/Ogden/AB/Yeoh (JAX `base.MaterialModel` + Ogden Numba).
- 불변량 헬퍼 `_invariants_from_C_flat` (I1/I1b/J) — I4/I6 없음. 섬유방향 개념 없음.

## 3. 정식화 (구현 시)
- `W = W_iso(I1b) + Σ_{i=4,6} (k1/2k2)·[exp(k2·E_i²)-1]`, `E_i = κ·I1b + (1-3κ)·I_i - 1`,
  `I_i = a_i·C·a_i` (κ=0: HGO, κ>0: GOH 분산).
- 1차 구현은 JAX 경로 (`_strain_energy_from_C_flat` 확장 + 섬유방향 params) — autodiff가
  접선 자동생성, §4.3 eigh 회피 (불변량 기반이라 spectral 분해 불필요).
- Numba 경로는 J-후순위 (PSA/PET 주력 아님).
- SDV 0 (초탄성 무상태). params: `{k1, k2, kappa, a4[3], a6[3], K}` + 등방 기저.
- `E_i < 0` → 섬유좌굴: 해당 항 0 (if-분기, autodiff 불연속점은 접선 FD 게이트로 감시).

## 4. 게이트 (구현 후)
- κ=0 + k1→0 퇴화 → 등방 기저와 일치 (1e-12).
- 등축인장 + 섬유방향 단축: 해석해 (HGO 원전식) 대조.
- C8 (FD Jacobian) + C6 (회전 AR sweep — 섬유방향 객관성) + 등방 스위트 회귀.
