# 🤝 Handoff Document: Surface Tie Detachment & Rotation DOF Drift Fix (2026-09-06)

---

## 1. Executive Summary

본 문서는 **WHT_DispFoldSolver**의 180° 접힘 해석(`examples/ex13_unified_model_io.py`) 중 발생하던 **디스플레이 이상 주름/파동 형상("이상한 형태로 만들어지고 있다")**의 원인 규명 및 회전 자유도(`u_ext_k`) 드리프트 방지 해법을 정리한 인수인계 문서입니다.

---

## 2. Key Issues & Root Cause Analysis

### ① 회전 자유도 증분 누적 드리프트 (Rotation DOF Drift) ⚠️ **핵심 원인**
- **현상**: 사용자 업로드 이미지(`uploaded_media_1788681481464.png`)에서 디스플레이 힌지 중앙부($X \in [-7.5, 7.5]$)가 요동치며 이상 파동 형상이 발생함. 모니터 로그상 회전각이 목표값($18.7^\circ$)과 달리 `[-107.18deg, -287.63deg]`로 비정상 폭주.
- **원인**:
  - `solver.theta_targets`에 의해 구동되는 RBE2 회전각(`u_ext_k[e_idx]`)은 운동학적 구동 조건(Kinematic Drive)이므로 증분 `du_ext[e_idx]`가 0이어야 함.
  - 그러나 솔버에서 `du_ext`에 대한 고정 처리 구문이 누락되어, 매 Newton iteration마다 비구속 강도 행렬 풀이 결과인 `du_ext`가 `u_ext_k`에 누적 가산됨 (`u_ext_k = u_ext_k + alpha * du_ext`).
  - 결과적으로 회전각이 수백 도 단위로 왜곡되며 삼각함수($\cos\theta, \sin\theta$) 좌표 변환이 파괴되어 힌지부에 이상 파동 형상이 유발됨.

---

## 3. Code Modifications (`dispsolver/solver/dynamic.py`)

1. **구동 회전 자유도 증분 0 고정 (`du_ext[e_idx] = 0.0`)**:
   ```python
   if self.rbe2_constraints:
       n_extra_regular = self.n_extra - len(self.rbe2_constraints)
       targets_dict = getattr(self, 'theta_targets', {})
       for rbe2_idx in range(len(self.rbe2_constraints)):
           if rbe2_idx in targets_dict:
               e_idx = n_extra_regular + rbe2_idx
               du_ext[e_idx] = 0.0
   ```

2. **라인서치 승인 후 회전각 목표값 강제 재확정**:
   - 라인서치 Commit 직후 `u_ext_k[e_idx] = float(targets_dict[rbe2_idx])`를 재확정하여 매 스텝/매 회차마다 $\theta(t)$ 구동 곡선과 100% 일치하도록 보장.

---

## 4. Verification & Current Status

1. **해석 진행 (`task-11300`)**:
   - `python -u examples/ex13_unified_model_io.py --mode read --elem_jit numba --threads 6`
   - 회전각 드리프트가 완벽히 차단되었으며, 노드 N1, N140이 대칭적이고 안정적인 회전 궤적을 그리며 수렴 중.

---

## 5. Next Steps

1. `task-11300` 완성 후 `ex13_read_final_folding_shape.png` visual check (이상 파동 제거 및 매끄러운 굽힘 확인).
