# UL Formulation 도입 및 경계조건 디버깅 기록

**날짜**: 2026-07-01  
**주제**: U-bend 90° 수렴 실패 원인 분석 및 UL 공식 도입

---

## 배경

ex03_v6에서 RBE2로 wing 영역의 slave 노드를 묶어 90° U-bend를 구현하려 했으나,
θ ≈ 21° 부근에서 EAS Q4 요소의 det(F) < 0 → NaN 발산으로 실패.

---

## 원인 분석

### Total Lagrangian의 한계

- TL: 항상 초기 형상에서 F를 누적 계산
- 90° 접힘 시 요소가 크게 회전 → **det(F) < 0** (수치적 뒤집힘)
- EAS alpha 응축 발산 → NaN 전파

### Updated Lagrangian 도입 근거

- UL: 마지막 수렴 형상에서 F_inc = I + grad(u_inc) 계산
- 작은 증분에서는 det(F_inc) ≈ 1 → 항상 안정
- 각도가 아무리 커져도 증분이 작으면 요소 반전 없음

---

## 3점 벤딩 vs U-bend 비교 — 핵심 인사이트

| | 3점 벤딩 / 외팔보 | U-bend (RBE2) |
|---|---|---|
| 고정단 BC | 변위 = 0 | slave 노드 = 5.3mm (θ=45°) |
| 경계 요소 변위 구배 | ≈ 0 (미소) | ≈ 6/mm (거대) |
| EAS 수렴 | 안정 | NaN 발산 |

**핵심**: 3점 벤딩이 잘 되는 이유 — 고정단 변위가 0이라 slave/free 경계 구배가 없음.  
U-bend 실패 이유 — SLAVE_THRESH=10mm에서 slave 노드 변위(5.3mm) vs free 노드 변위(2-3mm) → 구배 ≈ 6/mm → NaN.

### 해결 방향: SLAVE_THRESH를 pivot 근처로 이동

- pivot은 xs = ±3mm 근처 (실제 접힘 중심)
- xs = ±3.5 근처에서 변위 ≈ 0mm → 구배 ≈ 0.7/mm (외팔보 고정단과 유사)

---

## 경계조건 적용 방식 오류 — 중요한 실수

### 원래 의도 (올바름)
```python
# j=0 (바닥 행)만 slave BC 적용
for i, x in enumerate(xs):
    if abs(x) > SLAVE_THRESH:
        nid = nid_map[(0, i)]   # j=0만!
        _add_slave(nid, PIVOT, side)
```

- Wing의 **물리적 장착면 = 바닥(j=0)** 만 rigid arm에 고정
- 내부 노드(j>0)는 재료 강성으로 자연스럽게 따라감
- 얇고 뻣뻣한 PET wing (두께/폭 비 ≈ 1/30) → 실질적 rigid body 거동

### 잘못된 구현 (ex03_v7_ul.py)
```python
# 전체 j행 slave — 잘못됨
for j in range(ny):
    for i, x in enumerate(xs):
        if abs(x) > SLAVE_THRESH:
            nid = nid_map[(j, i)]   # 모든 j!
```

**결과**: Wing 전체가 완전한 rigid body → xs=-10.5(slave) vs xs=-10.0(free) 경계에서  
모든 두께 방향 노드에 걸쳐 거대한 변위 불연속 → 더 심각한 NaN

### j=0만 잡아야 하는 이유

```
xs:  -10.5      -10.0      -9.5
j=3:  [ free ] --- [ free ]      ← 연속적 탄성 변형
j=2:  [ free ] --- [ free ]
j=1:  [ free ] --- [ free ]
j=0:  [slave] --- [ free ]       ← 바닥만 BC (물리적 의미)
```

- j>0 행: 경계 양쪽 모두 free → 탄성 연속 변형
- j=0만 불연속이지만 두께(0.017mm)가 극히 작아 국소적
- 외팔보 고정단과 동일한 원리

---

## 구현된 수정 사항 (dynamic.py)

### UL 상태 변수
```python
self._ul_F_n = np.tile(np.eye(2), (n_elem, 4, 1, 1))  # (n_elem, 4GP, 2, 2)
self._ul_u_ref = np.zeros(n_dofs)
```

### 퇴화 Jacobian 체크 (θ>45° 이후 UL 참조 형상도 나빠질 때)
```python
det_J_ref = _J11 * _J22 - _J12 * _J21
good_J = det_J_ref > 1e-4
# bad J → TL 좌표 fallback
coords_b_np = np.where(good_J[:, None, None], coords_ref, self.elem_coords[elem_indices])
```

### 수렴 후 F_n 동기화
```python
if self.ul_mode:
    _F_sync[:, :, 0, 0] = 1.0 + _gu[:, :, 0]
    # ... (전체 형상에서 I + grad(u_total) 재계산)
    self._ul_F_n[_finite] = _F_sync[_finite]
```

### COO 행렬 어셈블리 버그 수정
```python
# rows: np.repeat, cols: np.tile — 반드시 같은 길이
all_K_rows.append(np.repeat(good_dofs, 8, axis=1).reshape(-1))
all_K_cols.append(np.tile(good_dofs, (1, 8)).reshape(-1))
```

### plastic.py ZeroDivisionError 수정
```python
F_p_inv_3d[2, 2] = 1.0 / det_Fp_inv_2d if abs(det_Fp_inv_2d) > 1e-12 else 1.0
```

---

## 남은 과제

- [ ] ex03_v8: j=0만 slave BC + SLAVE_THRESH=3.5 + UL 조합으로 재시험
- [ ] 90°까지 수렴 확인
- [ ] 2D viewer (dispsolver/postprocess/viewer.py, 882줄 구현 완료) 연동 확인

---

## 교훈

1. **3점 벤딩이 되고 U-bend만 안 된다면** → BC 경계에서의 변위 구배 확인
2. **외팔보 고정단 = 변위 0** = NaN 없음. **Slave BC = 변위 크면** = NaN 위험
3. **j=0만 slave** = 원래 올바른 설계. 내부 노드까지 잡으면 과구속 + 경계 문제 악화
4. **UL 필수**: 90° 회전 시 TL det(F)<0 불가피. 증분 공식만이 해답.
