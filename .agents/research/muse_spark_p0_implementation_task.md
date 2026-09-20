[미션: Abaqus 호환 P0 열-역학 연계 (Thermal-Mechanical Coupling) 구현]
역할: Muse Spark 1.3 Contributor (수석 구현 엔지니어)
참조 문서:
- `.agents/plans/material_master_plan_20260919_team_ABC.md`
- `C:\Users\GOODMAN\.gemini\antigravity-cli\brain\d68310e6-08c9-4aba-8b3d-e5f9258b5941\p0_thermal_mechanical_implementation_plan.md`

[구현 목표]
1. `dispsolver/material/dependence_table.py` 신규 작성:
   - Abaqus `*DEPENDENCIES` 순수함수 테이블 모듈
   - `eval_table_1d_numba(table_x, table_y, x_val)`: 1D 선형 보간 (경계 밖 clamp)
   - `eval_table_2d_numba(grid_x, grid_y, table_val, x_val, y_val)`: 2D 바이리니어 보간
   - Numba `@njit(fastmath=True, cache=True)` 지원

2. `tests/test_dependence_table.py` 신규 작성:
   - 노드점 exact 일치 검증 (< 1e-15)
   - 중간점 보간 정밀도 (< 1e-12)
   - 경계 외삽 clamp 검증

3. `dispsolver/model/material.py` 확장:
   - `Expansion` 클래스 추가: `Expansion(type="ISO", table=((alpha, temp), ...), zero=0.0)`
   - `Material.Elastic`, `Material.Plastic`에 temperature 테이블 지원 연동
   - `Material.to_numba_props()`에 직렬화 매핑 연동

4. 엄밀한 무결성 원칙 준수 (Anti-Mocking):
   - 난수/더미 행렬 금지, 이론 정해 완벽 부합.
