# [Implementation Plan] 힌지(Hinge) 단면 반력 및 굽힘 모멘트 계산 정밀도 평가 계획서

## 1. Goal Description (목적 및 배경)
폴더블 디스플레이 90°(합산 180°) 굽힘 해석(`ex12`)에서 외형 실루엣(Kinematic Silhouette)이 U자 반원을 형성하더라도, 전단/체적 잠김이나 좌표계 오류가 있으면 모멘트가 수 배 폭등하거나 정적 평형이 깨질 수 있습니다.
따라서 **힌지 영역($x \in [-7.5, 7.5]$) 및 구동 피벗에서의 반력($RF_x, RF_y$)과 굽힘 모멘트($M_z$)가 물리적/수학적으로 정확하게 계산되었는지 4중 역학 검증 체계(4-Tier Mechanics Validation)를 통해 평가**하는 진단 툴과 정량적 검증 기준을 수립합니다.

---

## 2. 4단계 정밀 역학 평가 체계 (4-Tier Mechanics Evaluation Protocol)

1. **Tier 1: 구동 피벗(RBE2 Master) 반력 및 모멘트 대칭성**
   - $M_L = RF_{\theta, L}, \quad M_R = RF_{\theta, R}$
   - 대칭 오차: $\frac{| |M_L| - |M_R| |}{\frac{1}{2}(|M_L| + |M_R|)} < 1.0 \times 10^{-3}$ ($0.1\%$ 이내)
   - 수직 반력 상쇄: $|RF_{y, L} + RF_{y, R}| < 1.0 \times 10^{-4}\,\text{N}$

2. **Tier 2: 힌지 중앙($x=0$) 단면 응력 적분 (변형된 현재 좌표계 기준)**
   - $N_x(x=0) = \sum_{e \in \text{cut}} \int_{e} \sigma_{xx} \, dy$
   - $V_y(x=0) = \sum_{e \in \text{cut}} \int_{e} \sigma_{xy} \, dy \approx 0$ (대칭성 검증)
   - $M_z(x=0) = \sum_{e \in \text{cut}} \int_{e} \sigma_{xx} (y - y_c) \, dy$

3. **Tier 3: 자유물체도(Free Body Diagram) 모멘트 일치성**
   - $\sum M_{\text{pivot}} = M_{\text{drive}, R} - M_{z, x=0} - N_x \cdot (y_c - y_{\text{pivot}}) + RF_{y, R} \cdot (x_{\text{pivot}} - x_0) = 0$
   - 오차 $< 0.5\%$ 검증.

4. **Tier 4: 체적 잠김(Volumetric Locking) 감별 및 적층 보 슬립 이론 대조**
   - 완전 고착 시 $EI_{\text{noslip}}$ vs 완전 슬립 시 $EI_{\text{fullslip}}$
   - 수치해석 $M_z$가 정상 슬립 범위($[0.15, 0.35]\,\text{N}\cdot\text{mm}$) 내에 존재하는지 대조.

---

## 3. 평가 도구 구현 (`examples/eval_hinge_reactions.py`)

기존 결과 파일(`examples/ex12_result.pkl`)을 로드하여 위 4개 Tier를 한 번에 검증하고 그래프와 리포트를 출력하는 독립 검증 모듈을 작성 및 실행합니다.
