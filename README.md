# VAD Safety Guard: 종단간 자율주행을 위한 운동학적 안전 계층 (Kinematic Safety Layer)

##  개요 (Overview)
**VAD Safety Guard**는 VAD (Vectorized Autonomous Driving) 모델의 출력을 검증하기 위한 안전 모듈입니다.
기존의 종단간(End-to-End) 자율주행 모델은 시스템 지연(System Delay)이나 차량의 동역학적 한계를 고려하지 않고, 물리적으로 주행 불가능한(Infeasible) 경로를 생성하는 경우가 있습니다. 본 노드는 안전 필터(Safety Filter)로서 작동하여, 운동학적 제약 조건을 기반으로 후보 경로들을 검증하고, 위험 상황 시 능동적으로 개입하여 사고를 방지합니다.

### 주요 기능 (Key Features)
* **시스템 지연 보상 (Latency Compensation):** 안전성 검사 시 시스템 지연 시간($\tau$)을 수식에 반영.
* **운동학적 주행 가능성 검증 (Kinematic Feasibility Check):** 횡방향(곡률) 및 종방향(제동 거리) 안전성 검증.
* **능동적 개입 (Active Intervention):** 1순위 경로(Rank 0)가 위험할 경우 이를 차단하고, 안전한 하위 순위(Rank $N$) 경로를 재선택.
* **제어 연속성 보장 (Temporal Consistency):** 급격한 조향 변화를 방지하는 최적 경로 선정 알고리즘 적용.

---

##  핵심 알고리즘 (Methodology)

Safety Guard는 VAD 모델이 생성한 $N$개의 후보 경로 $\{T_0, T_1, ..., T_N\}$를 입력받아, 다음의 물리 모델을 기반으로 최적의 경로 $T^*$를 선택합니다.

### 1. 횡방향 안전성 (Lateral Safety)
차량의 횡방향 불안정(Oscillation) 및 전복을 방지하기 위해 경로의 곡률(Curvature)을 기반으로 횡방향 가속도($a_{lat}$)를 검증합니다.
경로상의 연속된 세 점 ($P_1, P_2, P_3$)에 대해 **멩거 곡률(Menger Curvature)** 공식을 사용하여 곡률 $\kappa$를 계산합니다.

$$
\kappa = \frac{4 \cdot \text{Area}(P_1, P_2, P_3)}{|P_1 P_2| \cdot |P_2 P_3| \cdot |P_3 P_1|}
$$

현재 속도 $v$에서의 예상 횡방향 가속도가 임계값($a_{lat}^{max}$)을 초과하면 해당 경로는 주행 불가(Infeasible)로 판정합니다.

$$
a_{lat} = v^2 \cdot \kappa \quad \text{s.t.} \quad a_{lat} \le a_{lat}^{max}
$$

### 2. 종방향 안전성 (Longitudinal Safety)
시스템 지연($\tau$) 상황에서의 충돌을 방지하기 위해 최소 필요 제동 거리($d_{req}$)를 계산합니다. 이는 지연 시간 동안의 공주 거리(Free-running distance)와 물리적 제동 거리를 합산한 값입니다.

$$
d_{req} = \underbrace{v \cdot \tau}_{\text{Reaction Distance}} + \underbrace{\frac{v^2}{2 \cdot |a_{long}^{max}|}}_{\text{Braking Distance}}
$$

생성된 정지 경로의 길이 $L_{path}$가 필요 제동 거리보다 짧을 경우, 충돌 위험이 있는 것으로 간주하여 주행 불가(Infeasible)로 판정합니다.

$$
\text{Condition: } L_{path} \ge d_{req}
$$

### 3. 최적 경로 선정 (Optimization Strategy)
안전성 검증을 통과한 경로 집합($S_{feasible}$) 중에서, 직전 제어 단계의 경로 곡률($\kappa_{prev}$)과 차이가 가장 적은 경로를 최종 선택하여 조향 제어의 연속성을 보장합니다.

$$
T^* = \underset{t \in S_{feasible}}{\mathrm{argmin}} | \bar{\kappa}(t) - \kappa_{prev} |
$$

만약 $S_{feasible} = \emptyset$ (모든 후보 경로가 위험함)일 경우, 시스템은 즉시 **비상 정지(Emergency Stop)** 경로를 생성합니다.

---

##  시스템 구조 (System Architecture)

```mermaid
graph TD
    A["VAD 모델 출력<br>(후보 경로 리스트)"] --> B{"주행 가능성 검증<br>(횡방향 & 종방향)"}
    B -- "불가 (Infeasible)" --> C["후보 탈락"]
    B -- "가능 (Feasible)" --> D["주행 가능 집합에 추가"]
    D --> E{"집합이 비었는가?"}
    E -- "Yes (위험)" --> F["비상 정지 생성<br>(최대 감속 제어)"]
    E -- "No (안전)" --> G["최적 경로 선정<br>(곡률 변화 최소화)"]
    F --> H["최종 제어 명령"]
    G --> H
