## ADDED Requirements

### Requirement: Discrete Langevin sampler follows section 4 of theory
Модуль сэмплирования SHALL реализовывать дискретный Langevin dynamics (§4 теории): `x_{k+1} = x_k + (ε/2)·s(x_k) + √ε·z`, `z ~ N(0,I)`, начиная со стартовых точек (uniform по области данных или фиксированный seed). Сэмплер SHALL принимать любую функцию score `(x, σ) → score`.

#### Scenario: Trajectories follow the field
- **WHEN** Ланжевен запускают с *истинным* score GMM
- **THEN** финальные точки концентрируются в окрестности мод (не остаются в low-density регионах)

#### Scenario: Reproducible with fixed seed
- **WHEN** сэмплер запускают дважды с одинаковым seed стартовых точек и шума
- **THEN** траектории идентичны

### Requirement: Annealed Langevin walks the sigma ladder
Модуль SHALL реализовывать annealed Langevin (§8 теории): последовательность уровней σ_L=max … σ_1=min; на каждом уровне выполняется заданное число шагов Ланжевена, затем σ уменьшается и процесс продолжается с текущих позиций (не перезапускаясь). Внутри уровня шаг ε_i SHALL масштабироваться по σ_i (пропорционально σ_i²/σ_min² или эквивалентно).

#### Scenario: Ladder from sigma_max to sigma_min
- **WHEN** annealed Ланжевен настроен с лестницей [σ_max=1.0, …, σ_min=0.02], L=10
- **THEN** уровни проходятся в убывающем порядке, состояния переносятся между уровнями

#### Scenario: Step scales with level
- **WHEN** уровень σ уменьшается в 100 раз от первого к последнему
- **THEN** шаг Ланжевена на последнем уровне значительно меньше, чем на первом (масштабирование по σ)

### Requirement: Single-sigma failure is demonstrable
Лаба SHALL включать демонстрацию «одного σ» (§8: почему одного уровня недостаточно): Ланжевен только с σ_min на одномодовоподобном ландшафте застревает в локальной области / покрывает малую долю мод; annealed покрывает большинство.

#### Scenario: Mode coverage comparison
- **WHEN** сравнивают mode-coverage (доля мод GMM, куда попал хотя бы один сэмпл) single-σ vs annealed на `gaussians8`
- **THEN** annealed покрывает ≥90% мод, single-σ — заметно меньше (типично <50%)

### Requirement: Intermediate states returned for visualization
Оба сэмплера SHALL опционально возвращать траектории/снимки промежуточных состояний (для отрисовки прогресса annealing по уровням).

#### Scenario: Snapshot per level
- **WHEN** annealed Ланжевен запущен с `return_trajectory=True`
- **THEN** результат содержит снимки позиций на каждом уровне лестницы
