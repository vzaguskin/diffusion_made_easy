## ADDED Requirements

### Requirement: 2D toy datasets with sampling
Модуль данных SHALL предоставлять три 2D-датасета: `gaussians8` (кольцо из 8 гауссианов), `moons` (две полумесяца), `swiss-roll` (спираль). Каждый SHALL возвращать батч сэмплов `Tensor[B, 2]` по запросу. Датасет выбирается конфигом (`data.dataset`).

#### Scenario: Sample batch from each dataset
- **WHEN** запрашивают батч у каждого из трёх датасетов
- **THEN** каждый возвращает `Tensor[B, 2]` с конечными значениями в диапазоне данных (~[-3, 3])

#### Scenario: Dataset selected via config
- **WHEN** конфиг задаёт `data.dataset: moons`
- **THEN** используется датасет moons без изменения другого кода

### Requirement: Analytic log-density and score
Для каждого датасета модуль SHALL предоставлять аналитические `log_p(x)` и `score(x) = ∇log p(x)`. Для `gaussians8` — точная смесь гауссианов (§6 теории); для `moons` и `swiss-roll` — GMM-аппроксимация (смесь узких гауссианов вдоль кривой), что документировано в README.

#### Scenario: Score of a Gaussian mixture
- **WHEN** вычисляют `score(x)` для смеси гауссианов в точке x
- **THEN** результат равен `Σᵢ wᵢ Nᵢ(x)·(−(x−μᵢ)/σᵢ²) / Σᵢ wᵢ Nᵢ(x)` (взвешенное среднее пер-компонентных score)

#### Scenario: Autograd cross-check
- **WHEN** численный градиент `log_p(x)` (autograd) сравнивают с аналитическим `score(x)` на сетке точек
- **THEN** они совпадают с точностью не хуже 1e-4

### Requirement: GMM approximation is tight enough for visualization
GMM-аппроксимации для `moons`/`swiss-roll` SHALL использовать достаточно узкие компоненты (σ ≤ 0.1) и достаточно много компонент (≥50), чтобы визуально неотличимо покрывать данные; расхождение с данными SHALL быть видно только при σ-noise мельче ширины компонент.

#### Scenario: GMM density covers the data curve
- **WHEN** рисуют heatmap log p поверх сэмплов данных
- **THEN** высокая плотность проходит вдоль всей кривой данных, без видимых «дыр»
