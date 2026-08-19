## Purpose

EDM-параметризация (Karras et al. 2022) для VE-ветки лабы 2: preconditioned denoiser, взвешенный лосс и Heun-сэмплер — лекарство от разбалансировки уровней σ, спеклов и острой вершины бюджетной кривой наивной ε-ветки.

## ADDED Requirements

### Requirement: Preconditioned denoiser

Модуль предоставляет обёртку над ε-сетью в форме `D_θ(x, σ) = c_skip(σ)·x + c_out(σ)·F_θ(c_in(σ)·x, σ)` с коэффициентами EDM: `c_in = 1/√(σ² + σ_data²)`, `c_out = σ·σ_data/√(σ² + σ_data²)`, `c_skip = σ_data²/(σ² + σ_data²)`, где `σ_data` — эмпирическая std данных (конфигурируется). Предельные случаи: при `σ → 0` `D_θ(x,σ) → x`, при `σ → ∞` выход остаётся ограниченным (не растёт как x/σ-ошибка наивной ветки).

#### Scenario: Limiting behaviour of the coefficients

- **WHEN** вычислены c_skip/c_out/c_in при σ → 0 и при большом σ
- **THEN** `c_skip(0) = 1`, `c_out(0) = 0` (тождественный denoiser), и `c_skip + c_out·(±1/σ)` остаётся ограниченным при σ → ∞

### Requirement: EDM-weighted training loss

Лосс: `λ(σ)·‖D_θ(x + σε, σ) − x‖²` с весом `λ(σ) = (σ² + σ_data²)/(σ·σ_data)²` и σ, сэмплируемым log-normal (медиана P_mean, ширина P_std из конфига), а не равномерно по лестнице. Эффективная цель имеет единичную дисперсию на всех σ.

#### Scenario: Unit-variance effective target across sigma

- **WHEN** σ пробегает от малых к большим и вычислена дисперсия взвешенной невязки аналитического denoiser гауссовских данных
- **THEN** она не зависит от σ (в пределах константы)

### Requirement: Deterministic Heun sampler on the sigma grid

Сэмплер интегрирует вероятностный поток ODE в σ-параметризации методом Хьюна (2 NFE/шаг, последний шаг — Euler), по сетке σ от σ_max до 0 с роем дискретизации из конфига; старт — `N(0, σ_max²·I)`. Возвращает также точный NFE.

#### Scenario: Matches PF-ODE at coarse resolution

- **WHEN** EDM-сэмплер и Euler на PF-ODE запущены с одной сеткой σ и одного старта
- **THEN** их траектории близки (Heun — уточнение того же поля), NFE = 2·(число интервалов) − 1

### Requirement: Comparison script against the naive VE branch

`train_edm.py` обучает EDM-ветку тем же бюджетом эпох, что ve_long, и сохраняет: лосс-кривую, пер-уровневый лосс обеих веток на общей оси, бюджетный стрип сэмплов 40–1000 NFE (те же пресеты, что у наивной ветки) и метрику спеклов. Все артефакты в `runs/edm/`.

#### Scenario: Budget strip regenerated for EDM

- **WHEN** скрипт завершён
- **THEN** в `runs/edm/` есть стрип с теми же 8 бюджетами и подписями NFE, сравнимый с наивным бок о бок
