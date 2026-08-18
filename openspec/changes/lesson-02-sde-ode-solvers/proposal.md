## Why

Теория урока 2 (§11–15) доводит историю до непрерывного времени: Wiener process → forward SDE → reverse SDE (Anderson, 1982) → probability flow ODE → солверы (Euler-Maruyama, RK, DPM-Solver, NFE). Но лаба 2 заканчивается на дискретных вещах: annealed Langevin в 2D и наивные Euler+corrector сэмплеры VE на MNIST. Разрыв «теория говорит про SDE/ODE — код их не содержит» особенно ощутим после разбора сэмплера VE: мы вручную подобрали корректор, чтобы облако отслеживало расписание σ, хотя правильная постановка — интегрировать reverse SDE / PF-ODE с измеримым NFE.

Лаба должна дать руками то, что §15 обещает на бумаге: одни и те же обученные сети, разные численные методы, измеримые время (NFE и wall-clock) и качество.

## What Changes

- **Новый модуль `src/score_lab/sde.py`**: непрерывные расписания VE и VP в общей форме `dx = f(x,t)dt + g(t)dW` (§12), score через ε-модель (`s = −ε/std шума`, §9), reverse SDE drift (§13) и PF-ODE drift (§14) — единый интерфейс `drift(x, t)` для обоих.
- **Новый модуль `src/score_lab/solvers.py`**:
  - **Euler-Maruyama** для reverse SDE (стохастический, §15);
  - **Euler** и **Heun (RK2)** для PF-ODE (детерминированные, §15);
  - опционально RK4 — показать, что точность растёт, но NFE ×2;
  - все солверы считают NFE и wall-clock время.
- **Новый скрипт `scripts/compare_solvers.py`**: грузит уже обученные `model_ve.pt`/`model_vp.pt` (из `runs/ve_vs_vp/`, обучение не повторяется), сэмплирует одним стартовым шумом каждым методом × каждую ветку (VE/VP × SDE/ODE × Euler/Heun/+RK4), сохраняет сетки сэмплов, таблицу CSV (метод, ветка, NFE, время, метрика) и итоговый график quality-vs-NFE.
- **Качество**: визуальные сетки + простая количественная метрика без внешних зависимостей — FID не влезает в принцип «минимум зависимостей», поэтому связка из (a) per-level ε-MSE сэмплов по σ-лестнице и (b) sharpness/спекл-метрик из текущих диагностик; README честно объясняет ограничения.
- **README лабы**: новый раздел «SDE/ODE-солверы» с ожиданиями, чтением графиков и картой «раздел теории → код» для §11–15.
- **Тесты**: обратимость drift-ов (PF-ODE обоих веток воспроизводит маргиналы по σ-лестнице на гауссовских данных), согласованность NFE-счётчиков, детерминизм ODE-солверов при фиксированном старте.

## Capabilities

### New Capabilities
- `sde-formulations`: непрерывные VE/VP расписания в форме SDE `dx = f dt + g dW`; reverse-SDE и PF-ODE drift из одной ε-модели; время `t ∈ [0,1]` непрерывное.
- `numerical-solvers`: Euler-Maruyama (SDE), Euler/Heun/RK4 (ODE) с общим интерфейсом, счётчиком NFE и wall-clock замером.
- `solver-benchmark`: скрипт сравнения методов на обученных моделях лабы: сетки сэмплов, CSV с NFE/временем/качеством, график quality-vs-NFE.

### Modified Capabilities
- `lab2-entrypoints`: **MODIFIED** — добавляется `compare_solvers.py` и раздел README о SDE/ODE-части (регенерация картинок, ожидания по времени).
- `lab2-config`: **MODIFIED** — новая секция `solvers` (методы, шаги/NFE-бюджет, seed, метрики).

## Impact

- **Новые файлы**: `lessons/02-score-matching/lab/src/score_lab/sde.py`, `.../solvers.py`, `lessons/02-score-matching/lab/scripts/compare_solvers.py`, тесты в `tests/`.
- **Модифицируются**: `configs/default.yaml` (секция `solvers`), `README.md` лабы, возможно `mnist_ve_vp.py` (переиспользование загрузки моделей).
- **Зависимости**: без новых — torch/matplotlib/numpy/omegaconf.
- **Модели переиспользуются**: обучение не повторяется, `compare_solvers.py` требует предварительного `compare_ve_vp.py`.
- **Артефакты**: `runs/ve_vs_vp/solvers/` (сетки, `solver_benchmark.csv`, `quality_vs_nfe.png`) + картинки в `images/` урока.
