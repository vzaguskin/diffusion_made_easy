## ADDED Requirements

### Requirement: Linear noise schedule matching theory
Модуль DDPM-ядра SHALL реализовывать линейное расписание шума: `β_t` растёт линейно от `β_start` до `β_end` для `t = 1..T`, с дефолтами `β_start = 1e-4`, `β_end = 0.02`, `T = 1000` (раздел 3 теории). Значения `T`, `β_start`, `β_end` SHALL быть настраиваемыми через конфиг.

#### Scenario: Default schedule values
- **WHEN** `DiffusionCore` создан с дефолтным конфигом
- **THEN** `betas[0] ≈ 1e-4`, `betas[T-1] ≈ 0.02`, `len(betas) == 1000`

#### Scenario: Custom schedule via config
- **WHEN** конфиг задаёт `diffusion.num_timesteps: 500`
- **THEN** `DiffusionCore` создаётся с `T = 500` без ошибок

### Requirement: Precompute α / ᾱ / √ᾱ / √(1-ᾱ)
Ядро SHALL precompute и хранить на device тензоры: `alphas = 1 - betas`, `alphas_cumprod = cumprod(alphas)`, `sqrt_alphas_cumprod`, `sqrt_one_minus_alphas_cumprod`. Эти тензоры используются в `q_sample` и сэмплерах без пересчёта на каждый шаг.

#### Scenario: Precomputed tensors on device
- **WHEN** `DiffusionCore` создан с `device="cuda"`
- **THEN** все precomputed тензоры (`alphas_cumprod` и т.д.) лежат на `cuda`

### Requirement: Forward process q_sample in closed form
Метод `q_sample(x0, t, noise=None)` SHALL реализовывать «прыжок через все шаги» (раздел 4 теории): `x_t = √ᾱ_t·x₀ + √(1-ᾱ_t)·ε`, где `ε ~ N(0, I)` если `noise` не передан. Реализация SHALL принимать batch `x0` и тензор индексов `t` (по одному `t` на сэмпл) и возвращать зашумлённые `x_t`.

#### Scenario: Forward with provided noise is deterministic
- **WHEN** вызывают `q_sample(x0, t, noise=fixed_noise)`
- **THEN** результат детерминирован и равен `√ᾱ_t·x0 + √(1-ᾱ_t)·fixed_noise`

#### Scenario: Forward without noise samples N(0,I)
- **WHEN** вызывают `q_sample(x0, t)` дважды с одним `x0`, `t`
- **THEN** результаты разные (шум случаен), оба имеют ожидаемое распределение

### Requirement: Variance-preserving property
Расписание и формула `q_sample` SHALL сохранять дисперсию (variance-preserving, раздел 3 теории): при `Var(x0) ≈ 1` для большого `T` распределение `x_T` близко к `N(0, I)`.

#### Scenario: x_T approximately standard normal
- **WHEN** берут партию MNIST, нормализованную к единичной дисперсии, и применяют `q_sample(x0, t=T)`
- **THEN** эмпирическая дисперсия результата близка к 1 (в пределах численной точности)

### Requirement: Loss is unweighted MSE on predicted noise
Ядро SHALL предоставлять метод `compute_loss(eps_pred, eps_target)`, возвращающий средний MSE `mean((eps_pred - eps_target)²)` без взвешивания по `t` (раздел 15 теории: «отбрасывание веса»). Цель прогноза — шум `ε`, не `x₀` и не `μ`.

#### Scenario: Loss matches manual MSE
- **WHEN** `eps_pred` и `eps_target` — известные тензоры
- **THEN** `compute_loss(eps_pred, eps_target)` равен `torch.mean((eps_pred - eps_target)**2)`

### Requirement: Predict x₀ from noise and x_t
Метод `predict_start_from_noise(xt, t, noise)` SHALL реализовывать `x₀ = (x_t − √(1-ᾱ_t)·ε) / √ᾱ_t` (раздел 14 теории). Используется сэмплерами для вывода `x₀` из предсказанного сетью шума.

#### Scenario: Inverse of q_sample
- **WHEN** `xt = q_sample(x0, t, noise=ε)` и затем `x0_rec = predict_start_from_noise(xt, t, ε)`
- **THEN** `x0_rec` приближённо равен `x0` (с численной погрешностью)
