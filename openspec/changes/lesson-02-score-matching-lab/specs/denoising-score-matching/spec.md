## ADDED Requirements

### Requirement: DSM loss follows section 7 of theory
Модуль обучения SHALL реализовывать denoising score matching (§7 теории): `L = E_{x~data, ε~N(0,I)}[‖s_θ(x + σ·ε, σ) + ε/σ‖²]`. Цель регрессии — `−ε/σ` (равна `∇log q(x̃|x)`), НЕ `+ε/σ` и НЕ `ε`.

#### Scenario: Loss target sign
- **WHEN** DSM-loss вычисляют с известными `s_θ`, `x`, `ε`, `σ`
- **THEN** он равен `mean((s_θ(x+σε,σ) + ε/σ)²)` — с плюсом перед целью

#### Scenario: Minimizer matches smoothed score
- **WHEN** сеть обучена до сходимости на одном фиксированном σ на GMM-датасете
- **THEN** её предсказания близки к аналитическому `∇log p_σ` (score сглаженной плотности) на сетке точек

### Requirement: Multi-sigma training with uniform level sampling
Обучение SHALL сэмплировать уровень σ равномерно из лестницы (§8 теории: многоуровневый шум) для каждого мини-батча, обучая одну сеть всем уровням одновременно.

#### Scenario: All levels trained together
- **WHEN** обучение идёт с лестницей из L уровней
- **THEN** в каждом батче уровни σ выбираются случайно (равномерно) из всех L

### Requirement: VE noise parameterization
Зашумление в DSM SHALL использовать VE-параметризацию (§10 теории): `x̃ = x + σ·ε` — аддитивно, без масштабирования сигнала (в отличие от VP `√(1−β)x + √β ε`).

#### Scenario: Noisy sample statistics
- **WHEN** применяют зашумление с σ=0.5 к батчу данных
- **THEN** зашумлённые сэмплы = данные + N(0, σ²I) (сигнал не масштабируется)
