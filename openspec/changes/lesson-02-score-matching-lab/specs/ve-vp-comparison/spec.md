## ADDED Requirements

### Requirement: VE schedule for MNIST
Модуль сравнения SHALL реализовывать VE-расписание (§10 теории): геометрическая лестница σ с σ_max уровня NCSN (≈80 в шкале данных NCSN; для unit-variance нормализованного MNIST эквивалент SNR≈0 — σ_max≈5, дефолт лабы) до σ_min≈0.01, forward `x_t = x₀ + σ_t·ε` (аддитивный шум, сигнал не масштабируется). Выбор шкалы σ_max SHALL документироваться в README.

#### Scenario: VE forward is additive
- **WHEN** применяют VE-forward с σ_t
- **THEN** `x_t = x₀ + σ_t·ε`, без множителя при x₀

#### Scenario: Geometric sigma ladder
- **WHEN** строят VE-лестницу из L уровней
- **THEN** соседние σ отличаются на постоянный множитель (геометрическая прогрессия)

### Requirement: VP schedule matching lesson 1
Тот же модуль SHALL реализовывать VP-расписание как в лабе 1 (линейные β, `x_t = √ᾱ·x₀ + √(1−ᾱ)·ε`), чтобы сравнение было честным в одном коде.

#### Scenario: VP forward matches lesson-1 formula
- **WHEN** применяют VP-forward
- **THEN** результат равен `√ᾱ_t·x₀ + √(1−ᾱ_t)·ε`

### Requirement: Shared training loop, equal budget
Обучение VE и VP SHALL использовать один и тот же тренировочный цикл, архитектуру и бюджет (эпохи, lr, batch); различаться — только расписание и целевая параметризация. Скрипт SHALL обучать обе модели последовательно и сохранять обе.

#### Scenario: Equal training budgets
- **WHEN** сравнение запущено с `epochs=N`
- **THEN** обе модели обучаются ровно N эпох с одинаковыми остальными гиперпараметрами

### Requirement: Prediction targets per parameterization
VE-ветка SHALL обучать ε-предсказание с потерей, приведённой к score-виду через `score = −ε/σ` (§9 теории); VP-ветка — как в лабе 1. README SHALL объяснять связь `ε_θ ↔ score` обеих веток.

#### Scenario: Score-noise correspondence documented
- **WHEN** студент читает README раздел о связи score и шума
- **THEN** видит формулу `s_θ(x_t, t) = −ε_θ(x_t, t)/σ_t` и объяснение, почему это одна и та же сеть в разных параметризациях

### Requirement: Side-by-side samples and loss comparison
Скрипт сравнения SHALL генерировать side-by-side сетку сэмплов (VE слева, VP справа) из одинакового стартового шума и график сравнения loss-кривых; результаты сохраняются в `runs/` и (опционально) `images/` урока.

#### Scenario: Same starting noise for both
- **WHEN** генерируют сэмплы VE и VP
- **THEN** обе сетки стартуют из одного и того же x_T (общий seed) — сравнение честное

#### Scenario: Comparison artifacts written
- **WHEN** скрипт сравнения завершён
- **THEN** в `runs/` лежат PNG сэмплов обеих моделей и график loss-кривых
