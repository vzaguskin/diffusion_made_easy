## ADDED Requirements

### Requirement: MLP maps (x, sigma) to score
Модуль моделей SHALL предоставлять score-сеть: MLP `forward(x: Tensor[B,2], sigma: Tensor[B,1]) -> Tensor[B,2]`. Одна и та же сеть SHALL обслуживать все уровни шума (conditioning на σ), как в NCSN.

#### Scenario: Output shape matches input
- **WHEN** в сеть подают `x=[B,2]` и `sigma=[B,1]`
- **THEN** выход имеет форму `[B,2]`

#### Scenario: Same network, different sigmas
- **WHEN** одну и ту же пару `(x, σ₁)` и `(x, σ₂)` с σ₁ ≠ σ₂ пропускают через сеть
- **THEN** выходы различаются (сеть действительно обусловлена на σ)

### Requirement: Sigma conditioning via Fourier features
Уровень шума σ SHALL кодироваться Fourier-признаками его логарифма (sin/cos на геометрических частотах) перед подачей в MLP. Это позволяет одной сети работать с σ, меняющимися на порядки (лестница из §8 теории).

#### Scenario: Log-scale sensitivity
- **WHEN** σ меняется с 1.0 на 0.01
- **THEN** кодирование σ меняется существенно (обе области входного пространства сети различимы)

### Requirement: Compact architecture with documented defaults
Дефолтная сеть SHALL быть маленьким MLP (3–5 слоёв, ~128–256 нейронов, SiLU/ReLU) с числом параметров ≤ 1M. Гиперпараметры (ширина, глубина, число Fourier-признаков) настраиваются конфигом.

#### Scenario: Defaults trainable in minutes
- **WHEN** дефолтную сеть обучают на 2D-датасете
- **THEN** обучение полного пайплайна занимает минуты на ноутбучном GPU или CPU
