# ddpm-samplers Specification

## Purpose
TBD - created by archiving change lesson-01-ddpm-mnist-lab. Update Purpose after archive.
## Requirements
### Requirement: DDPM sampler follows section 17 of theory
Модуль `samplers/ddpm.py` SHALL реализовывать стохастический алгоритм генерации из раздела 17 теории: старт с `x_T ~ N(0, I)`, цикл `t = T, T-1, ..., 1`, на каждом шаге `ε_pred = model(x_t, t)`, затем `μ = (1/√α_t)·(x_t − (1-α_t)/√(1-ᾱ_t)·ε_pred)` и `x_{t-1} = μ + σ_t·z` с `z ~ N(0,I)` для `t > 1`; на последнем шаге `x_0 = μ` без шума.

#### Scenario: Last step adds no noise
- **WHEN** DDPM-сэмплер доходит до шага `t = 1`
- **THEN** финальный `x_0` равен `μ` (без добавления `σ_t·z`)

#### Scenario: Full 1000-step sampling produces an image
- **WHEN** запускают DDPM-сэмплер с `num_steps = 1000` на обученной модели
- **THEN** на выходе — тензор картинок формы `[N, 1, 28, 28]`

### Requirement: DDIM sampler follows section 18 of theory
Модуль `samplers/ddim.py` SHALL реализовывать детерминированный DDIM-сэмплер (раздел 18 теории): тот же forward-процесс и та же обучаемая модель, но не Марковский reverse; параметр `η` (уровень стохастичности) с дефолтом `η = 0` (чистый детерминизм). Сэмплер SHALL поддерживать число шагов `< T` (подмножество временных шагов) для ускорения генерации.

#### Scenario: Deterministic with eta=0
- **WHEN** DDIM-сэмплер запускают дважды с одинаковым стартовым шумом `x_T` и `η = 0`
- **THEN** результаты идентичны (до численной точности)

#### Scenario: Fewer steps than T
- **WHEN** запускают DDIM с `num_steps = 25` (при `T = 1000`)
- **THEN** сэмплер использует подмножество из 25 временных шагов и завершается быстрее, чем 1000-шаговый DDPM

### Requirement: Unified sampler interface
Каждый сэмплер SHALL иметь сигнатуру `sample(model, shape, num_steps, *, generator=None, **kwargs) -> Tensor`, возвращающую партию сгенерированных картинок. Одинаковый интерфейс позволяет использовать любой сэмплер в коллбэках визуализации и в `sample.py`.

#### Scenario: Swap sampler in visualization
- **WHEN** коллбэк визуализации переключают с DDPM на DDIM
- **THEN** остальной код (формирование shape, логирование в TB) не меняется

### Requirement: Reproducible sampling via optional generator
Сэмплеры SHALL принимать опциональный `torch.Generator` (или seed) для воспроизводимости стартового шума и промежуточных шумовых шагов. Это позволяет делать «фиксированный набор сэмплов» между эпохами для визуализации прогресса.

#### Scenario: Same seed gives same samples
- **WHEN** DDPM-сэмплер запускают дважды с одинаковым generator/seed на одной модели
- **THEN** сгенерированные картинки идентичны

