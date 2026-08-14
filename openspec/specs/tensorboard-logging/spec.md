# tensorboard-logging Specification

## Purpose
TBD - created by archiving change lesson-01-ddpm-mnist-lab. Update Purpose after archive.
## Requirements
### Requirement: Scalar logging for losses and metrics
Пайплайн SHALL логировать в TensorBoard скаляры: `train/loss`, `val/loss`, bucketed losses, coverage, lr, и другие метрики. Логирование SHALL использовать встроенную интеграцию TensorBoard Lightning (`logger="tensorboard"`).

#### Scenario: Loss curves visible in TensorBoard
- **WHЕН** после нескольких эпох обучения открывают TensorBoard на `lab/runs/`
- **THEN** видны графики `train/loss` и `val/loss` по шагам/эпохам

### Requirement: Sample visualization callback each epoch
Пайплайн SHALL иметь коллбэк, который на каждой эпохе валидации (или с настраиваемой частотой `callbacks.sample_freq`) генерирует партию картинок и логирует их в TensorBoard как изображение (`add_image` с сеткой `make_grid`). Стартовый шум для этих сэмплов SHALL быть фиксирован между эпохами (фиксированный seed), чтобы прогресс генерации был наглядным.

#### Scenario: Samples logged every epoch
- **WHEN** завершается эпоха валидации
- **THEN** в TensorBoard появляется сетка сгенерированных картинок (тег, например, `samples/ddim`)

#### Scenario: Fixed noise across epochs
- **WHEN** сравнивают сэмплы эпохи 1 и эпохи 5
- **THEN** они происходят из одного и того же стартового `x_T` (одинаковый seed) — виден прогресс денуазинга

### Requirement: Configurable number and frequency of samples
Коллбэк визуализации SHALL брать из конфига число сэмплов (`callbacks.num_samples`, дефолт 64), частоту (`callbacks.sample_freq`, дефолт 1 = каждая эпоха), и используемый сэмплер (`callbacks.sampler`, дефолт `ddim` для скорости).

#### Scenario: Custom sample count
- **WHEN** конфиг задаёт `callbacks.num_samples: 16`
- **THEN** коллбэк генерирует и логирует сетку из 16 картинок

### Requirement: DDPM vs DDIM comparison logged
На финальной генерации (или по запросу) коллбэк/скрипт SHALL логировать сравнение DDPM vs DDIM сэмплов из одного стартового шума — чтобы показать разницу между стохастическим и детерминированным сэмплингом (раздел 17 vs 18 теории).

#### Scenario: Both samplers visualized from same seed
- **WHEN** запускают сравнение DDPM vs DDIM
- **THEN** в TensorBoard видны две сетки из одного `x_T`: стохастическая (DDPM) и детерминированная (DDIM)

