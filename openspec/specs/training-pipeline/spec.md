# training-pipeline Specification

## Purpose
TBD - created by archiving change lesson-01-ddpm-mnist-lab. Update Purpose after archive.
## Requirements
### Requirement: End-to-end training via PyTorch Lightning
Обучение SHALL быть реализовано как `LightningModule`, оборачивающий модель ε_θ, `DiffusionCore` и loss. Цикл обучения (forward, loss, backward, optimizer step) SHALL использовать стандартные `training_step` / `validation_step` / `configure_optimizers` — без ручного управления градиентами.

#### Scenario: Single training step
- **WHEN** `training_step` получает батч `x0`
- **THEN** внутри: сэмплируется `t ~ U(1,T)`, `ε ~ N(0,I)`, `x_t = q_sample(x0, t, ε)`, `eps_pred = model(x_t, t)`, возвращается `loss = compute_loss(eps_pred, ε)`

### Requirement: Training samples random timestep per example
В `training_step` для каждого сэмпла в батче SHALL сэмплироваться независимый `t ~ Uniform{1..T}` (раздел 16 теории: шаг 2). Это значит, что `t` — тензор формы `[B]`, не скаляр.

#### Scenario: Different t per sample in batch
- **WHEN** батч размера 64 проходит `training_step`
- **THEN** 64 значения `t` сэмплированы независимо из `{1..T}`

### Requirement: Checkpointing on best validation loss
Пайплайн SHALL сохранять чекпойнт модели по лучшей val-loss (`ModelCheckpoint(monitor="val/loss", mode="min")`) и SHALL поддерживать загрузку чекпойнта для последующей генерации через `sample.py`. Путь к чекпойнтам SHALL быть настраиваем (дефолт `lab/checkpoints/`).

#### Scenario: Best checkpoint saved
- **WHEN** val-loss улучшается на какой-то эпохе
- **THEN** в `checkpoints/` обновляется файл лучшего чекпойнта

#### Scenario: Resume from checkpoint
- **WHEN** `sample.py` передают путь к чекпойнту
- **THEN** модель загружается из него и готова к генерации

### Requirement: Device-agnostic run
Запуск SHALL автоматически выбирать GPU если доступен, иначе CPU (`Trainer(accelerator="auto", devices="auto")`). Пользователь не должен вручную `.cuda()` тензоры — Lightning и `DiffusionCore` (через `register_buffer`) корректно размещают их.

#### Scenario: GPU used when available
- **WHEN** запуск на машине с CUDA
- **THEN** обучение идёт на GPU (видно в логе Lightning)

#### Scenario: CPU fallback
- **WHEN** запуск на машине без CUDA
- **THEN** обучение идёт на CPU без ошибок (медленнее, но работает)

### Requirement: Configurable optimizer
Оптимизатор SHALL быть настраиваем через конфиг (`optim.type`, `optim.lr`, `optim.weight_decay`), с дефолтом Adam (`lr=2e-4`, без weight decay) — стандарт для DDPM.

#### Scenario: Custom learning rate
- **WHEN** конфиг задаёт `optim.lr: 1e-4`
- **THEN** оптимизатор создаётся с этим lr

