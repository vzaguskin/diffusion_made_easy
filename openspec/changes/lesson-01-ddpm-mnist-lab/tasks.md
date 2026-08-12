# Implementation Tasks: lesson-01-ddpm-mnist-lab

> Порядок отражает зависимости. Каждая задача верифицируема. Ссылки на capability/spec — в скобках.

## 1. Структура курса и теория

- [x] 1.1 Создать каталог `lessons/01-ddpm-from-scratch/` (пустой) — [course-structure]
- [x] 1.2 Скопировать `ml_interview_cheat_sheet/11-diffusion-models.md` → `lessons/01-ddpm-from-scratch/theory.md` побайтово (без правок) — [course-structure, lesson-01-theory]
- [x] 1.3 Проверить, что номера разделов theory.md используются как цели для ссылок из кода (зафиксировать список: §3 forward, §4 прыжок, §14 predict_x0, §15 loss, §17 DDPM, §18 DDIM) — [lesson-01-theory]

## 2. Скелет лабы и зависимости

- [x] 2.1 Создать структуру `lessons/01-ddpm-from-scratch/lab/` (`pyproject.toml`, `configs/`, `scripts/`, `src/ddpm_lab/` с `__init__.py` и подкаталогами `models/`, `samplers/`) — [lab-entrypoints, lab-config]
- [x] 2.2 Написать `pyproject.toml`: имя `ddpm-lab`, Python ≥3.10, зависимости `torch`, `lightning`, `tensorboard`, `torchvision`, `numpy`, `omegaconf`, `tqdm`; dev-зависимости по необходимости — [lab-entrypoints]
- [x] 2.3 Запустить `uv sync` и убедиться, что окружение собирается; закоммитить `uv.lock` — [lab-entrypoints]
- [x] 2.4 Создать `.gitignore` в корне лабы: `data/`, `checkpoints/`, `runs/`, `*.png`, `__pycache__/`, `*.ckpt` — [lab-entrypoints]

## 3. DDPM-ядро (математика)

- [x] 3.1 Реализовать `src/ddpm_lab/schedules.py` — `linear_beta_schedule(beta_start, beta_end, num_timesteps)` (раздел 3 теории) — [ddpm-core]
- [x] 3.2 Реализовать `src/ddpm_lab/core.py` — класс `DiffusionCore`: хранение `betas`, precompute `alphas`, `alphas_cumprod`, `sqrt_alphas_cumprod`, `sqrt_one_minus_alphas_cumprod` через `register_buffer` (device-aware) — [ddpm-core]
- [x] 3.3 В `DiffusionCore` реализовать `q_sample(x0, t, noise=None)` → `x_t = √ᾱ·x₀ + √(1-ᾱ)·ε` (раздел 4); `noise=None` → сэмплирует `N(0,I)` — [ddpm-core]
- [x] 3.4 В `DiffusionCore` реализовать `predict_start_from_noise(xt, t, noise)` → `x₀ = (x_t − √(1-ᾱ)·ε)/√ᾱ` (раздел 14) — [ddpm-core]
- [x] 3.5 В `DiffusionCore` реализовать `compute_loss(eps_pred, eps_target)` → невзвешенный MSE (раздел 15: «отбрасывание веса») — [ddpm-core]
- [x] 3.6 Юнит-проверки (в скрипте или pytest): `q_sample` + `predict_start_from_noise` взаимно обратны; `x_T` имеет дисперсию ≈1 (variance-preserving) — [ddpm-core]

## 4. Модели ε_θ

- [x] 4.1 Реализовать `src/ddpm_lab/models/common.py` — `SinusoidalTimeEmbedding` (общий для MLP и U-Net) — [noise-prediction-models]
- [x] 4.2 Реализовать `src/ddpm_lab/models/mlp.py` — ε_θ как MLP: flatten → time-embed → residual-блоки → reshape; `forward(x, t) -> eps` той же формы — [noise-prediction-models]
- [x] 4.3 Реализовать `src/ddpm_lab/models/unet.py` — маленький U-Net с time-embedding, residual-блоками, up/down-sampling; влезает в 6GB VRAM при дефолтном батче на MNIST — [noise-prediction-models]
- [x] 4.4 Реализовать фабрику `build_model(cfg)` в `models/__init__.py`: выбор `mlp`/`unet` по `cfg.model.type`, дефолт `unet` — [noise-prediction-models]
- [x] 4.5 Проверка: выход `model(x, t)` имеет ту же форму, что `x`; обе модели возвращают единственный тензор (без дисперсии) — [noise-prediction-models]

## 5. Сэмплеры

- [x] 5.1 Реализовать `src/ddpm_lab/samplers/ddpm.py` — стохастический сэмплер (раздел 17): цикл `t=T...1`, `μ = (1/√α)(x_t − (1-α)/√(1-ᾱ)·ε_pred)`, `x_{t-1}=μ+σ_t·z` при `t>1`, `x_0=μ` при `t=1` — [ddpm-samplers]
- [x] 5.2 Реализовать `src/ddpm_lab/samplers/ddim.py` — детерминированный сэмплер с `η` (раздел 18), поддержка `num_steps < T` через подмножество временных шагов — [ddpm-samplers]
- [x] 5.3 Общий интерфейс `sample(model, shape, num_steps, *, generator=None, **kwargs) -> Tensor` для обоих сэмплеров; опциональный `generator` для воспроизводимости — [ddpm-samplers]
- [x] 5.4 Проверка: DDIM с `η=0` детерминирован (повторный запуск = тот же результат); DDIM с `num_steps=25` быстрее 1000-шагового DDPM — [ddpm-samplers]

## 6. Данные MNIST

- [x] 6.1 Реализовать `src/ddpm_lab/data.py` — `MNISTDataModule` (Lightning): `download=True`, кеш в `cfg.data_dir`, нормализация `Normalize((0.1307,), (0.3081,))` (variance-preserving) — [mnist-data]
- [x] 6.2 Разбиение train/val: 55k/5k hold-out из тренировочного множества; `test` = стандартные 10k — [mnist-data]
- [x] 6.3 Настраиваемые `train_batch_size`, `val_batch_size`, `num_workers` из конфига — [mnist-data]
- [x] 6.4 Проверка: первый запуск скачивает MNIST, повторный — берёт из кеша; дисперсия батча ≈1 после нормализации — [mnist-data]

## 7. Конфиг

- [x] 7.1 Реализовать `src/ddpm_lab/config.py` — загрузка/слияние через OmegaConf: дефолт `configs/default.yaml` + override через CLI `key=value` + опция `--config <file>` — [lab-config]
- [x] 7.2 Написать `configs/default.yaml` со всеми секциями: `diffusion` (`num_timesteps=1000`, `beta_start=1e-4`, `beta_end=0.02`), `model` (`type=unet`, размеры), `data` (`train_batch_size=128`, `num_workers`), `optim` (`type=adam`, `lr=2e-4`), `train` (`epochs=20`), `callbacks` (`sample_freq=1`, `num_samples=64`, `sampler=ddim`), `paths` — [lab-config]
- [x] 7.3 Проверить, что дефолты влезают в 6GB VRAM (прогнать 1–2 шага обучения на GPU; при OOM — уменьшить `train_batch_size` и зафиксировать в дефолте) — [lab-config]

## 8. Метрики

- [x] 8.1 Реализовать `src/ddpm_lab/metrics.py` — `loss_by_t_bucket(eps_pred, eps_target, t, num_buckets=10)` → MSE по интервалам `t` (без тяжёлых сетей) — [evaluation-metrics]
- [x] 8.2 Реализовать `coverage_and_mode_collapse(samples, real_images, real_labels)` → `(coverage, nn_class_distribution)` через NN в пиксельном пространстве (L2 по flatten) — [evaluation-metrics]
- [x] 8.3 Проверка: метрики не загружают Inception/внешние сети; coverage ∈ [0,1] — [evaluation-metrics]

## 9. Lightning-модуль и коллбэки

- [x] 9.1 Реализовать `src/ddpm_lab/lightning_module.py` — `DDPMLightningModule`: `training_step` (сэмпл `t ~ U(1,T)` per-example, `ε`, `x_t`, `eps_pred`, loss), `validation_step`, `configure_optimizers` (Adam, `cfg.optim.lr`) — [training-pipeline]
- [x] 9.2 В `validation_step` считать и логировать `val/loss` и bucketed losses (через `metrics.loss_by_t_bucket`) в TensorBoard — [training-pipeline, evaluation-metrics, tensorboard-logging]
- [x] 9.3 Реализовать `src/ddpm_lab/callbacks.py` — `SamplingCallback`: на `val_epoch_end` (с частотой `cfg.callbacks.sample_freq`) генерит `num_samples` картинок фиксированным seed’ом (фиксированный буфер `x_T`), логирует сетку через `add_image` — [tensorboard-logging]
- [x] 9.4 В коллбэке опционально логировать DDPM vs DDIM сравнение из одного `x_T` (раздел 17 vs 18) — [tensorboard-logging]
- [x] 9.5 `ModelCheckpoint(monitor="val/loss", mode="min")` с путём `cfg.checkpoint_dir` — [training-pipeline]

## 10. Точки входа

- [x] 10.1 Реализовать `scripts/train.py` — загрузка конфига, сборка DataModule/model/LightningModule/callbacks/logger, `Trainer(accelerator="auto", devices="auto")`, `trainer.fit()` — [training-pipeline, lab-entrypoints]
- [x] 10.2 Реализовать `scripts/sample.py` — загрузка чекпойнта, генерация выбранным сэмплером (`--sampler ddpm|ddim`, `--num-steps`, `--num-samples`, `--seed`), сохранение PNG-сетки и/или лог в TB — [lab-entrypoints, ddpm-samplers]
- [x] 10.3 Проверка: `uv run python scripts/train.py` идёт end-to-end на 1 эпохе (или `max_steps=5` для smoke-теста) без ошибок; CPU-фолбэк работает — [lab-entrypoints, training-pipeline]
- [x] 10.4 Проверка: `uv run python scripts/sample.py --checkpoint <ckpt> --sampler ddim --num-steps 25 --num-samples 16 --seed 0` создаёт PNG-сетку — [lab-entrypoints]

## 11. README и финальный прогон

- [x] 11.1 Написать `lab/README.md`: требования (uv, GPU/CPU), `uv sync`, запуск `train.py`, запуск `sample.py`, как смотреть TensorBoard (`tensorboard --logdir runs/`), структура каталогов, описание ключей конфига, что ожидать (10–20 эпох до узнаваемых цифр, типичные проблемы/OOM) — [lab-entrypoints, lab-config]
- [x] 11.2 Финальный прогон: полный запуск обучения на дефолтах (или уменьшенных для скорости), убедиться что сэмплы в TensorBoard показывают прогресс, loss падает, coverage считается — [training-pipeline, tensorboard-logging, evaluation-metrics]
- [x] 11.3 Smoke-проверка всех capability: пройти по спекам и убедиться, что каждый requirement покрыт кодом/README — [все capability]
