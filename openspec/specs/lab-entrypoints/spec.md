# lab-entrypoints Specification

## Purpose
TBD - created by archiving change lesson-01-ddpm-mnist-lab. Update Purpose after archive.
## Requirements
### Requirement: `train.py` is the single entry point for training
Скрипт `scripts/train.py` SHALL быть единственной точкой входа для обучения: он загружает конфиг, настраивает DataModule / модель / LightningModule / callbacks / logger, запускает `Trainer.fit()` и завершается. Никаких отдельных скриптов для «только скачать» или «только одну эпоху» не требуется — всё через конфиг/CLI.

#### Scenario: Fresh run trains end-to-end
- **WHEN** студент запускает `uv run python scripts/train.py`
- **THEN** скачивается MNIST (если нужно), запускается обучение, пишутся логи в TensorBoard и сохраняются чекпойнты

### Requirement: `sample.py` generates images from checkpoint
Скрипт `scripts/sample.py` SHALL загружать модель из указанного чекпойнта и генерировать партию картинок выбранным сэмплером (DDPM или DDIM), сохраняя результат как файл(ы) картинок (например, PNG-сетку) и/или логируя в TensorBoard. Аргументы: путь к чекпойнту, сэмплер, число шагов, число картинок, seed.

#### Scenario: Generate with DDIM from checkpoint
- **WHEN** запускают `sample.py --checkpoint best.ckpt --sampler ddim --num-steps 50 --num-samples 64`
- **THEN** создаётся файл с сеткой из 64 сгенерированных картинок

#### Scenario: Reproducible generation
- **WHEN** `sample.py` запускают дважды с одинаковым `--seed`
- **THEN** сгенерированные картинки (при детерминированном сэмплере) идентичны

### Requirement: README with run instructions
Лаба SHALL иметь `README.md` с инструкцией: требования (uv, GPU/CPU), установка (`uv sync`), запуск обучения, запуск генерации, как смотреть TensorBoard, описание структуры каталогов, список ключей конфига, и что ожидать (сколько эпох до узнаваемых цифр, типичные проблемы и их решение).

#### Scenario: New student can run the lab
- **WHEN** студент клонирует репо и идёт по шагам README
- **THEN** он успешно запускает обучение и видит сэмплы в TensorBoard

### Requirement: uv-based dependency management
Все зависимости SHALL быть зафиксированы в `pyproject.toml`, а точные версии — в `uv.lock`. Команда `uv sync` SHALL создавать воспроизводимое окружение. Запуск скриптов через `uv run python scripts/...` SHALL работать без ручного активации venv.

#### Scenario: Reproducible environment
- **WHEN** на новой машине выполняют `uv sync`
- **THEN** устанавливаются версии зависимостей, зафиксированные в `uv.lock`

#### Scenario: Run without manual venv
- **WHEN** выполняют `uv run python scripts/train.py`
- **THEN** запуск происходит в корректном окружении без ручного `source venv/bin/activate`

### Requirement: Generated artifacts ignored by git
Каталоги с генерируемыми артефактами (данные MNIST, чекпойнты, логи TensorBoard, сгенерированные картинки) SHALL быть добавлены в `.gitignore`, чтобы не попадать в коммиты.

#### Scenario: Artifacts not committed
- **WHEN** после прогона обучения выполняют `git status`
- **THEN** `data/`, `checkpoints/`, `runs/`, сгенерированные картинки не отображаются как untracked

