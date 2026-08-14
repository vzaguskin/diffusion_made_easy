# mnist-data Specification

## Purpose
TBD - created by archiving change lesson-01-ddpm-mnist-lab. Update Purpose after archive.
## Requirements
### Requirement: MNIST download via torchvision
Модуль данных SHALL загружать MNIST через `torchvision.datasets.MNIST` с автоматическим скачиванием при первом запуске (`download=True`). Скачанные данные SHALL кешироваться в локальной папке (по умолчанию `lab/data/`), чтобы повторные запуски не скачивали их заново.

#### Scenario: First run downloads dataset
- **WHEN** запуск `train.py` происходит впервые (папка `data/` пуста)
- **THEN** MNIST скачивается и распаковывается автоматически

#### Scenario: Subsequent runs use cache
- **WHEN** запуск `train.py` повторяется
- **THEN** скачивание не происходит, данные берутся из кеша

### Requirement: Normalization to unit-variance range
Данные SHALL быть нормализованы так, чтобы пиксели имели нулевое среднее и единичную дисперсию (например, `transforms.Normalize((0.1307,), (0.3081,))` для MNIST). Это соответствует предположению variance-preserving диффузии (раздел 3 теории: `Var(x₀) ≈ 1`).

#### Scenario: Normalization applied
- **WHEN** батч MNIST проходит через даталоадер
- **THEN** эмпирическое среднее пикселей близко к 0, дисперсия близка к 1

### Requirement: Train/val/test split via LightningDataModule
Модуль данных SHALL предоставлять `LightningDataModule` с разбиением train/val/test. Стандартное разбиение MNIST: 55k train / 5k val (hold-out из train) / 10k test. DataModule SHALL отдавать `train_dataloader`, `val_dataloader`, `test_dataloader`.

#### Scenario: All three loaders available
- **WHEN** `MNISTDataModule` настроен и `.setup()` вызван
- **THEN** доступны `train_dataloader()`, `val_dataloader()`, `test_dataloader()`

### Requirement: Configurable batch size and workers
Размер батча для train/val/test и число DataLoader-workers SHALL быть настраиваемы через конфиг (`data.batch_size`, `data.num_workers`). Дефолты подобраны под 6GB GPU.

#### Scenario: Batch size from config
- **WHEN** конфиг задаёт `data.train_batch_size: 64`
- **THEN** `train_dataloader()` отдаёт батчи размера 64

