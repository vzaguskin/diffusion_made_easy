# course-structure Specification

## Purpose
TBD - created by archiving change lesson-01-ddpm-mnist-lab. Update Purpose after archive.
## Requirements
### Requirement: Course organized as numbered lessons under `lessons/`
Курс SHALL хранить материалы в каталоге `lessons/` на верхнем уровне репозитория. Каждый урок SHALL быть отдельным каталогом вида `lessons/<NN>-<slug>/`, где `<NN>` — двузначный порядковый номер (`01`, `02`, ...), а `<slug>` — описательный kebab-case (например, `01-ddpm-from-scratch`). Нумерация SHALL отражать рекомендуемый порядок прохождения.

#### Scenario: First lesson directory
- **WHEN** урок 1 добавлен в репозиторий
- **THEN** существует каталог `lessons/01-ddpm-from-scratch/` с подпапками/файлами урока

#### Scenario: Lesson slug is descriptive and kebab-case
- **WHEN** создается новый урок про DDIM-сэмплинг
- **THEN** его каталог называется `lessons/02-<kebab-slug>/` (не `lesson_2/`, не `02_ddim/`)

### Requirement: Each lesson has `theory.md` and `lab/`
Каждый урок SHALL содержать файл `theory.md` с теоретическим материалом урока. Уроки с практикой SHALL дополнительно содержать каталог `lab/` с лабораторной работой. Файл `theory.md` урока SHALL быть source-of-truth теории; код в `lab/` MAY ссылаться на разделы `theory.md` по номерам.

#### Scenario: Lesson 1 layout
- **WHEN** урок 1 реализован
- **THEN** существуют `lessons/01-ddpm-from-scratch/theory.md` и `lessons/01-ddpm-from-scratch/lab/`

#### Scenario: Code references theory sections
- **WHEN** в коде `lab/` реализована формула forward process
- **THEN** рядом есть комментарий со ссылкой на номер раздела `theory.md` (например, «# см. theory.md §4: прыжок через все шаги»)

### Requirement: Theory files are copied verbatim from source
Файл `theory.md` для урока 1 SHALL быть точной копией `ml_interview_cheat_sheet/11-diffusion-models.md` без правок содержания (исправления опечаток допустимы только как отдельный явный change в источнике, не здесь). Это гарантирует отсутствие дрейфа между источниками.

#### Scenario: Verbatim copy of lesson 1 theory
- **WHEN** `theory.md` урока 1 создан
- **THEN** его текст побайтово совпадает с источником `ml_interview_cheat_sheet/11-diffusion-models.md`

