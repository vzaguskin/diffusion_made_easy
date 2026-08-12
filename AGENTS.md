# AGENTS.md — diffusion_made_easy

Этот проект использует [OpenSpec](https://github.com/Fission-AI/OpenSpec) для spec-driven разработки.
Читай и пиши спецификации и артефакты изменений в согласовании с CLI `openspec`.

---

## Source of truth

| Что | Где |
|-----|-----|
| Текущие (утверждённые) спецификации | `openspec/specs/<capability>/spec.md` |
| Активные изменения (proposals) | `openspec/changes/<change-name>/` |
| Архив завершённых изменений | `openspec/changes/archive/YYYY-MM-DD-<change-name>/` |
| Конфиг openspec | `openspec/config.yaml` |

**Принцип**: `openspec/specs/` — это утверждённое состояние системы. `openspec/changes/` — proposed дельты к нему. После имплементации и архивации дельта переносится в `specs/`.

---

## Структура change (schema: spec-driven)

```
openspec/changes/<name>/
├── proposal.md     ← WHY: зачем это изменение
├── specs/          ← WHAT: дельта-спеки по capabilities (ADDED/MODIFIED/REMOVED/RENAMED Requirements)
├── design.md       ← HOW: технические решения, tradeoffs
└── tasks.md        ← чек-лист имплементации (двигает фазу apply)
```

---

## Workflow

1. **Explore** — подумать над проблемой (`/opsx-explore`)
2. **New / FF** — создать change (`/opsx-new` шагами или `/opsx-ff` всё сразу)
3. **Continue** — создавать артефакты по очереди (`/opsx-continue`)
4. **Apply** — имплементировать таски (`/opsx-apply`)
5. **Verify** — сверить имплементацию со спеками (`/opsx-verify`)
6. **Archive** — синхронизировать дельту в main specs и архивировать (`/opsx-archive`)

---

## Доступные команды ZCode

| Команда | Назначение |
|---------|------------|
| `/opsx-explore` | Режим размышления: исследовать проблему, без написания кода |
| `/opsx-new <name>` | Создать change, показать инструкции для первого артефакта, остановиться |
| `/opsx-ff <name>` | Fast-forward: создать change и **все** артефакты за один заход |
| `/opsx-continue` | Создать следующий артефакт (по одному за вызов) |
| `/opsx-apply` | Имплементировать таски из change |
| `/opsx-verify` | Проверить, что имплементация соответствует артефактам |
| `/opsx-sync` | Синхронизировать дельта-спеки в main specs (без архивации) |
| `/opsx-archive` | Заархивировать завершённый change (sync + move в archive/) |
| `/opsx-bulk-archive` | Заархивировать несколько changes разом, с разрешением конфликтов спеков |
| `/opsx-onboard` | Обучающий проход по полному циклу на реальной задаче |

---

## Ключевые правила для агента

- **`context` и `rules` из `openspec instructions` — это ограничения для ТЕБЯ**, а не содержимое файла. Никогда не копируй блоки `<context>`, `<rules>`, `<project_context>` в артефакты.
- **Используй CLI как источник истины**, не додумывай схему: `openspec status --change X --json`, `openspec instructions <artifact> --change X --json`, `openspec list --json`.
- **Не создавай артефакты с потолка** — только через `openspec instructions` (там шаблон, outputPath, зависимости).
- **Зачёркивай таски** сразу после выполнения: `- [ ]` → `- [x]`.
- **Delta-спека — это intent, не полная замена**. При merge в main specs сохраняй существующие требования/сценарии, не упомянутые в дельте.
- **Архивация**: `mv openspec/changes/<name> openspec/changes/archive/YYYY-MM-DD-<name>`. Проверяй, что target не существует.
- **Перед archive** прогоняй `openspec validate --changes` и сверяй статус артефактов/тасков.

---

## Требования к окружению

- **Node.js ≥ 20.19** (рекомендуется через nvm: `nvm use default`)
- **openspec CLI**: `npm install -g @fission-ai/openspec@latest`
- Проверка: `openspec --version`
