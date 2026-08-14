# Implementation Tasks: lesson-02-score-matching-lab

> Порядок отражает зависимости. Математика — сначала, визуализация — поверх неё.

## 1. Скелет лабы и окружение

- [x] 1.1 Создать структуру `lessons/02-score-matching/lab/`: `pyproject.toml` (torch, numpy, matplotlib, omegaconf, tqdm; без lightning), `configs/`, `scripts/`, `src/score_lab/` (`__init__.py`), `tests/`, `.gitignore` (по образцу лабы 1) — [lab2-entrypoints, course-structure]
- [x] 1.2 `uv sync` — окружение собирается; `uv.lock` закоммитить — [lab2-entrypoints]
- [x] 1.3 Реализовать `src/score_lab/config.py` — OmegaConf: дефолт + `key=value` + `--config` (как в лабе 1); хелпер `geometric_sigma_ladder(sigma_max, sigma_min, n_levels)` — [lab2-config]

## 2. 2D-датасеты с аналитическим score (§6)

- [x] 2.1 `src/score_lab/toy_data.py` — `GaussianMixture2D`: сэмплинг, `log_p(x)`, `score(x)` (аналитически, смесь гауссианов; §6) — [toy-2d-distributions]
- [x] 2.2 `gaussians8` — кольцо из 8 гауссианов (σ≈0.1–0.15) как честная смесь — [toy-2d-distributions]
- [x] 2.3 `moons` и `swiss_roll` — генерация точек вдоль кривой + GMM-аппроксимация (≥50 компонент, σ≤0.1); `log_p`/`score` через ту же `GaussianMixture2D` — [toy-2d-distributions]
- [x] 2.4 Тест: аналитический `score` == autograd-градиент `log_p` на сетке (atol 1e-4) для всех трёх датасетов — [toy-2d-distributions]

## 3. Score-сеть (conditioning на σ)

- [x] 3.1 `src/score_lab/models.py` — `ScoreMLP`: вход `(x, fourier(log σ))` → MLP (SiLU, residual-опция) → score `[B,2]`; гиперпараметры из конфига — [score-network]
- [x] 3.2 Fourier-фичи для log σ (sin/cos, геометрические частоты) — [score-network]
- [x] 3.3 Тест: выход `[B,2]`; выходы при разных σ различаются; ≤1M параметров на дефолтах — [score-network]

## 4. DSM-обучение (§7–§8)

- [x] 4.1 `src/score_lab/dsm.py` — функция потерь `mean(‖s_θ(x+σε, σ) + ε/σ‖²)` (цель `−ε/σ`; комментарий про `∇log q(x̃|x)`), VE-зашумление `x̃ = x + σε` — [denoising-score-matching]
- [x] 4.2 `src/score_lab/train_loop.py` — простой цикл: равномерный сэмплинг уровня σ из лестницы на батч, Adam, CSV с лоссом (по уровням σ тоже), seed из конфига — [denoising-score-matching]
- [x] 4.3 Тест: крошечная сеть на одном σ сходится к аналитическому `∇log p_σ` на `gaussians8` (средняя ошибка направления < ~15° в high-density) — [denoising-score-matching]
- [x] 4.4 Тест: DSM-цель с правильным знаком (численная сверка формулы) — [denoising-score-matching]

## 5. Ланжевен (§4) и annealed Ланжевен (§8)

- [x] 5.1 `src/score_lab/langevin.py` — базовый `langevin_sample(score_fn, x0, n_steps, eps)`: `x ← x + ε/2·s + √ε·z`; опция возврата траекторий — [langevin-sampling]
- [x] 5.2 `annealed_langevin(score_fn, sigmas, steps_per_level, step_scale)`: проход по убывающей лестнице, шаг масштабируется по σ_i, состояния переносятся между уровнями, снимки по уровням — [langevin-sampling]
- [x] 5.3 `mode_coverage(samples, means, radius)` — доля мод, покрытых сэмплами — [langevin-sampling]
- [x] 5.4 Тесты: Ланжевен с истинным score сходится к модам GMM (coverage ≥90%); воспроизводимость по seed; annealed coverage > single-σ coverage на `gaussians8` — [langevin-sampling]

## 6. Визуализация (гражданин первого класса)

- [x] 6.1 `src/score_lab/viz.py` — единый стиль (данные синие, истинный score серый, выученный красный); сохранение PNG в `runs/<exp>/` — [score-visualization]
- [x] 6.2 `plot_score_field(true_fn, learned_fn, sigma)` — quiver true vs learned, нормированные стрелки, colorbar по log-норме — [score-visualization]
- [x] 6.3 `plot_density(dist)` — heatmap log p + сэмплы — [score-visualization]
- [x] 6.4 `plot_trajectories(trajs, field)` — пути поверх quiver/heatmap, маркеры финальных точек — [score-visualization]
- [x] 6.5 `plot_annealing_grid(snapshots, sigmas)` — сетка уровни×шаги, сгущение облака — [score-visualization]

## 7. Скрипт 2D-пайплайна

- [x] 7.1 `configs/default.yaml` — все секции (data/model/sigmas/langevin/train/paths), дефолты: gaussians8, L=10 (σ 1.0→0.02), epochs~30, minutes-scale — [lab2-config]
- [x] 7.2 `scripts/train_2d.py` — end-to-end: обучение → все графики (поля, плотность, траектории, annealing-грид, single-σ collapse демо с числом coverage) → `runs/<exp>/` — [lab2-entrypoints]
- [x] 7.3 Прогнать, проверить глазами все PNG; при артефактах — тюнинг дефолтов (step_scale, epochs) — [lab2-entrypoints]

## 8. VE vs VP на MNIST (§9–§10)

- [x] 8.1 `src/score_lab/mnist_ve_vp.py` — расписание VE (геом. σ; σ_max≈5 в unit-variance шкале, эквивалент NCSN-80 — README/спека) и VP (линейные β как в лабе 1) в одном модуле; forward обеих параметризаций — [ve-vp-comparison]
- [x] 8.2 Мини-U-Net (адаптированная копия из лабы 1, base_channels 32, time/σ-conditioning); один и тот же для обеих веток — [ve-vp-comparison]
- [x] 8.3 Обучающий цикл с равным бюджетом: VE-ветка — ε-предсказание с потерей, связанной со score через `−ε/σ` (§9); VP-ветка — как в лабе 1; CSV-кривые обеих — [ve-vp-comparison]
- [x] 8.4 Сэмплеры: VP — ancestral как в лабе 1; VE — Euler по probability-flow ODE + Ланжевен-корректор (README фиксирует упрощение; голый predict-x̂₀-прыжок не работает, см. README) — [ve-vp-comparison]
- [x] 8.5 `scripts/compare_ve_vp.py` — обе модели последовательно, side-by-side сэмплы из одного x_T, график loss-кривых, артефакты в `runs/ve_vs_vp/` — [ve-vp-comparison]
- [x] 8.6 Прогнать сравнение (2×~4 мин), проверить сэмплы обеих веток — [ve-vp-comparison]

## 9. README и финал

- [x] 9.1 `lab/README.md` — quickstart, карта «раздел теории → код» (§2/§4/§6/§7/§8/§9/§10), как читать каждый график, тюнинг Ланжевена, известные упрощения (GMM-аппроксимация, упрощённый VE-сэмплер), перегенерация картинок — [lab2-entrypoints]
- [x] 9.2 Скопировать удачные PNG в `lessons/02-score-matching/images/` и встроить в README урока/лабы (поля, annealing-грид, collapse-демо, VE vs VP) — [score-visualization, ve-vp-comparison]
- [x] 9.3 Финальная проверка: полный прогон `train_2d.py` + `compare_ve_vp.py` + `pytest` зелёный; сверка всех requirements спек по коду — [все]
