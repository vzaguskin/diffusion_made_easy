# Обучение моделей: жизненный цикл от претрейна до дистилляции

> Конспект лекции Stanford CME296 (Afshine & Shervine Amidi), с заполнением всех пропусков.
> Оригинал: https://www.youtube.com/watch?v=IvXTl3yj-4Y
> Цель: понять, **как** обучают text-to-image модель на практике — от претрейна до быстрого инференса.

---

## Оглавление

1. [Идея одним абзацем](#1-идея-одним-абзацем)
2. [Жизненный цикл обучения: четыре этапа](#2-жизненный-цикл-обучения-четыре-этапа)
3. [Претрейн: трюки для ускорения](#3-претрейн-трюки-для-ускорения)
4. [Resolution-dependent timestep shift](#4-resolution-dependent-timestep-shift)
5. [REPA: Representation Alignment](#5-repa-representation-alignment)
6. [Continued training и Supervised Fine-Tuning](#6-continued-training-и-supervised-fine-tuning)
7. [Preference tuning: обучение на предпочтениях](#7-preference-tuning-обучение-на-предпочтениях)
8. [Reward model и Bradley-Terry](#8-reward-model-и-bradley-terry)
9. [Flow-GRPO: групповая оптимизация](#9-flow-grpo-групповая-оптимизация)
10. [Diffusion-DPO: прямая оптимизация предпочтений](#10-diffusion-dpo-прямая-оптимизация-предпочтений)
11. [Prompt enhancement](#11-prompt-enhancement)
12. [DreamBooth: персонализация модели](#12-dreambooth-персонализация-модели)
13. [LoRA: эффективная адаптация](#13-lora-эффективная-адаптация)
14. [Дистилляция: ускорение инференса](#14-дистилляция-ускорение-инференса)
15. [Progressive distillation: halves steps](#15-progressive-distillation-halves-steps)
16. [InstaFlow: reflow + дистилляция](#16-instaflow-reflow--дистилляция)

---

## 1. Идея одним абзацем

Обучение production text-to-image модели — не один шаг, а **конвейер из четырёх этапов**. (1) **Претрейн**: учим модель генерировать картинки на миллионах пар (image, text) — просто «как генерировать». (2) **Continued training / SFT**: дообучаем на новых данных и эстетичных примерах — «как генерировать **хорошо**». (3) **Preference tuning**: подстраиваем под человеческие предпочтения через reward model или DPO — «как генерировать то, что **людям нравится**». (4) **Tuning** (DreamBooth, LoRA): персонализируем под конкретный объект или стиль. Наконец, **дистилляция** — ускоряем инференс с 1000 шагов до 1–4, сохраняя качество. Аналогия с кулинарным обучением: учимся готовить → изучаем рецепты → food inspector говорит, что вкуснее → специализация на одном блюде → ускоряем готовку.

---

## 2. Жизненный цикл обучения: четыре этапа

| Этап | Цель | Аналогия (кухня) |
|------|------|-------------------|
| **Pretraining** | Научиться генерировать изображения | Выучить базовые продукты и техники |
| **Continued training** | Новые данные/концепции | Прочитать новые рецепты |
| **SFT** | Эстетика, следование промпту | Подача блюда, красивый вид |
| **Preference tuning** | Больше того, что нравится людям | Food inspector: «вот это лучше» |
| **Tuning** (DreamBooth/LoRA) | Персонализация | Фирменное блюдо шефа |

> Лектор: «Pretraining is about learning how to generate images. Post-training is about learning how to generate *good* images.»

---

## 3. Претрейн: трюки для ускорения

Претрейн — самый дорогой этап (недели на кластерах GPU). Два трюка ускоряют его:

1. **Resolution-dependent timestep shift** — корректируем шаг $t$ для разных разрешений
2. **REPA (Representation Alignment)** — выравниваем промежуточные представления с pretrained encoder

---

## 4. Resolution-dependent timestep shift

### Проблема

Один и тот же шаг зашумления $t$ «выглядит» по-разному для разных разрешений. На высоком разрешении (512×512) тот же $t$ означает **меньше** воспринимаемого шума, чем на низком (64×64), потому что есть больше пикселей для усреднения.

### Решение

Корректируем шаг $t$: для разрешения $m$ (больше пикселей) сдвигаем $t_m$ так, чтобы «воспринимаемый уровень шума» совпадал с эталонным разрешением $n$.

> **Интуиция**: представь однотонную картинку со значением $c$ во всех пикселях. Мы не знаем $c$ (картинка зашумлена). Оцениваем $c$ усреднением по всем пикселям. На высоком разрешении больше пикселей → усреднение точнее → неопределённость меньше → нужно больший $t$, чтобы получить ту же «слепоту».

---

## 5. REPA: Representation Alignment

### Идея

Ускорить обучение, дав модели «учебник» — представления от уже обученного encoder'а (например, DINO, CLIP).

### Метод

На промежуточном слое $l$ диффузионной модели берём скрытое представление $h_t^{(l)}$ для патча $n$ и сравниваем с представлением **чистого** патча от pretrained encoder'а:

$$L_{\text{REPA}} = -\cos\left(W \cdot h_t^{(l)}[n], \; \text{DINO}(\text{clean patch}_n)\right)$$

Полный loss: $L = L_{\text{diffusion}} + \lambda \cdot L_{\text{REPA}}$

### Почему ранние слои?

REPA работает лучше на **ранних** слоях диффузионной модели. Причина: ранние слои содержат **семантические** представления (глобальная структура), а pretrained encoder'ы хороши именно в семантике. Поздние слои — это детали (перцептуальный уровень), которые pretrained encoder не представляет.

### Результат

REPA ускоряет обучение в **~18 раз** — сравнимое качество при гораздо меньшем количестве итераций.

> **Аналогия**: дать студенту учебник перед экзаменом. Студент не учится с нуля, а «выравнивает» свои знания с эталоном.

---

## 6. Continued training и Supervised Fine-Tuning

### Continued training

Дообучение на **новых данных** (например, новые концепции, стили, домены). Модель «читает новые рецепты».

### Supervised Fine-Tuning (SFT)

Дообучение на **качественных** парах (image, text) с хорошими промптами и эстетичными картинками. Учит модель:
- Следовать промпту точнее
- Генерировать визуально приятные картинки

> На практике SFT часто совмещают с промптами, написанными LLM (а не людьми) — подробнее в разделе 11.

---

## 7. Preference tuning: обучение на предпочтениях

### Постановка

У нас есть пары сгенерированных картинок: для одного промпта картинка A лучше картинки B (оценка человека или модели). Хотим дообучить генеративную модель так, чтобы она чаще генерировала «как A».

### Два подхода (как в LLM)

| Метод | Аналогия из LLM | Идея |
|-------|-----------------|------|
| Reward learning + RL | RLHF | Обучить reward model → максимизировать reward через RL |
| Direct Preference Optimization | DPO | Без отдельной reward model — напрямую из пар |

---

## 8. Reward model и Bradley-Terry

### Reward model

Обучаем модель $r_\phi(c, x)$, которая принимает промпт $c$ и картинку $x$, и выдаёт **скор** — насколько хороша картинка для данного промпта.

### Bradley-Terry loss

Для пары $(x_w, x_l)$ где $x_w$ — «победитель» (win), $x_l$ — «проигравший» (lose):

$$L_{\text{BT}} = -\log\sigma\left(r_\phi(c, x_w) - r_\phi(c, x_l)\right)$$

Хотим, чтобы скор победителя был выше скора проигравшего.

### Reward Feedback Learning

Используем reward model для **обратного распространения** градиента через генеративную модель:

$$\max_\theta \; \mathbb{E}_{c, \varepsilon}\left[r_\phi(c, \text{generate}_\theta(c, \varepsilon))\right]$$

Поскольку reward model дифференцируема, градиент течёт через весь процесс генерации обратно в $\theta$.

> **Проблема**: reward hacking — модель «обманывает» reward model, генерируя картинки, которые получают высокий скор, но не являются действительно хорошими.

---

## 9. Flow-GRPO: групповая оптимизация

### Идея

Адаптация GRPO (Group Relative Policy Optimization, популярной в LLM) для диффузионных/flow-моделей.

### Как работает

1. Для одного промпта генерируем **группу** из $G$ картинок (разные стартовые шумы $\varepsilon$)
2. Каждую оцениваем через reward model → скоры $r_1, \ldots, r_G$
3. Вычисляем **advantage** (относительное качество):
$$A_i = \frac{r_i - \bar{r}}{\sigma_r}$$
где $\bar{r}$ — средний скор группы, $\sigma_r$ — стандартное отклонение
4. Обновляем модель так, чтобы картинки с $A_i > 0$ генерировались чаще

### Зачем relative?

Использование **относительных** скоров (а не абсолютных) убирает смещение reward model. Важно только «какая из группы лучше», а не абсолютная оценка.

> Для разнообразия генерации используется SDE (стохастический процесс), который увеличивает разброс сгенерированных картинок.

---

## 10. Diffusion-DPO: прямая оптимизация предпочтений

### Идея

DPO (Direct Preference Optimization) показывает, что можно оптимизировать предпочтения **без** обучения отдельной reward model. Прямо из пар $(x_w, x_l)$.

### Loss

$$L_{\text{DPO}} = -\log\sigma\left(\beta \log\frac{p_\theta(x_w | c)}{p_{\text{ref}}(x_w | c)} - \beta \log\frac{p_\theta(x_l | c)}{p_{\text{ref}}(x_l | c)}\right)$$

где $p_{\text{ref}}$ — reference model (замороженная копия до preference tuning), $\beta$ — регуляризация.

### Интерпретация

Хотим, чтобы $\log\frac{p_\theta(x_w)}{p_{\text{ref}}(x_w)}$ (насколько модель стала чаще генерировать победителя) было больше, чем $\log\frac{p_\theta(x_l)}{p_{\text{ref}}(x_l)}$ (для проигравшего).

> Для диффузионных моделей $p_\theta(x|c)$ вычисляется через PF-ODE (лекция 2) — лог-вероятность данных при данной модели.

---

## 11. Prompt enhancement

### Проблема

Пользователь пишет: «медведь читает книгу». Но модель обучалась на детальных промптах вроде: «a cute fluffy brown teddy bear sitting on a wooden chair, reading a hardcover book, warm lighting, detailed fur texture, 4K».

### Решение

LLM (например, GPT-4) переписывает короткий пользовательский промпт в **детальный, in-distribution** промпт:

```
User:  "медведь читает книгу"
LLM:   "a cute brown teddy bear with round glasses sitting at a wooden desk,
        reading an open book, warm afternoon light, cozy study room background,
        highly detailed fur, soft focus, 4K"
```

> **Аналогия**: в ресторане клиент говорит «хочу мясо». Официант (prompt enhancement) переводит: «стейк рибай, medium-rare, с розмарином и морской солью». Повар понимает, что готовить.

---

## 12. DreamBooth: персонализация модели

### Задача

Хотим, чтобы модель генерировала **конкретный** объект (например, вашего кота), а не любого кота.

### Метод (Ruiz et al., 2023)

1. Фотографируем объект с разных раксурсов (3–10 фото)
2. Связываем объект с **редким токеном** `[V]` (необычное слово, которое модель не встречала)
3. Обучаем модель генерировать изображения по промпту «A [V] cat sitting on a sofa»
4. Модель учится: `[V]` → конкретный объект

### Prior preservation loss

**Проблема**: если дообучать только на фото объекта, модель «забудет» всё остальное (catastrophic forgetting).

**Решение**: добавляем регуляризационный терм:

$$L = L_{\text{DreamBooth}} + \lambda \cdot L_{\text{prior}}$$

где $L_{\text{prior}}$ — loss на обычных парах (image, text), которые модель уже умела генерировать. Это заставляет модель «помнить» старые навыки.

> **Аналогия**: шеф-повар учится готовить новое фирменное блюдо, но продолжает готовить и старые блюда — чтобы не разучиться.

---

## 13. LoRA: эффективная адаптация

### Идея

Вместо дообучения **всех** параметров модели (миллиарды), добавляем **маленькие** обучаемые матрицы рядом с существующими весами.

### Математика

Оригинальный вес: $W_0$ (замороженный). Добавляем:

$$W = W_0 + \Delta W = W_0 + B \cdot A$$

где $A \in \mathbb{R}^{r \times d}$, $B \in \mathbb{R}^{d \times r}$, $r \ll d$ (например, $r = 8$, $d = 4096$).

Обучаемых параметров: $2 \cdot r \cdot d$ вместо $d^2$. При $r = 8, d = 4096$: $65K$ вместо $16M$ — **в 250 раз меньше**.

### Преимущества

- Быстрое дообучение (минуты, а не часы)
- Малый размер adapter'а (мегабайты, а не гигабайты)
- Несколько LoRA можно менять на лету (без перезагрузки модели)

> LoRA используется для стилей, персонажей, доменов. На CivitAI тысячи LoRA-адаптеров для Stable Diffusion.

---

## 14. Дистилляция: ускорение инференса

### Проблема

Production модель делает 1000 шагов (DDPM) или 50 (DDIM/DPM-Solver). Для real-time нужно 1–4 шага.

### Teacher-Student дистилляция

```
Teacher (1000 шагов) → генерирует картинку
Student (1-4 шага)   → пытается получить тот же результат
```

Loss: $\|x_{\text{student}} - x_{\text{teacher}}\|^2$ (MSE) или LPIPS (перцептуальное расстояние).

### Почему не 1 шаг сразу?

«Попросить студента нарисовать шедевр одним движением кисти — слишком много информации для одного шага». Нужно постепенное уменьшение шагов.

---

## 15. Progressive distillation: halves steps

### Метод

Каждая итерация дистилляции **уменьшает** количество шагов вдвое:

```
Teacher: 1024 шагов
  → Student 1: 512 шагов
    → Student 2: 256 шагов
      → ...
        → Student log₂(1024): 1 шаг
```

Каждый «студент» учится делать 2 шага учителя за 1 шаг.

### Визуализация

Каждый шаг — секущая линия на кривой ODE. Учитель делает маленькие шаги (много секущих), студент — большие (меньше секущих). Каждый уровень — упрощение задачи.

### Результат

Можно выбрать количество шагов по trade-off качество/скорость. На равном количестве шагов — лучше, чем DDIM.

---

## 16. InstaFlow: reflow + дистилляция

### Идея

Вместо того чтобы «биться» с кривыми траекториями (трудно аппроксимировать), сначала **выпрямим** их (reflow из лекции 3), потом дистиллируем.

### Шаги

1. **Reflow** (из лекции 3): обучаем новую модель на парах (noise, data), полученных интегрированием ODE старой модели. Траектории становятся **прямее**.
2. **Progressive distillation**: уменьшаем шаги до 1–4.

### Loss

Два этапа:
- **Warm-up**: MSE между student и teacher
- **Perceptual**: LPIPS — сравнение feature maps из pretrained VGG:

$$L_{\text{LPIPS}} = \sum_l \frac{1}{H_l W_l} \|\phi_l(x_{\text{student}}) - \phi_l(x_{\text{teacher}})\|^2$$

где $\phi_l$ — feature map слоя $l$ предобученной VGG.

> MSE фокусируется на пикселях, а LPIPS — на **воспринимаемом** сходстве. LPIPS ближе к человеческому восприятию.

### Важные наблюдения

- **Первый reflow** даёт основное выпрямление; последующие — мало эффекта (и добавляют ошибки дискретизации)
- Reflow + distillation лучше, чем просто reflow или просто distillation

---

## Шпаргалка: жизненный цикл обучения в 10 строках

```
PRETRAIN:     L_diffusion на миллионах пар (image, text)     ← учимся генерировать
              + REPA (alignment с DINO) → ускорение 18×
              + resolution-dependent timestep shift

POST-TRAIN:   Continued training (новые данные)
              SFT (эстетика, prompt-following)

PREFERENCE:   Reward model (Bradley-Terry) → RL
              Flow-GRPO (групповая оптимизация, advantage)
              Diffusion-DPO (без reward model, напрямую из пар)
              + prompt enhancement (LLM переписывает промпт)

TUNING:       DreamBooth: rare token + prior preservation loss
              LoRA: W = W₀ + BA, r≪d → в 250× меньше параметров

DISTILLATION: Teacher (1000 шагов) → Student (1-4 шага)
              Progressive distillation: halving шагов
              InstaFlow: reflow (выпрямление) + distillation
              Loss: MSE + LPIPS (перцептуальное расстояние)
```

---

## Ссылки

- **REPA**: Yu et al. (2024). *Representation Alignment for Generation: Training Diffusion Transformers Is Faster Than You Think*. arXiv:2410.06924
- **DreamBooth**: Ruiz et al. (2023). *DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation*. arXiv:2208.12242
- **LoRA**: Hu et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685
- **Diffusion-DPO**: Wallace et al. (2023). *Diffusion Model Alignment Using Direct Preference Optimization*. arXiv:2311.12908
- **Flow-GRPO**: Zhang et al. (2025). *Flow-GRPO: Training Diffusion Models with Group Relative Policy Optimization*.
- **Progressive Distillation**: Salimans & Ho (2022). *Progressive Distillation for Fast Sampling of Diffusion Models*. arXiv:2202.00512
- **InstaFlow**: Liu et al. (2023). *InstaFlow! One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation*. arXiv:2309.06380
- **LPIPS**: Zhang et al. (2018). *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric*. arXiv:1801.03924
- **Курс**: Stanford CME296, Spring 2026. https://cme296.stanford.edu
- **Видео лекции**: https://www.youtube.com/watch?v=IvXTl3yj-4Y
