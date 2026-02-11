# Дебагинг Keyframes

## Как использовать

### 1. Включить сохранение keyframes во время обучения

Добавьте в конфигурацию (`configs/Replica/office0/ActiveSem.py`):

```python
slam = dict(
    # ... другие параметры ...
    save_keyframes_debug = True,  # Включить сохранение keyframes для дебагинга
)
```

### 2. Визуализировать keyframes после обучения

```bash
# Базовое использование
bash scripts/debug/visualize_keyframes.sh results/Replica/office0/ActiveSem/run_0

# С указанием dataset директории
bash scripts/debug/visualize_keyframes.sh \
    results/Replica/office0/ActiveSem/run_0 \
    data/Replica/office0

# Или напрямую через Python
python src/debug/visualize_keyframes.py \
    --result_dir results/Replica/office0/ActiveSem/run_0 \
    --dataset_dir data/Replica/office0 \
    --output_dir results/Replica/office0/ActiveSem/run_0/debug_keyframes
```

### 3. Результаты

Скрипт создаст:
- `debug_keyframes/keyframes_rgb/` - RGB изображения всех keyframes
- `debug_keyframes/keyframes_visualization.html` - HTML визуализация в браузере
- `debug_keyframes/keyframes_info.json` - JSON с информацией о keyframes

## Что показывают keyframes

- **Обычные keyframes**: Кадры, выбранные каждые `keyframe_every` шагов (по умолчанию 5)
- **Global keyframes**: Кадры, добавленные на основе качества или полноты покрытия (зеленая рамка в HTML)

## Анализ keyframes

Keyframes показывают:
1. **Какие области сцены были исследованы** - по расположению keyframes можно понять, какие части комнаты робот посетил
2. **Качество покрытия** - если keyframes сконцентрированы в одной области, возможно, робот не исследовал всю сцену
3. **Эффективность активного планирования** - сравните распределение keyframes в Active vs Passive режимах





