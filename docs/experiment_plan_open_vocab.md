# План эксперимента: Гибридная семантика (Closed + Open-Vocabulary)

## 1. Цель эксперимента

Добавить Open-Vocabulary семантику через CLIP embeddings в каждый Gaussian, сохранив существующую closed-vocabulary семантику для планирования.

**Архитектура:**
- **Closed vocabulary** (как сейчас): top-k class probabilities (16 классов) → для планирования и reward
- **Open vocabulary** (новое): CLIP-compatible embeddings (32-64D) → для querying и памяти

## 2. Текущая структура семантики

### 2.1 Хранение в Gaussians

**Файл:** `src/slam/semsplatam/modified_ver/splatam/splatam.py`

```python
# Текущая структура params:
params = {
    'means3D': [N, 3],
    'rgb_colors': [N, 3],
    'semantic_logits': [N, TOPK],  # TOPK=16, sparse representation
    'unnorm_rotations': [N, 4],
    'logit_opacities': [N, 1],
    'log_scales': [N, 1] or [N, 3],
}

# Текущая структура variables:
variables = {
    'seman_cls_ids': [N, TOPK],  # Индексы классов для sparse representation
}
```

### 2.2 Использование в планировании

**Файл:** `src/planner/active_gs_planner_v2.py:1269-1271`

```python
# Рендеринг семантики для планирования
_, logits = gs_slam.render_semantic(cand_pose, seen)  # logits: [C, H, W]

# Вычисление энтропии для IG
topk_prob, _ = torch.topk(logits, 16, dim=0)
entropy = calc_shannon_entropy(topk_prob, dim=0).mean()
seman_entropies.append(entropy)
```

### 2.3 Рендеринг семантики

**Файл:** `src/slam/semsplatam/semsplatam.py:381-422`

- Использует `SEMRenderer_sparse` для рендеринга sparse semantic logits
- Возвращает `class_id` и `logits` для планирования

## 3. Техническая реализация

### 3.1 Модификация структуры параметров

**Файл:** `src/slam/semsplatam/modified_ver/splatam/splatam.py`

#### 3.1.1 Добавление CLIP embeddings в params

```python
def initialize_params_with_seman(init_pt_cld, num_frames, mean3_sq_dist, 
                                 gaussian_distribution, TOPK=16, 
                                 clip_embed_dim=64):  # НОВЫЙ ПАРАМЕТР
    # ... существующий код ...
    
    params = {
        'means3D': means3D,
        'rgb_colors': init_pt_cld[:, 3:6],
        'semantic_logits': init_pt_cld[:, 6:],  # Closed vocab (как сейчас)
        'clip_embeddings': torch.zeros((num_pts, clip_embed_dim), 
                                       dtype=torch.float32, device="cuda"),  # НОВОЕ
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': log_scales,
    }
    
    # ... остальной код ...
    
    # CLIP embeddings НЕ участвуют в оптимизации через градиенты
    # Они обновляются через проекцию из CLIP features
    params['clip_embeddings'] = torch.nn.Parameter(
        params['clip_embeddings'].requires_grad_(False)  # requires_grad=False
    )
```

#### 3.1.2 Инициализация CLIP embeddings

**Новый файл:** `src/slam/semsplatam/modified_ver/semantic/clip_extractor.py`

```python
import torch
import clip
from PIL import Image
import torchvision.transforms as transforms

class CLIPFeatureExtractor:
    def __init__(self, model_name="ViT-B/32", device="cuda", embed_dim=64):
        """
        Args:
            model_name: CLIP model variant ("ViT-B/32", "ViT-L/14", etc.)
            device: CUDA device
            embed_dim: Target embedding dimension (32, 64, etc.)
        """
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.embed_dim = embed_dim
        
        # Проекция из CLIP dimension (512 для ViT-B/32) в embed_dim
        self.projection = torch.nn.Linear(
            self.model.visual.output_dim, embed_dim
        ).to(device)
        
        # Заморозить CLIP модель
        for param in self.model.parameters():
            param.requires_grad = False
    
    def extract_features(self, rgb_image: torch.Tensor) -> torch.Tensor:
        """
        Извлекает CLIP embeddings из RGB изображения.
        
        Args:
            rgb_image: [H, W, 3] или [3, H, W], диапазон [0, 1] или [0, 255]
        
        Returns:
            features: [H, W, embed_dim] CLIP-compatible embeddings
        """
        # Нормализация и подготовка изображения
        if rgb_image.max() > 1.0:
            rgb_image = rgb_image / 255.0
        
        if rgb_image.shape[0] == 3:  # [3, H, W] -> [H, W, 3]
            rgb_image = rgb_image.permute(1, 2, 0)
        
        # Преобразование в PIL Image и применение CLIP preprocessing
        pil_image = Image.fromarray((rgb_image.cpu().numpy() * 255).astype(np.uint8))
        image_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        
        # Извлечение features через CLIP visual encoder
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)  # [1, 512]
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Проекция в низкоразмерное пространство
        projected = self.projection(image_features)  # [1, embed_dim]
        projected = projected / projected.norm(dim=-1, keepdim=True)
        
        # Upsample до разрешения изображения (простая интерполяция)
        # В реальности лучше использовать per-pixel features через DINOv2 или SAM
        # Для MVP используем простую интерполяцию
        h, w = rgb_image.shape[:2]
        features_2d = projected.unsqueeze(0).repeat(h, w, 1)  # [H, W, embed_dim]
        
        return features_2d
    
    def extract_per_pixel_features(self, rgb_image: torch.Tensor) -> torch.Tensor:
        """
        Извлекает per-pixel CLIP features (более точный метод).
        Использует DINOv2 или SAM для per-pixel features, затем проекцию через CLIP.
        
        TODO: Реализовать через DINOv2 или SAM
        """
        # Placeholder для более точной реализации
        return self.extract_features(rgb_image)
```

### 3.2 Интеграция CLIP в процесс обучения

**Файл:** `src/slam/semsplatam/semsplatam.py`

#### 3.2.1 Инициализация CLIP extractor

```python
class SemSplatam(SplatamOurs):
    def __init__(self, main_cfg, info_printer, logger):
        # ... существующий код ...
        
        # Инициализация CLIP extractor
        self.clip_extractor = None
        if hasattr(main_cfg.slam, 'use_clip_embeddings') and main_cfg.slam.use_clip_embeddings:
            from src.slam.semsplatam.modified_ver.semantic.clip_extractor import CLIPFeatureExtractor
            clip_embed_dim = getattr(main_cfg.slam, 'clip_embed_dim', 64)
            self.clip_extractor = CLIPFeatureExtractor(
                model_name=getattr(main_cfg.slam, 'clip_model', "ViT-B/32"),
                device=self.device,
                embed_dim=clip_embed_dim
            )
```

#### 3.2.2 Обновление CLIP embeddings при добавлении новых Gaussians

**Файл:** `src/slam/semsplatam/modified_ver/splatam/splatam.py`

```python
def add_new_gaussians_with_seman(params, variables, curr_data, sil_thres,
                                 time_idx, mean_sq_dist_method, 
                                 gaussian_distribution, TOPK=16,
                                 clip_extractor=None):  # НОВЫЙ ПАРАМЕТР
    """
    Добавляет новые Gaussians с семантикой и CLIP embeddings.
    """
    # ... существующий код для добавления Gaussians ...
    
    # После создания новых Gaussians, обновить CLIP embeddings
    if clip_extractor is not None and 'clip_embeddings' in params:
        # Получить RGB изображение для текущего кадра
        rgb_image = curr_data['im']  # [3, H, W]
        
        # Извлечь CLIP features
        clip_features = clip_extractor.extract_per_pixel_features(rgb_image)  # [H, W, embed_dim]
        
        # Найти индексы новых Gaussians (после densification)
        new_gaussian_indices = ...  # Индексы только что добавленных Gaussians
        
        # Обновить CLIP embeddings для новых Gaussians
        # Проекция из 2D features в 3D Gaussians через nearest neighbor или interpolation
        new_clip_embeddings = project_2d_to_3d_gaussians(
            clip_features, 
            new_gaussian_positions,  # [N_new, 3]
            curr_data['cam'],
            curr_data['w2c']
        )  # [N_new, embed_dim]
        
        # Обновить params['clip_embeddings'] для новых Gaussians
        params['clip_embeddings'][new_gaussian_indices] = new_clip_embeddings
```

#### 3.2.3 Функция проекции 2D → 3D

**Новый файл:** `src/slam/semsplatam/modified_ver/splatam/clip_utils.py`

```python
def project_2d_to_3d_gaussians(clip_features_2d: torch.Tensor,
                                gaussian_positions_3d: torch.Tensor,
                                camera,
                                w2c: torch.Tensor) -> torch.Tensor:
    """
    Проецирует 2D CLIP features в 3D позиции Gaussians.
    
    Args:
        clip_features_2d: [H, W, embed_dim] CLIP features
        gaussian_positions_3d: [N, 3] 3D позиции Gaussians
        camera: Camera object
        w2c: [4, 4] world-to-camera transformation
    
    Returns:
        clip_features_3d: [N, embed_dim] CLIP embeddings для Gaussians
    """
    # Проецировать 3D позиции в 2D пиксели
    gaussian_positions_homogeneous = torch.cat([
        gaussian_positions_3d,
        torch.ones(gaussian_positions_3d.shape[0], 1, device=gaussian_positions_3d.device)
    ], dim=1)  # [N, 4]
    
    # Преобразование в camera space
    gaussian_positions_cam = (w2c @ gaussian_positions_homogeneous.T).T  # [N, 4]
    gaussian_positions_cam = gaussian_positions_cam[:, :3]  # [N, 3]
    
    # Проецирование в 2D
    fx, fy = camera.fx, camera.fy
    cx, cy = camera.cx, camera.cy
    
    x_2d = (fx * gaussian_positions_cam[:, 0] / gaussian_positions_cam[:, 2]) + cx
    y_2d = (fy * gaussian_positions_cam[:, 1] / gaussian_positions_cam[:, 2]) + cy
    
    # Округление до целых пикселей
    x_2d = torch.clamp(x_2d.long(), 0, clip_features_2d.shape[1] - 1)
    y_2d = torch.clamp(y_2d.long(), 0, clip_features_2d.shape[0] - 1)
    
    # Билинейная интерполяция (опционально, для более плавных features)
    # Для простоты используем nearest neighbor
    clip_features_3d = clip_features_2d[y_2d, x_2d]  # [N, embed_dim]
    
    return clip_features_3d
```

### 3.3 Сохранение и загрузка CLIP embeddings

**Файл:** `src/slam/semsplatam/semsplatam.py`

```python
def save_params(self, save_path: str):
    """Сохраняет параметры Gaussians, включая CLIP embeddings."""
    # ... существующий код ...
    
    if 'clip_embeddings' in self.params:
        np.savez_compressed(
            save_path,
            # ... существующие параметры ...
            clip_embeddings=self.params['clip_embeddings'].detach().cpu().numpy(),
        )

def load_params(self, load_path: str):
    """Загружает параметры Gaussians, включая CLIP embeddings."""
    # ... существующий код ...
    
    if 'clip_embeddings' in loaded_params:
        self.params['clip_embeddings'] = torch.nn.Parameter(
            torch.from_numpy(loaded_params['clip_embeddings']).to(self.device)
        )
```

### 3.4 Querying через CLIP embeddings

**Новый файл:** `src/slam/semsplatam/modified_ver/splatam/clip_query.py`

```python
import torch
import clip

class CLIPQueryEngine:
    def __init__(self, clip_model_name="ViT-B/32", device="cuda"):
        self.device = device
        self.model, _ = clip.load(clip_model_name, device=device)
        for param in self.model.parameters():
            param.requires_grad = False
    
    def query_text(self, text_query: str, gaussian_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Находит Gaussians, соответствующие текстовому запросу.
        
        Args:
            text_query: Текст запроса (например, "red chair", "kitchen table")
            gaussian_embeddings: [N, embed_dim] CLIP embeddings Gaussians
        
        Returns:
            similarity_scores: [N] similarity scores между запросом и Gaussians
        """
        # Кодирование текстового запроса
        text_tokens = clip.tokenize([text_query]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)  # [1, 512]
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Проекция text_features в то же пространство, что и gaussian_embeddings
        # (нужно использовать ту же проекцию, что и в CLIPFeatureExtractor)
        # Для простоты предполагаем, что gaussian_embeddings уже в правильном пространстве
        
        # Вычисление cosine similarity
        similarity = torch.nn.functional.cosine_similarity(
            text_features, gaussian_embeddings, dim=-1
        )  # [N]
        
        return similarity
    
    def query_image(self, image_query: torch.Tensor, 
                    gaussian_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Находит Gaussians, соответствующие изображению-запросу.
        
        Args:
            image_query: [3, H, W] RGB изображение
            gaussian_embeddings: [N, embed_dim] CLIP embeddings Gaussians
        
        Returns:
            similarity_scores: [N] similarity scores
        """
        # Кодирование изображения
        image_tensor = self.preprocess(image_query).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)  # [1, 512]
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Проекция и similarity (аналогично query_text)
        similarity = torch.nn.functional.cosine_similarity(
            image_features, gaussian_embeddings, dim=-1
        )
        
        return similarity
```

## 4. Конфигурация

**Файл:** `configs/Replica/office0/ActiveSem.py`

```python
slam = dict(
    # ... существующие параметры ...
    
    # CLIP embeddings настройки
    use_clip_embeddings = True,
    clip_model = "ViT-B/32",  # или "ViT-L/14" для лучшего качества
    clip_embed_dim = 64,  # Размерность embedding (32, 64, 128)
)
```

## 5. План экспериментов

### 5.1 Фаза 1: Базовая интеграция (MVP)

**Цель:** Добавить CLIP embeddings без изменения планирования.

**Шаги:**
1. ✅ Создать `CLIPFeatureExtractor`
2. ✅ Добавить `clip_embeddings` в `params`
3. ✅ Обновлять embeddings при добавлении новых Gaussians
4. ✅ Сохранять/загружать embeddings
5. ✅ Тестировать на одной сцене (office0)

**Метрики:**
- Время обучения (должно увеличиться < 10%)
- Память GPU (должна увеличиться на ~embed_dim * N * 4 bytes)
- Качество реконструкции (не должно ухудшиться)

### 5.2 Фаза 2: Querying функциональность

**Цель:** Реализовать текстовый и визуальный querying.

**Шаги:**
1. ✅ Создать `CLIPQueryEngine`
2. ✅ Реализовать `query_text()` и `query_image()`
3. ✅ Добавить визуализацию результатов querying
4. ✅ Тестировать на различных запросах

**Метрики:**
- Precision@K для текстовых запросов
- Recall@K для визуальных запросов
- Время выполнения querying (< 100ms для 1M Gaussians)

### 5.3 Фаза 3: Оптимизация и улучшения

**Цель:** Улучшить качество и производительность.

**Шаги:**
1. Заменить простую интерполяцию на per-pixel features (DINOv2/SAM)
2. Добавить обновление embeddings при refinement
3. Оптимизировать память (quantization, pruning)
4. Добавить batch processing для querying

**Метрики:**
- Качество querying (mAP@K)
- Память (сжатие embeddings)
- Скорость querying

## 6. Ожидаемые результаты

### 6.1 Преимущества

1. **Open-Vocabulary querying:** Поиск объектов по текстовым/визуальным запросам
2. **Zero-shot перенос:** Работа с новыми классами без переобучения
3. **OOD detection:** Обнаружение объектов вне закрытого словаря
4. **Совместимость:** Сохранение стабильности планирования через closed vocab

### 6.2 Ограничения

1. **Память:** +embed_dim * N * 4 bytes (для 1M Gaussians и 64D: +256MB)
2. **Время:** +CLIP encoding время (обычно < 50ms на кадр)
3. **Качество:** Зависит от качества CLIP features и проекции 2D→3D

## 7. Риски и митигация

| Риск | Вероятность | Влияние | Митигация |
|------|-------------|---------|-----------|
| Увеличение памяти > 20% | Средняя | Высокое | Использовать quantization (FP16) |
| Ухудшение качества планирования | Низкая | Среднее | CLIP embeddings не участвуют в planning |
| Медленный querying | Средняя | Низкое | Использовать FAISS для индексации |
| Несовместимость CLIP пространств | Средняя | Высокое | Использовать единую проекцию для всех embeddings |

## 8. Временная оценка

- **Фаза 1 (MVP):** 1-2 недели
- **Фаза 2 (Querying):** 1 неделя
- **Фаза 3 (Оптимизация):** 1-2 недели

**Итого:** 3-5 недель

## 9. Зависимости

```python
# Новые зависимости
torch >= 1.12.0
clip-by-openai  # pip install git+https://github.com/openai/CLIP.git
faiss-cpu  # или faiss-gpu для быстрого querying (опционально)
```

## 10. Следующие шаги

1. Создать ветку `feature/open-vocab-semantics`
2. Реализовать `CLIPFeatureExtractor`
3. Модифицировать `initialize_params_with_seman`
4. Добавить обновление embeddings в `add_new_gaussians_with_seman`
5. Тестировать на office0
6. Измерить метрики и сравнить с baseline





