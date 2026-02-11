# Технический план внедрения гибридной семантики (Closed + Open-Vocabulary)

## Текущая архитектура семантики

### Поток данных (как сейчас работает)

```
RGB изображение [H, W, 3]
    ↓
OneFormer → semantic_logits [H, W, num_classes=102]
    ↓
get_pointcloud_with_seman() → pointcloud [N, 3+3+102]
    (xyz + rgb + semantic_logits)
    ↓
initialize_params_with_seman() → params['semantic_logits'] [N, 102]
    ↓
topk(16) → sparse representation
    ↓
params['semantic_logits'] [N, 16] + variables['seman_cls_ids'] [N, 16]
```

### Ключевые функции

1. **`semantic_annotation()`** (`semsplatam.py:147`) - OneFormer сегментация
2. **`get_pointcloud_with_seman()`** (`splatam.py:74`) - создание pointcloud с семантикой
3. **`initialize_params_with_seman()`** (`splatam.py:139`) - инициализация params из pointcloud
4. **`add_new_gaussians_with_seman()`** (`splatam.py:280`) - добавление новых Gaussians при densification

## Технический план внедрения

### Шаг 1: Создать CLIP feature extractor

**Файл:** `src/slam/semsplatam/modified_ver/semantic/clip_extractor.py`

```python
import torch
import clip
from PIL import Image
import numpy as np

class CLIPFeatureExtractor:
    """
    Извлекает CLIP embeddings из RGB изображений для Open-Vocabulary семантики.
    """
    def __init__(self, model_name="ViT-B/32", device="cuda", embed_dim=64):
        """
        Args:
            model_name: CLIP model ("ViT-B/32", "ViT-L/14")
            device: CUDA device
            embed_dim: Целевая размерность embedding (32, 64, 128)
        """
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.embed_dim = embed_dim
        
        # Проекция из CLIP dimension (512 для ViT-B/32) в embed_dim
        clip_dim = self.model.visual.output_dim  # 512 для ViT-B/32
        self.projection = torch.nn.Linear(clip_dim, embed_dim, bias=False).to(device)
        
        # Инициализация проекции (Xavier)
        torch.nn.init.xavier_uniform_(self.projection.weight)
        
        # Заморозить CLIP модель
        for param in self.model.parameters():
            param.requires_grad = False
        self.projection.requires_grad_(False)  # Не обучаем проекцию
    
    @torch.no_grad()
    def extract_per_pixel_features(self, rgb_image: torch.Tensor) -> torch.Tensor:
        """
        Извлекает per-pixel CLIP features через patch-based подход.
        
        Args:
            rgb_image: [H, W, 3] или [3, H, W], диапазон [0, 1]
        
        Returns:
            features: [H, W, embed_dim] CLIP-compatible embeddings
        """
        # Нормализация
        if rgb_image.max() > 1.0:
            rgb_image = rgb_image / 255.0
        
        if rgb_image.shape[0] == 3:  # [3, H, W] -> [H, W, 3]
            rgb_image = rgb_image.permute(1, 2, 0)
        
        H, W = rgb_image.shape[:2]
        
        # Для per-pixel features используем patch-based подход:
        # 1. Разбиваем изображение на патчи
        # 2. Кодируем каждый патч через CLIP
        # 3. Интерполируем обратно в полное разрешение
        
        patch_size = 16  # Размер патча для CLIP ViT
        stride = 8  # Overlap для плавности
        
        # Вычисляем количество патчей
        num_patches_h = (H - patch_size) // stride + 1
        num_patches_w = (W - patch_size) // stride + 1
        
        # Создаем патчи
        patches = []
        patch_positions = []
        
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                y_start = i * stride
                x_start = j * stride
                y_end = min(y_start + patch_size, H)
                x_end = min(x_start + patch_size, W)
                
                patch = rgb_image[y_start:y_end, x_start:x_end]
                
                # Pad если патч меньше patch_size
                if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                    pad_h = patch_size - patch.shape[0]
                    pad_w = patch_size - patch.shape[1]
                    patch = torch.nn.functional.pad(
                        patch, (0, pad_w, 0, pad_h), mode='reflect'
                    )
                
                patches.append(patch)
                patch_positions.append((y_start + y_end) // 2, (x_start + x_end) // 2)
        
        # Кодируем патчи через CLIP
        patch_tensors = []
        for patch in patches:
            pil_patch = Image.fromarray((patch.cpu().numpy() * 255).astype(np.uint8))
            patch_tensor = self.preprocess(pil_patch).unsqueeze(0).to(self.device)
            patch_tensors.append(patch_tensor)
        
        if len(patch_tensors) > 0:
            batch_patches = torch.cat(patch_tensors, dim=0)  # [num_patches, 3, 224, 224]
            
            # CLIP encoding
            patch_features = self.model.encode_image(batch_patches)  # [num_patches, 512]
            patch_features = patch_features / patch_features.norm(dim=-1, keepdim=True)
            
            # Проекция в низкоразмерное пространство
            patch_features_proj = self.projection(patch_features)  # [num_patches, embed_dim]
            patch_features_proj = patch_features_proj / (patch_features_proj.norm(dim=-1, keepdim=True) + 1e-8)
        else:
            # Fallback: используем глобальное изображение
            pil_image = Image.fromarray((rgb_image.cpu().numpy() * 255).astype(np.uint8))
            image_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            global_features = self.model.encode_image(image_tensor)  # [1, 512]
            global_features = global_features / global_features.norm(dim=-1, keepdim=True)
            global_features_proj = self.projection(global_features)  # [1, embed_dim]
            global_features_proj = global_features_proj / (global_features_proj.norm(dim=-1, keepdim=True) + 1e-8)
            
            # Повторяем для всех пикселей
            patch_features_proj = global_features_proj.repeat(H * W, 1)
            patch_positions = [(i, j) for i in range(H) for j in range(W)]
        
        # Интерполируем обратно в полное разрешение
        features_2d = torch.zeros(H, W, self.embed_dim, device=self.device)
        count_map = torch.zeros(H, W, device=self.device)
        
        for idx, (y, x) in enumerate(patch_positions):
            y, x = int(y), int(x)
            if 0 <= y < H and 0 <= x < W:
                features_2d[y, x] += patch_features_proj[idx]
                count_map[y, x] += 1
        
        # Нормализация по количеству патчей
        count_map = count_map.unsqueeze(-1) + 1e-8
        features_2d = features_2d / count_map
        
        # L2 нормализация
        features_2d = features_2d / (features_2d.norm(dim=-1, keepdim=True) + 1e-8)
        
        return features_2d
```

### Шаг 2: Модифицировать get_pointcloud_with_seman

**Файл:** `src/slam/semsplatam/modified_ver/splatam/splatam.py`

```python
def get_pointcloud_with_seman(color, depth, seman, intrinsics, w2c, transform_pts=True,
                   mask=None, compute_mean_sq_dist=False, mean_sq_dist_method="projective",
                   clip_features=None):  # НОВЫЙ ПАРАМЕТР
    """
    Создает pointcloud с RGB, семантикой и CLIP embeddings.
    
    Args:
        color: [3, H, W] RGB изображение
        depth: [1, H, W] depth map
        seman: [C, H, W] semantic logits
        intrinsics: [3, 3] camera intrinsics
        w2c: [4, 4] world-to-camera transformation
        clip_features: [H, W, embed_dim] CLIP embeddings (опционально)
    
    Returns:
        point_cld: [N, 3+3+C+embed_dim] pointcloud (xyz + rgb + seman + clip)
    """
    # ... существующий код для создания pts, cols, seman ...
    
    width, height = color.shape[2], color.shape[1]
    # ... (код создания pts остается без изменений) ...
    
    # Colorize point cloud
    cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3)  # (H * W, 3)
    h, w = seman.shape[1], seman.shape[2]
    seman = torch.permute(seman, (1, 2, 0)).reshape(h*w, -1)  # (H * W, C)
    
    # Добавляем CLIP features
    if clip_features is not None:
        clip_features_flat = clip_features.reshape(-1, clip_features.shape[-1])  # (H * W, embed_dim)
        point_cld = torch.cat((pts, cols, seman, clip_features_flat), -1)  # (H * W, 3+3+C+embed_dim)
    else:
        # Если CLIP features не предоставлены, заполняем нулями
        embed_dim = getattr(get_pointcloud_with_seman, 'default_embed_dim', 64)
        clip_features_flat = torch.zeros(h*w, embed_dim, device=pts.device)
        point_cld = torch.cat((pts, cols, seman, clip_features_flat), -1)
    
    # Select points based on mask
    if mask is not None:
        point_cld = point_cld[mask]
        if compute_mean_sq_dist:
            mean3_sq_dist = mean3_sq_dist[mask]
    
    if compute_mean_sq_dist:
        return point_cld, mean3_sq_dist
    else:
        return point_cld
```

### Шаг 3: Модифицировать initialize_params_with_seman

**Файл:** `src/slam/semsplatam/modified_ver/splatam/splatam.py`

```python
def initialize_params_with_seman(init_pt_cld, num_frames, mean3_sq_dist, 
                                 gaussian_distribution, TOPK=16, clip_embed_dim=64):
    """
    Инициализирует параметры Gaussians с семантикой и CLIP embeddings.
    
    Args:
        init_pt_cld: [N, 3+3+C+embed_dim] pointcloud
        clip_embed_dim: размерность CLIP embedding
    
    Returns:
        params: словарь параметров Gaussians
        variables: словарь переменных
    """
    num_pts = init_pt_cld.shape[0]
    means3D = init_pt_cld[:, :3]  # [N, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1))  # [N, 4]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
    
    if gaussian_distribution == "isotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1))
    elif gaussian_distribution == "anisotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 3))
    else:
        raise ValueError(f"Unknown gaussian_distribution {gaussian_distribution}")
    
    # Определяем размерность семантики (C) из pointcloud
    # pointcloud: [N, 3+3+C+embed_dim]
    # Значит: C = init_pt_cld.shape[1] - 3 - 3 - clip_embed_dim
    seman_dim = init_pt_cld.shape[1] - 3 - 3 - clip_embed_dim
    
    params = {
        'means3D': means3D,
        'rgb_colors': init_pt_cld[:, 3:6],  # [N, 3]
        'semantic_logits': init_pt_cld[:, 6:6+seman_dim],  # [N, C] - Closed vocab
        'clip_embeddings': init_pt_cld[:, 6+seman_dim:6+seman_dim+clip_embed_dim],  # [N, embed_dim] - Open vocab
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': log_scales,
    }
    
    # ... существующий код для camera parameters ...
    cam_rots = np.tile([1, 0, 0, 0], (1, 1))
    cam_rots = np.tile(cam_rots[:, :, None], (1, 1, num_frames))
    params['cam_unnorm_rots'] = cam_rots
    params['cam_trans'] = np.zeros((1, 3, num_frames))
    
    # Sparse representation для semantic_logits (как сейчас)
    _, topk_indices = torch.topk(params['semantic_logits'], k=TOPK, dim=-1)
    dense_shape = params['semantic_logits'].shape
    seman_sparse = create_differentiable_sparse_tensor(params['semantic_logits'], topk_indices, dense_shape)
    
    for k, v in params.items():
        if not isinstance(v, torch.Tensor):
            v = torch.tensor(v)
        
        if k == 'semantic_logits':
            # Sparse representation для closed vocab (как сейчас)
            sparse_values = seman_sparse.coo()[2].reshape(dense_shape[0], TOPK)
            params[k] = torch.nn.Parameter(sparse_values.cuda().float().contiguous().requires_grad_(True))
        elif k == 'clip_embeddings':
            # CLIP embeddings НЕ участвуют в оптимизации
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(False))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))
    
    variables = {
        'seman_cls_ids': seman_sparse.coo()[1].reshape(dense_shape[0], TOPK),
    }
    
    return params, variables
```

### Шаг 4: Модифицировать initialize_new_params_with_seman

**Файл:** `src/slam/semsplatam/modified_ver/splatam/splatam.py`

```python
def initialize_new_params_with_seman(new_pt_cld, mean3_sq_dist, gaussian_distribution, 
                                     TOPK=16, clip_embed_dim=64):
    """
    Инициализирует параметры для новых Gaussians (при densification).
    """
    num_pts = new_pt_cld.shape[0]
    means3D = new_pt_cld[:, :3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1))
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
    
    if gaussian_distribution == "isotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1))
    elif gaussian_distribution == "anisotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 3))
    else:
        raise ValueError(f"Unknown gaussian_distribution {gaussian_distribution}")
    
    # Определяем размерность семантики
    seman_dim = new_pt_cld.shape[1] - 3 - 3 - clip_embed_dim
    
    params = {
        'means3D': means3D,
        'rgb_colors': new_pt_cld[:, 3:6],
        'semantic_logits': new_pt_cld[:, 6:6+seman_dim],
        'clip_embeddings': new_pt_cld[:, 6+seman_dim:6+seman_dim+clip_embed_dim],  # НОВОЕ
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': log_scales,
    }
    
    # Sparse representation для semantic_logits
    _, topk_indices = torch.topk(params['semantic_logits'], k=TOPK, dim=-1)
    dense_shape = params['semantic_logits'].shape
    seman_sparse = create_differentiable_sparse_tensor(params['semantic_logits'], topk_indices, dense_shape)
    
    for k, v in params.items():
        if k == 'semantic_logits':
            sparse_values = seman_sparse.coo()[2].reshape(dense_shape[0], TOPK)
            params[k] = torch.nn.Parameter(sparse_values.float().contiguous().requires_grad_(True))
        elif k == 'clip_embeddings':
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(False))  # НЕ обучаем
        else:
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))
    
    variables = {
        'seman_cls_ids': seman_sparse.coo()[1].reshape(dense_shape[0], TOPK),
    }
    
    return params, variables
```

### Шаг 5: Интегрировать CLIP в SemSplatam

**Файл:** `src/slam/semsplatam/semsplatam.py`

```python
class SemSplatam(SplatamOurs):
    def __init__(self, main_cfg, info_printer, logger):
        # ... существующий код ...
        
        # Инициализация CLIP extractor
        self.clip_extractor = None
        self.clip_embed_dim = 64
        
        if hasattr(main_cfg.slam, 'use_clip_embeddings') and main_cfg.slam.use_clip_embeddings:
            from src.slam.semsplatam.modified_ver.semantic.clip_extractor import CLIPFeatureExtractor
            
            self.clip_embed_dim = getattr(main_cfg.slam, 'clip_embed_dim', 64)
            clip_model = getattr(main_cfg.slam, 'clip_model', "ViT-B/32")
            
            self.clip_extractor = CLIPFeatureExtractor(
                model_name=clip_model,
                device=self.device,
                embed_dim=self.clip_embed_dim
            )
            info_printer(f"CLIP extractor initialized: {clip_model}, embed_dim={self.clip_embed_dim}", 
                        step=0, stage="Initialization")
    
    def init_camera_parameters_from_simulator(self, color, depth, c2w):
        """Initialize using data from simulator (Active mode)"""
        from src.slam.semsplatam.modified_ver.splatam.splatam import (
            get_pointcloud_with_seman, initialize_params_with_seman, setup_camera
        )
        import torch.nn.functional as F
        
        # Get semantic segmentation (как сейчас)
        seg_img = color.clone().to(self.semantic_device)
        _, seman = self.semantic_annotation(seg_img)
        seman = seman.to(self.device)
        
        # НОВОЕ: Извлечь CLIP features
        clip_features = None
        if self.clip_extractor is not None:
            rgb_for_clip = color.permute(2, 0, 1).to(self.device) / 255.0  # [3, H, W]
            clip_features = self.clip_extractor.extract_per_pixel_features(rgb_for_clip)  # [H, W, embed_dim]
        
        # ... существующий код для обработки данных ...
        color_processed = color.permute(2, 0, 1).to(self.device) / 255.0
        depth_processed = depth.unsqueeze(0).to(self.device)
        seman_processed = seman.permute(2, 0, 1).to(self.device)
        
        H, W = color_processed.shape[1], color_processed.shape[2]
        if (seman_processed.shape[1] != H) or (seman_processed.shape[2] != W):
            seman_processed = F.interpolate(seman_processed.unsqueeze(0), (H, W), mode='bilinear')[0]
        
        # Resize CLIP features если нужно
        if clip_features is not None and (clip_features.shape[0] != H or clip_features.shape[1] != W):
            clip_features = F.interpolate(
                clip_features.permute(2, 0, 1).unsqueeze(0), 
                (H, W), mode='bilinear'
            )[0].permute(1, 2, 0)  # [H, W, embed_dim]
        
        w2c = torch.linalg.inv(c2w.to(self.device))
        cam = setup_camera(W, H, intrinsics.cpu().numpy(), w2c.detach().cpu().numpy(), num_channels=self.n_cls)
        
        # Get Initial Point Cloud с CLIP features
        mask = (depth_processed > 0)
        mask = mask.reshape(-1)
        
        init_pt_cld, mean3_sq_dist = get_pointcloud_with_seman(
            color_processed, depth_processed, seman_processed, intrinsics, w2c,
            mask=mask, compute_mean_sq_dist=True,
            mean_sq_dist_method=self.config['mean_sq_dist_method'],
            clip_features=clip_features  # НОВОЕ
        )
        
        # Initialize Parameters с clip_embed_dim
        params, variables = initialize_params_with_seman(
            init_pt_cld, self.num_frames, mean3_sq_dist,
            self.config['gaussian_distribution'], self.topk,
            clip_embed_dim=self.clip_embed_dim  # НОВОЕ
        )
        
        # ... остальной код без изменений ...
```

### Шаг 6: Обновить add_new_gaussians_with_seman

**Файл:** `src/slam/semsplatam/modified_ver/splatam/splatam.py`

```python
def add_new_gaussians_with_seman(params, variables, curr_data, sil_thres,
                                 time_idx, mean_sq_dist_method, gaussian_distribution, 
                                 TOPK=16, clip_extractor=None, clip_embed_dim=64):  # НОВЫЕ ПАРАМЕТРЫ
    """
    Добавляет новые Gaussians при densification, включая CLIP embeddings.
    """
    # ... существующий код для silhouette rendering и определения non_presence_mask ...
    
    if torch.sum(non_presence_mask) > 0:
        # ... существующий код для получения curr_w2c ...
        
        # НОВОЕ: Извлечь CLIP features для текущего кадра
        clip_features = None
        if clip_extractor is not None:
            rgb_for_clip = curr_data['im'].permute(2, 0, 1).to(clip_extractor.device) / 255.0  # [3, H, W]
            clip_features = clip_extractor.extract_per_pixel_features(rgb_for_clip)  # [H, W, embed_dim]
        
        # Get the new pointcloud с CLIP features
        new_pt_cld, mean3_sq_dist = get_pointcloud_with_seman(
            curr_data['im'], curr_data['depth'], curr_data['seman'],
            curr_data['intrinsics'], curr_w2c, 
            mask=non_presence_mask, compute_mean_sq_dist=True,
            mean_sq_dist_method=mean_sq_dist_method,
            clip_features=clip_features  # НОВОЕ
        )
        
        # Initialize new params с clip_embed_dim
        new_params, new_variables = initialize_new_params_with_seman(
            new_pt_cld, mean3_sq_dist, gaussian_distribution, TOPK,
            clip_embed_dim=clip_embed_dim  # НОВОЕ
        )
        
        # Concatenate с существующими params (как сейчас)
        for k, v in new_params.items():
            params[k] = torch.nn.Parameter(torch.cat((params[k], v), dim=0).requires_grad_(params[k].requires_grad))
        
        variables['seman_cls_ids'] = torch.cat((variables['seman_cls_ids'], new_variables['seman_cls_ids']), dim=0)
        
        # ... остальной код без изменений ...
    
    return params, variables
```

### Шаг 7: Обновить вызовы функций

**Файл:** `src/slam/semsplatam/semsplatam.py`

```python
# В методе add_new_gaussians (где вызывается add_new_gaussians_with_seman)
def add_new_gaussians(self, curr_data, time_idx):
    # ... существующий код ...
    
    self.params, self.variables = add_new_gaussians_with_seman(
        self.params, self.variables, curr_data, 
        self.config['mapping']['sil_thres'], time_idx,
        self.config['mean_sq_dist_method'], 
        self.config['gaussian_distribution'],
        self.topk,
        clip_extractor=self.clip_extractor,  # НОВОЕ
        clip_embed_dim=self.clip_embed_dim   # НОВОЕ
    )
```

### Шаг 8: Сохранение и загрузка CLIP embeddings

**Файл:** `src/slam/semsplatam/semsplatam.py`

```python
def print_and_save_result(self, eval_dir_suffix="", ignore_first_frame=False, 
                         save_frames=False, max_frames=None):
    # ... существующий код ...
    
    # Сохранение params (уже включает clip_embeddings)
    params_path = os.path.join(eval_dir, 'params.npz')
    params_to_save = {}
    for k, v in self.params.items():
        if k == 'clip_embeddings':
            params_to_save[k] = v.detach().cpu().contiguous().numpy()  # НОВОЕ
        elif k == 'semantic_logits':
            params_to_save[k] = v.detach().cpu().contiguous().numpy()
            params_to_save['seman_cls_ids'] = self.variables['seman_cls_ids'].detach().cpu().contiguous().numpy()
        # ... остальные параметры ...
    
    np.savez_compressed(params_path, **params_to_save)
```

### Шаг 9: Конфигурация

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

## Итоговая структура данных

### Pointcloud
```
pointcloud [N, 3+3+C+embed_dim]
    - [:, 0:3]   : xyz координаты
    - [:, 3:6]   : RGB цвета
    - [:, 6:6+C] : semantic_logits (closed vocab, C=102)
    - [:, 6+C:]  : clip_embeddings (open vocab, embed_dim=64)
```

### Params
```python
params = {
    'means3D': [N, 3],
    'rgb_colors': [N, 3],
    'semantic_logits': [N, TOPK=16],  # Sparse closed vocab
    'clip_embeddings': [N, embed_dim=64],  # Open vocab (requires_grad=False)
    'unnorm_rotations': [N, 4],
    'logit_opacities': [N, 1],
    'log_scales': [N, 1] or [N, 3],
}
```

### Variables
```python
variables = {
    'seman_cls_ids': [N, TOPK=16],  # Индексы классов для sparse representation
    # ... остальные переменные ...
}
```

## Порядок внедрения

1. ✅ Создать `CLIPFeatureExtractor` класс
2. ✅ Модифицировать `get_pointcloud_with_seman` (добавить clip_features)
3. ✅ Модифицировать `initialize_params_with_seman` (добавить clip_embeddings)
4. ✅ Модифицировать `initialize_new_params_with_seman` (добавить clip_embeddings)
5. ✅ Интегрировать CLIP в `SemSplatam.__init__`
6. ✅ Обновить `init_camera_parameters_from_simulator` (извлечение CLIP features)
7. ✅ Обновить `add_new_gaussians_with_seman` (передача clip_extractor)
8. ✅ Обновить вызовы `add_new_gaussians_with_seman` в `SemSplatam`
9. ✅ Обновить сохранение/загрузку params
10. ✅ Добавить конфигурационные параметры

## Тестирование

1. Проверить, что CLIP features извлекаются корректно
2. Проверить, что pointcloud содержит CLIP features
3. Проверить, что params['clip_embeddings'] инициализируется
4. Проверить, что новые Gaussians получают CLIP embeddings
5. Проверить, что сохранение/загрузка работает
6. Измерить увеличение памяти и времени





