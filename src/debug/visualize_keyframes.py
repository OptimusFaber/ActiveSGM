"""
Скрипт для визуализации keyframes, найденных роботом во время обучения.
Сохраняет RGB изображения keyframes и информацию о них.
"""

import os
import numpy as np
import torch
import cv2
import json
from pathlib import Path
import argparse
from tqdm import tqdm


def load_keyframes_from_checkpoint(checkpoint_dir):
    """
    Загружает keyframes из checkpoint директории.
    
    Args:
        checkpoint_dir: Путь к директории с результатами (например, results/Replica/office0/ActiveSem/run_0)
    
    Returns:
        keyframe_info: dict с информацией о keyframes
    """
    keyframe_info = {
        'keyframe_time_indices': [],
        'keyframe_paths': [],
        'global_keyframe_indices': [],
    }
    
    # Ищем сохраненные keyframe indices
    checkpoint_files = list(Path(checkpoint_dir).glob("**/keyframe_time_indices*.npy"))
    if checkpoint_files:
        # Берем последний checkpoint
        latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
        keyframe_indices = np.load(latest_checkpoint)
        keyframe_info['keyframe_time_indices'] = keyframe_indices.tolist()
        print(f"Загружено {len(keyframe_indices)} keyframes из {latest_checkpoint}")
    
    # Ищем global keyframes (если сохранены отдельно)
    global_kf_files = list(Path(checkpoint_dir).glob("**/global_keyframe_time_indices*.npy"))
    if global_kf_files:
        latest_global = max(global_kf_files, key=lambda x: x.stat().st_mtime)
        global_indices = np.load(latest_global)
        keyframe_info['global_keyframe_indices'] = global_indices.tolist()
        print(f"Загружено {len(global_indices)} global keyframes из {latest_global}")
    
    return keyframe_info


def load_keyframes_from_dataset(dataset_dir, keyframe_indices):
    """
    Загружает RGB изображения keyframes из датасета.
    
    Args:
        dataset_dir: Путь к директории с данными (data/Replica/office0)
        keyframe_indices: Список индексов keyframes
    
    Returns:
        keyframes: dict с RGB изображениями
    """
    keyframes = {}
    results_dir = os.path.join(dataset_dir, "results_habitat")
    
    if not os.path.exists(results_dir):
        print(f"Директория {results_dir} не найдена!")
        return keyframes
    
    # Находим все RGB изображения
    rgb_files = sorted([f for f in os.listdir(results_dir) if f.startswith("frame") and f.endswith(".jpg")])
    
    print(f"Найдено {len(rgb_files)} RGB изображений в {results_dir}")
    print(f"Загружаем {len(keyframe_indices)} keyframes...")
    
    for idx in tqdm(keyframe_indices):
        if idx < len(rgb_files):
            rgb_path = os.path.join(results_dir, rgb_files[idx])
            if os.path.exists(rgb_path):
                img = cv2.imread(rgb_path)
                if img is not None:
                    keyframes[idx] = {
                        'rgb': img,
                        'path': rgb_path,
                        'filename': rgb_files[idx]
                    }
        else:
            print(f"Предупреждение: индекс {idx} выходит за пределы доступных изображений ({len(rgb_files)})")
    
    return keyframes


def visualize_keyframes(keyframes, output_dir, keyframe_info, dataset_name="Replica", scene_name="office0"):
    """
    Визуализирует keyframes и сохраняет их в выходную директорию.
    
    Args:
        keyframes: dict с RGB изображениями keyframes
        output_dir: Директория для сохранения результатов
        keyframe_info: dict с информацией о keyframes
        dataset_name: Название датасета
        scene_name: Название сцены
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Создаем поддиректории
    rgb_dir = os.path.join(output_dir, "keyframes_rgb")
    os.makedirs(rgb_dir, exist_ok=True)
    
    # Сохраняем RGB изображения keyframes
    print(f"\nСохраняем {len(keyframes)} keyframes в {rgb_dir}...")
    for idx, kf_data in tqdm(keyframes.items()):
        output_path = os.path.join(rgb_dir, f"keyframe_{idx:04d}.jpg")
        cv2.imwrite(output_path, kf_data['rgb'])
    
    # Создаем HTML визуализацию
    html_path = os.path.join(output_dir, "keyframes_visualization.html")
    create_html_visualization(keyframes, html_path, keyframe_info, dataset_name, scene_name)
    
    # Сохраняем JSON с информацией
    json_path = os.path.join(output_dir, "keyframes_info.json")
    save_keyframes_info(keyframes, json_path, keyframe_info, dataset_name, scene_name)
    
    print(f"\nВизуализация сохранена в {output_dir}")
    print(f"  - RGB изображения: {rgb_dir}")
    print(f"  - HTML визуализация: {html_path}")
    print(f"  - JSON информация: {json_path}")


def create_html_visualization(keyframes, output_path, keyframe_info, dataset_name, scene_name):
    """Создает HTML файл для визуализации keyframes в браузере."""
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Keyframes Visualization - {dataset_name}/{scene_name}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
        }}
        .info {{
            background-color: white;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .keyframes-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
        }}
        .keyframe-card {{
            background-color: white;
            border-radius: 5px;
            padding: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .keyframe-card img {{
            width: 100%;
            height: auto;
            border-radius: 5px;
        }}
        .keyframe-id {{
            font-weight: bold;
            color: #666;
            margin-top: 10px;
        }}
        .global-keyframe {{
            border: 3px solid #4CAF50;
        }}
    </style>
</head>
<body>
    <h1>Keyframes Visualization</h1>
    <div class="info">
        <p><strong>Dataset:</strong> {dataset_name}</p>
        <p><strong>Scene:</strong> {scene_name}</p>
        <p><strong>Total Keyframes:</strong> {len(keyframes)}</p>
        <p><strong>Global Keyframes:</strong> {len(keyframe_info.get('global_keyframe_indices', []))}</p>
    </div>
    
    <div class="keyframes-grid">
"""
    
    global_kf_set = set(keyframe_info.get('global_keyframe_indices', []))
    
    for idx in sorted(keyframes.keys()):
        kf_data = keyframes[idx]
        is_global = idx in global_kf_set
        global_class = "global-keyframe" if is_global else ""
        global_label = " (Global)" if is_global else ""
        
        # Относительный путь к изображению
        img_rel_path = f"keyframes_rgb/keyframe_{idx:04d}.jpg"
        
        html_content += f"""
        <div class="keyframe-card {global_class}">
            <img src="{img_rel_path}" alt="Keyframe {idx}">
            <div class="keyframe-id">Frame {idx}{global_label}</div>
        </div>
"""
    
    html_content += """
    </div>
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


def save_keyframes_info(keyframes, output_path, keyframe_info, dataset_name, scene_name):
    """Сохраняет информацию о keyframes в JSON."""
    info = {
        'dataset': dataset_name,
        'scene': scene_name,
        'total_keyframes': len(keyframes),
        'keyframe_indices': sorted(keyframes.keys()),
        'global_keyframe_indices': keyframe_info.get('global_keyframe_indices', []),
        'keyframe_details': {}
    }
    
    for idx, kf_data in keyframes.items():
        info['keyframe_details'][idx] = {
            'filename': kf_data['filename'],
            'path': kf_data['path'],
            'is_global': idx in set(keyframe_info.get('global_keyframe_indices', []))
        }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description='Визуализация keyframes из результатов обучения')
    parser.add_argument('--result_dir', type=str, required=True,
                        help='Путь к директории с результатами (например, results/Replica/office0/ActiveSem/run_0)')
    parser.add_argument('--dataset_dir', type=str, 
                        default=None,
                        help='Путь к директории с данными (например, data/Replica/office0). Если не указан, будет выведен из result_dir')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Директория для сохранения визуализации (по умолчанию: result_dir/debug_keyframes)')
    
    args = parser.parse_args()
    
    # Определяем dataset_dir если не указан
    if args.dataset_dir is None:
        # Пытаемся определить из result_dir
        result_path = Path(args.result_dir)
        scene_name = result_path.parent.name
        dataset_name = result_path.parent.parent.name
        args.dataset_dir = f"data/{dataset_name}/{scene_name}"
        print(f"Автоматически определен dataset_dir: {args.dataset_dir}")
    
    # Определяем output_dir если не указан
    if args.output_dir is None:
        args.output_dir = os.path.join(args.result_dir, "debug_keyframes")
    
    # Загружаем информацию о keyframes
    print("Загрузка информации о keyframes...")
    keyframe_info = load_keyframes_from_checkpoint(args.result_dir)
    
    if not keyframe_info['keyframe_time_indices']:
        print("Предупреждение: keyframe indices не найдены в checkpoint. Попробуем загрузить из params.npz...")
        # Попробуем загрузить из params.npz если есть
        params_files = list(Path(args.result_dir).glob("**/params.npz"))
        if params_files:
            print("Найдены params.npz файлы, но keyframe indices должны быть сохранены отдельно.")
            print("Попробуйте запустить обучение с save_checkpoints=True для сохранения keyframes.")
            return
        else:
            print("Не удалось найти информацию о keyframes. Выход.")
            return
    
    # Загружаем RGB изображения keyframes
    print(f"\nЗагрузка RGB изображений из {args.dataset_dir}...")
    keyframes = load_keyframes_from_dataset(args.dataset_dir, keyframe_info['keyframe_time_indices'])
    
    if not keyframes:
        print("Не удалось загрузить keyframes. Проверьте путь к dataset_dir.")
        return
    
    # Визуализируем
    scene_name = Path(args.dataset_dir).name
    dataset_name = Path(args.dataset_dir).parent.name
    visualize_keyframes(keyframes, args.output_dir, keyframe_info, dataset_name, scene_name)
    
    print("\nГотово!")


if __name__ == "__main__":
    main()





