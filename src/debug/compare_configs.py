"""
Скрипт для сравнения конфигураций и выявления возможных причин ухудшения метрик.
"""

import os
import json
from pathlib import Path
import mmengine
from deepdiff import DeepDiff


def load_config(config_path):
    """Загружает конфигурацию из файла."""
    if config_path.endswith('.py'):
        cfg = mmengine.Config.fromfile(config_path)
        return dict(cfg)
    elif config_path.endswith('.json'):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"Неподдерживаемый формат конфигурации: {config_path}")


def compare_configs(config1_path, config2_path, output_path=None):
    """
    Сравнивает две конфигурации и выводит различия.
    
    Args:
        config1_path: Путь к первой конфигурации (оригинальная)
        config2_path: Путь ко второй конфигурации (текущая)
        output_path: Путь для сохранения отчета (опционально)
    """
    print(f"Загрузка конфигураций...")
    print(f"  Оригинальная: {config1_path}")
    print(f"  Текущая: {config2_path}")
    
    config1 = load_config(config1_path)
    config2 = load_config(config2_path)
    
    # Конвертируем в dict если нужно
    if hasattr(config1, '__dict__'):
        config1 = dict(config1)
    if hasattr(config2, '__dict__'):
        config2 = dict(config2)
    
    # Сравниваем
    diff = DeepDiff(config1, config2, ignore_order=True, verbose_level=2)
    
    print("\n" + "="*80)
    print("РАЗЛИЧИЯ В КОНФИГУРАЦИЯХ")
    print("="*80)
    
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("РАЗЛИЧИЯ В КОНФИГУРАЦИЯХ")
    report_lines.append("="*80)
    report_lines.append(f"\nОригинальная: {config1_path}")
    report_lines.append(f"Текущая: {config2_path}\n")
    
    if not diff:
        print("Конфигурации идентичны!")
        report_lines.append("Конфигурации идентичны!")
    else:
        # Критические различия (могут влиять на метрики)
        critical_keys = [
            'dataset_eval_basedir',
            'mapping_iters', 'refine_map_iter',
            'num_iter',
            'semantic_device',
            'enable_active_planning',
            'max_exploration_steps',
            'eval_during_training',
            'eval_during_training_freq',
        ]
        
        print("\n🔴 КРИТИЧЕСКИЕ РАЗЛИЧИЯ (могут влиять на метрики):")
        report_lines.append("\n🔴 КРИТИЧИЧЕСКИЕ РАЗЛИЧИЯ (могут влиять на метрики):")
        
        critical_found = False
        for key in critical_keys:
            val1 = get_nested_value(config1, key)
            val2 = get_nested_value(config2, key)
            if val1 != val2:
                critical_found = True
                print(f"  {key}:")
                print(f"    Оригинал: {val1}")
                print(f"    Текущее:  {val2}")
                report_lines.append(f"  {key}:")
                report_lines.append(f"    Оригинал: {val1}")
                report_lines.append(f"    Текущее:  {val2}")
        
        if not critical_found:
            print("  Не найдено критических различий")
            report_lines.append("  Не найдено критических различий")
        
        # Все различия
        print("\n📋 ВСЕ РАЗЛИЧИЯ:")
        report_lines.append("\n📋 ВСЕ РАЗЛИЧИЯ:")
        
        if 'values_changed' in diff:
            print("\nИзмененные значения:")
            report_lines.append("\nИзмененные значения:")
            for key, change in diff['values_changed'].items():
                print(f"  {key}:")
                print(f"    Было: {change['old_value']}")
                print(f"    Стало: {change['new_value']}")
                report_lines.append(f"  {key}:")
                report_lines.append(f"    Было: {change['old_value']}")
                report_lines.append(f"    Стало: {change['new_value']}")
        
        if 'dictionary_item_added' in diff:
            print("\nДобавленные параметры:")
            report_lines.append("\nДобавленные параметры:")
            for item in diff['dictionary_item_added']:
                print(f"  {item}")
                report_lines.append(f"  {item}")
        
        if 'dictionary_item_removed' in diff:
            print("\nУдаленные параметры:")
            report_lines.append("\nУдаленные параметры:")
            for item in diff['dictionary_item_removed']:
                print(f"  {item}")
                report_lines.append(f"  {item}")
    
    # Сохраняем отчет
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        print(f"\nОтчет сохранен в {output_path}")
    
    return diff


def get_nested_value(d, key):
    """Получает значение из вложенного словаря по ключу (поддерживает вложенность через точку)."""
    keys = key.split('.')
    value = d
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return None
    return value


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Сравнение конфигураций для выявления причин ухудшения метрик')
    parser.add_argument('--original', type=str, 
                       default='ActiveSGM-Orig/configs/Replica/office0/ActiveSem.py',
                       help='Путь к оригинальной конфигурации')
    parser.add_argument('--current', type=str,
                       default='ActiveSGM/configs/Replica/office0/ActiveSem.py',
                       help='Путь к текущей конфигурации')
    parser.add_argument('--output', type=str, default=None,
                       help='Путь для сохранения отчета (опционально)')
    
    args = parser.parse_args()
    
    # Проверяем существование файлов
    if not os.path.exists(args.original):
        print(f"Ошибка: файл {args.original} не найден!")
        return
    
    if not os.path.exists(args.current):
        print(f"Ошибка: файл {args.current} не найден!")
        return
    
    compare_configs(args.original, args.current, args.output)


if __name__ == "__main__":
    main()





