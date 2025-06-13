import itertools
import subprocess
import yaml
from typing import Dict, List, Any, Optional
from datetime import datetime
from pprint import pprint
import os
import sys
import requests
import psutil
import platform


TELEGRAM_BOT_TOKEN = "6586847284:AAG0vs9D0S5TES04HVwvN7tbz9-TgRcOz64"
TELEGRAM_CHAT_ID = "697505256"   
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

BASE_CONFIGS = [
    "configs/bert-base-uncased.yaml",
    "configs/bert-large-uncased.yaml",
    "configs/roberta-base.yaml",
    "configs/roberta-large.yaml",
    "configs/distilbert-base-uncased.yaml",
    "configs/albert-base-v2.yaml"
]

EXPERIMENT_TEMPLATE = {
    "compression": {
        "pruning": {
            "type": "l1_unstructured",
            "amount": 0.3,
            "permanent": True
        },
        "quantization": {
            "type": 16
        },
        "weight_sharing": {
            "algorithm": "kmeans",
            "n_clusters": 32,
            "layers": []
        },
        "low_rank_factorization": {
            "ff_ratio": 0.9,
            "min_ff_rank": 128
        },
        "activation_approximation": {
            "n_segments": 4,
            "replace_all": True,
            "range": "auto"
        }
    }
}


EXPERIMENT_PLANS = [
    # ["pruning"],
    # ["quantization"],
    # ["weight_sharing"],
    # ["low_rank_factorization"],
    # ["activation_approximation"],

    ["pruning", "quantization"],
    ["quantization", "pruning"],
    ["weight_sharing", "quantization"],
    ["quantization", "weight_sharing"],
    ["weight_sharing", "pruning"],
    ["pruning", "weight_sharing"],
]

# HYPERPARAMS = {
#     "pruning": {
#         "type": ["l1_unstructured", "random_unstructured"],
#         "amount": [0.1, 0.3, 0.5, 0.7],
#         "permanent": [True]
#     },
#     "quantization": {
#         "type": [8, 16]
#     },
#     "weight_sharing": {
#         "algorithm": ["kmeans", "uniform"],
#         "n_clusters": [16, 32, 64, 128]
#     },
#     "low_rank_factorization": {
#         "ff_ratio": [0.9, 0.95, 0.97, 0.99],
#         "min_ff_rank": [128]
#     },
#     "activation_approximation": {
#         "n_segments": [2, 4, 8],
#         "replace_all": [True],
#         "range": ["auto"]
#     }
# }

HYPERPARAMS = {
    "pruning": {
        "type": ["l1_unstructured"],
        "amount": [0.1, 0.2, 0.3],
        "permanent": [True]
    },
    "quantization": {
        "type": [8, 16]
    },
    "weight_sharing": {
        "algorithm": ["kmeans"],
        "n_clusters": [32, 64, 128]
    }
}



def send_telegram_message(message: str, parse_mode: Optional[str] = None) -> bool:
    """Send message to Telegram bot"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram bot not configured. Skipping notification.")
        return False
    
    try:
        payload = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'disable_web_page_preview': True
        }
        if parse_mode:
            payload['parse_mode'] = parse_mode
            
        response = requests.post(
            TELEGRAM_API_URL,
            json=payload,
            timeout=10
        )
        response.raise_for_status()
        return True
    except Exception as e:
        print(f"Failed to send Telegram notification: {str(e)}")
        return False
    

def get_cpu_temperature():
    """Получает температуру ЦП (работает на Linux)"""
    try:
        if platform.system() == "Linux":
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                temp = float(f.read()) / 1000.0
            return f"{temp:.1f}°C"
        else:
            # Для Windows/Mac можно использовать другие методы
            return "N/A (Linux only)"
    except:
        return "N/A"


def get_memory_usage():
    """Получает информацию об использовании памяти"""
    mem = psutil.virtual_memory()
    return {
        "total": f"{mem.total / (1024**3):.1f} GB",
        "used": f"{mem.used / (1024**3):.1f} GB",
        "percent": f"{mem.percent}%"
    }


def get_system_status_message():
    """Формирует сообщение о состоянии системы"""
    # Получаем данные
    cpu_temp = get_cpu_temperature()
    memory = get_memory_usage()
    cpu_usage = f"{psutil.cpu_percent()}%"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    hostname = platform.node()
    
    # Формируем сообщение
    message = (
        "System Status Report\n"
        f"{timestamp}\n"
        "CPU:\n"
        f"  - Usage: {cpu_usage}\n"
        f"  - Temp: {cpu_temp}\n\n"
        "Memory:\n"
        f"  - Total: {memory['total']}\n"
        f"  - Used: {memory['used']} ({memory['percent']})\n\n"
        "System monitoring active"
    )
    
    return message


def generate_method_configs(method: str) -> List[Dict[str, Any]]:
    """Генерирует все конфигурации для одного метода"""
    if method not in HYPERPARAMS:
        return [{}]
    
    params = HYPERPARAMS[method]
    keys = params.keys()
    values = [params[key] for key in keys]
    
    configs = []
    for combination in itertools.product(*values):
        config = {}
        for key, value in zip(keys, combination):
            config[key] = value
        configs.append(config)
    
    return configs


def generate_experiment_configs() -> List[Dict[str, Any]]:
    """Генерирует все возможные конфигурации экспериментов"""
    experiment_configs = []
    
    for plan in EXPERIMENT_PLANS:
        # Генерируем все комбинации параметров для каждого метода в плане
        method_configs = []
        for method in plan:
            method_configs.append(generate_method_configs(method))
        
        # Создаем все возможные комбинации методов с их параметрами
        for config_combination in itertools.product(*method_configs):
            config = {"compression": {"order": plan.copy()}}
            
            for method, method_config in zip(plan, config_combination):
                if method in EXPERIMENT_TEMPLATE["compression"]:
                    # Берем базовые значения из шаблона
                    base_config = EXPERIMENT_TEMPLATE["compression"][method].copy()
                    # Обновляем значениями из гиперпараметров
                    base_config.update(method_config)
                    config["compression"][method] = base_config
            
            experiment_configs.append(config)
    
    return experiment_configs


def run_experiments():
    """Запускает все эксперименты последовательно с обработкой ошибок"""
    configs = generate_experiment_configs()
    error_log = "error_log.txt"
    python_path = sys.executable
    
    # Создаем заголовок лога ошибок, если файл не существует
    with open(error_log, 'a') as f:
        f.write(f"\n\n=== Error Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    start_message = (
        "Starting compression experiments\n"
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Total experiments: {len(configs)}\n"
        f"Host: {os.uname().nodename}"
    )
    send_telegram_message(start_message)

    for BASE_CONFIG in BASE_CONFIGS:
        for i, config in enumerate(configs):
            try:
                # Сохраняем конфигурацию во временный файл
                config_path = f"configs/temp_experiment_{i}.yaml"
                with open(config_path, 'w') as f:
                    yaml.dump(config, f)

                experiment_info = (
                    f"🔧 Experiment {i+1}/{len(configs)}\n"
                    f"Methods: {', '.join(config['compression']['order'])}\n"
                    "Config:\n"
                    f"<pre>{yaml.dump(config, indent=2)}</pre>"
                )

                
                # Запускаем эксперимент
                print(f"\nRunning experiment {i+1}/{len(configs)}: {config['compression']['order']}")
                print(yaml.dump(config, indent=2))
                cmd = f"{python_path} main.py --config {BASE_CONFIG} --experiment {config_path}"

                system_message = get_system_status_message()
                send_telegram_message(system_message, parse_mode="HTML")
                send_telegram_message(experiment_info, parse_mode="HTML")
                
                # Запускаем с выводом в реальном времени
                result = subprocess.run(cmd, shell=True)
                
                if result.returncode != 0:
                    raise subprocess.CalledProcessError(
                        returncode=result.returncode,
                        cmd=cmd
                    )
                
                success_message = (
                    f"✅ Experiment {i+1}/{len(configs)} completed successfully\n"
                    f"Methods: {', '.join(config['compression']['order'])}"
                )
                send_telegram_message(success_message)
                    
            except Exception as e:
                # Логируем ошибку
                error_msg = (
                    f"\nExperiment {i+1}/{len(configs)} failed:\n"
                    f"Config: {config}\n"
                    f"Error: {str(e)}\n"
                    f"Command: {cmd}\n"
                    f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    "----------------------------------------\n"
                )
                
                print(f"ERROR: {error_msg}")
                
                with open(error_log, 'a') as f:
                    f.write(error_msg)

                if experiment_info:
                    error_message = (
                        f"❌ Experiment {i+1}/{len(configs)} failed\n"
                        f"Methods: {', '.join(config['compression']['order'])}\n"
                        f"Error: {str(e)}\n"
                        f"Time: {datetime.now().strftime('%H:%M:%S')}"
                    )
                    send_telegram_message(error_message)
                
                continue  # Переходим к следующему эксперименту
            
            finally:
                # Удаляем временный файл конфигурации, если он существует
                try:
                    if 'config_path' in locals():
                        os.remove(config_path)
                except OSError:
                    pass
    
    end_message = (
        "🏁 All experiments completed\n"
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Total experiments: {len(configs)}\n"
        "Check error_log.txt for any failures."
    )
    send_telegram_message(end_message)

    print("\nAll experiments completed. Check error_log.txt for any failures.")


def check_configs():
    configs = generate_experiment_configs()
    for i, config in enumerate(configs):
        pprint(config)
        print('\n\n\n')
    print(len(configs) * len(BASE_CONFIGS))


if __name__ == "__main__":
    check_configs()
    # run_experiments()