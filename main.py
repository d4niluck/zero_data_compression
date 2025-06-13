import os
import yaml
import argparse
import json
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from datetime import datetime
import time
import pickle
from pathlib import Path
from transformers import AutoModelForMaskedLM, AutoTokenizer
from datasets import load_dataset
from copy import deepcopy
from pprint import pprint


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def evaluate_model(model, dataset, device):
    model.eval()
    total_loss = 0
    total_samples = 0
    all_embeddings = []
    
    with torch.no_grad():
        for example in dataset:
            inputs = {
                "input_ids": example["input_ids"].unsqueeze(0).to(device),
                "attention_mask": example["attention_mask"].unsqueeze(0).to(device),
                "labels": example["labels"].unsqueeze(0).to(device)
            }
            
            outputs = model(**inputs)
            total_loss += outputs.loss.item()
            total_samples += 1
            
            last_hidden_state = outputs.hidden_states[-1]
            cls_embedding = last_hidden_state[:, 0, :].squeeze().cpu().numpy()
            all_embeddings.append(cls_embedding)
    
    perplexity = np.exp(total_loss / total_samples)
    return perplexity, np.array(all_embeddings)


def measure_latency(model, dataset, device, num_samples=100):
    model.eval()
    latencies = []
    
    with torch.no_grad():
        for i in range(num_samples):
            example = dataset[i]
            inputs = {
                "input_ids": example["input_ids"].unsqueeze(0).to(device),
                "attention_mask": example["attention_mask"].unsqueeze(0).to(device)
            }
            
            start = time.time()
            _ = model(**inputs)
            latencies.append(time.time() - start)
    
    return np.mean(latencies)


def get_model_size(model):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024**2)


def compare_embeddings(emb1, emb2):
    cos_sim = np.mean([cosine_similarity(x.reshape(1,-1), y.reshape(1,-1))[0][0] 
                      for x, y in zip(emb1, emb2)])
    mse = mean_squared_error(emb1, emb2)
    return {"cosine_similarity": cos_sim, "mse": mse}


def preprocess_mlm(examples, tokenizer):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )
    
    input_ids = tokenized["input_ids"]
    labels = input_ids.clone()
    
    # Create masking
    probability_matrix = torch.full(labels.shape, 0.15)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) 
        for val in labels.tolist()
    ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100
    
    # Replace 80% with [MASK]
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    input_ids[indices_replaced] = tokenizer.mask_token_id
    
    # Replace 10% with random
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    input_ids[indices_random] = random_words[indices_random]
    
    tokenized["input_ids"] = input_ids
    tokenized["labels"] = labels
    return tokenized


def apply_compression(model, config):
    
    for method in config.get("order", []):
        if method == "quantization":
            method_params = config.get("quantization", {})
            qtype = {16:torch.float16, 8:torch.qint8}.get(method_params["type"])
            model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=qtype)
            logger.info("Applied dynamic quantization")

        elif method == "pruning":
            method_params = config.get("pruning", {})
            pruning_type = method_params.get("type", "l1_unstructured")
            # l1_unstructured - удаление (зануление) весов с наименьшими абсолютными значениями
            # random_unstructured - удаление случайных весов
            amount = method_params.get("amount", 0.2) # доля весов для обрезки
            layers_to_prune = [
                (module, 'weight') 
                for module in model.modules() 
                if isinstance(module, nn.Linear)
            ]

            for layer, param_name in layers_to_prune:
                if pruning_type == "l1_unstructured":
                    prune.l1_unstructured(layer, name=param_name, amount=amount)
                elif pruning_type == "random_unstructured":
                    prune.random_unstructured(layer, name=param_name, amount=amount)
                else:
                    logger.warning(f"Unknown pruning type: {pruning_type}")
                    continue

                if method_params.get("permanent", True):
                    prune.remove(layer, param_name)
            logger.info("Applied pruning")

        elif method == "weight_sharing":
            method_params = config.get("weight_sharing", {})
            algorithm = method_params.get("algorithm", "kmeans")
            n_clusters = method_params.get("n_clusters", 16)
            layers_to_share = method_params.get("layers", [])
            
            if not layers_to_share:
                layers_to_share = [
                    (name, module) for name, module in model.named_modules()
                    if isinstance(module, (nn.Linear, nn.Conv2d))
                ]
            
            for name, layer in layers_to_share:
                original_weights = layer.weight.data
                flattened_weights = original_weights.view(-1).cpu().numpy()
                
                if algorithm == "kmeans":
                    # Применяем K-means кластеризацию
                    kmeans = KMeans(n_clusters=n_clusters, random_state=505).fit(flattened_weights.reshape(-1, 1))
                    shared_weights = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
                elif algorithm == "uniform":
                    # Равномерное квантование (гистограммный метод)
                    min_val, max_val = flattened_weights.min(), flattened_weights.max()
                    bins = np.linspace(min_val, max_val, n_clusters)
                    indices = np.digitize(flattened_weights, bins) - 1
                    shared_weights = np.array([bins[i] for i in indices])
                else:
                    raise ValueError(f"Unknown weight sharing algorithm: {algorithm}")
                
                # Заменяем веса
                layer.weight.data = torch.tensor(shared_weights, device=original_weights.device).view_as(original_weights)
                layer.weight.requires_grad_(False)
                
            logger.info(f"Applied weight sharing ({algorithm} with {n_clusters} clusters)")

        elif method == "low_rank_factorization":
            method_params = config.get("low_rank_factorization", {})
            ff_ratio = method_params.get("ff_ratio", 0.95)
            min_ff_rank = method_params.get("min_ff_rank", 128)

            model = deepcopy(model)

            for name, module in model.named_modules():
                if isinstance(module, nn.Linear) and ('intermediate.dense' in name or 'output.dense' in name):
                    W = module.weight.data
                    # print(f"Compressing layer: {name} | Original shape: {W.shape}")

                    U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)

                    # Адаптивное отсечение малых сингулярных значений
                    threshold = S[0] * 1e-6
                    valid = S > threshold
                    S, U, Vh = S[valid], U[:, valid], Vh[valid, :]

                    # Расчет ранга
                    energy = torch.cumsum(S / S.sum(), dim=0)
                    effective_rank = torch.sum(energy < ff_ratio).item() + 1
                    new_rank = max(min_ff_rank, effective_rank)
                    # print(f"Effective rank: {effective_rank} | Selected rank: {new_rank}")

                    # Реконструкция
                    A = U[:, :new_rank]
                    B = torch.diag(S[:new_rank]) @ Vh[:new_rank, :]

                    # Замена слоя
                    new_layer = nn.Sequential(
                        nn.Linear(W.size(1), new_rank, bias=False),
                        nn.Linear(new_rank, W.size(0), bias=module.bias is not None)
                    ).to(W.device)

                    new_layer[0].weight.data = B.to(W.dtype)
                    new_layer[1].weight.data = A.to(W.dtype)
                    if module.bias is not None:
                        new_layer[1].bias.data = module.bias.data.clone()

                    # Встраивание
                    parent = model
                    for part in name.split('.')[:-1]:
                        parent = getattr(parent, part)
                    setattr(parent, name.split('.')[-1], new_layer)
            logger.info(f"Applied low rank factorization")

        elif method == "activation_approximation":
            method_params = config.get("activation_approximation", {})
            n_segments = method_params.get("n_segments", 4)  
            replace_all = method_params.get("replace_all", True)
            approx_range = method_params.get("range", "auto")  # "auto" или (min, max)


            def get_activation_range(activation_type):
                """Определяет характерный интервал для аппроксимации"""
                if activation_type in [nn.ReLU, nn.LeakyReLU, 'relu', 'leaky_relu']:
                    return (0, 6)  # ReLU-подобные активации
                elif activation_type in [nn.Sigmoid, 'sigmoid']:
                    return (-4, 4)  # Сигмоида
                elif activation_type in [nn.Tanh, 'tanh']:
                    return (-2, 2)  # Tanh
                elif activation_type in [nn.GELU, 'gelu', 'silu', 'swish']:
                    return (-4, 4)  # GELU/SiLU/Swish
                elif activation_type == 'mish':
                    return (-4, 4)  # Mish
                else:
                    return (-3, 3)  # Универсальный интервал

            def create_pwl_activation(original_activation, n_segments):
                """Создает PWL-аппроксимацию для любой функции активации"""
                activation_type = type(original_activation) if isinstance(original_activation, nn.Module) else original_activation

                # Определяем интервал аппроксимации
                if approx_range == "auto":
                    x_min, x_max = get_activation_range(activation_type)
                else:
                    x_min, x_max = approx_range

                # Генерируем точки излома
                x = torch.linspace(x_min, x_max, n_segments + 1)

                # Вычисляем значения функции в точках излома
                if isinstance(original_activation, nn.Module):
                    y = original_activation(x)
                else:
                    if activation_type == 'gelu':
                        y = x * 0.5 * (1.0 + torch.erf(x / 2.0**(1/2)))
                    elif activation_type in ['silu', 'swish']:
                        y = x * torch.sigmoid(x)
                    elif activation_type == 'mish':
                        y = x * torch.tanh(torch.log(1 + torch.exp(x)))
                    else:
                        raise ValueError(f"Unsupported activation: {activation_type}")

                # Возвращаем функцию с интерполяцией
                def pwl_activation(input):
                    return torch.interp(input, x, y)

                return pwl_activation

            # Заменяем все активации в модели
            for name, module in model.named_modules():
                if replace_all:
                    # Стандартные модули активации
                    if isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh, nn.GELU)):
                        new_activation = create_pwl_activation(module, n_segments)
                        setattr(model, name, new_activation)
                    # Встроенные активации (например, в Transformer'ах)
                    elif hasattr(module, 'activation'):
                        if callable(module.activation):
                            module.activation = create_pwl_activation(module.activation.__name__, n_segments)
            logger.info(f"Applied PWL approximation ({n_segments} segments, range={approx_range})")

        else:
            logger.warning(f"Unknown compression method: {method}")

    return model


def analyze_model_compression(model):
    total_params = 0
    nonzero_params = 0
    all_weights = []
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            param_data = param.detach().cpu().numpy()
            flat_weights = param_data.ravel()
            n_params = flat_weights.size
            n_nonzero = np.count_nonzero(flat_weights)
            
            
            total_params += n_params
            nonzero_params += n_nonzero
            all_weights.extend(flat_weights.tolist())
            
    return {
        'total_params': total_params,
        'nonzero_params': nonzero_params,
        'sparsity': calculate_sparsity(model),
        'weight_min': float(np.min(all_weights)),
        'weight_max': float(np.max(all_weights)),
        'weight_mean': float(np.mean(all_weights)),
        'weight_std': float(np.std(all_weights)),
        'unique_weights': count_unique_weights(model)
    }


def calculate_sparsity(model):
    total = 0
    zeros = 0
    
    for param in model.parameters():
        if param.requires_grad:
            total += param.numel()
            zeros += torch.sum(param == 0).item()
    
    return zeros / total if total > 0 else 0.0


def count_unique_weights(model):
    unique = set()
    
    for param in model.parameters():
        if param.requires_grad:
            unique.update(param.detach().cpu().numpy().ravel().round(decimals=4).tolist())
    
    return len(unique)


def convert_numpy(obj):
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_results(results, parameters, seed, output_dir="results"):
    model_dir = os.path.join(output_dir, parameters['model'])
    os.makedirs(model_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{parameters['model']}_{seed}_{timestamp}.json"

    with open(os.path.join(model_dir, filename), "w") as f:
        json.dump({
            "parameters": parameters,
            "results": results
        }, f, indent=4, default=convert_numpy)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to base config")
    parser.add_argument("--experiment", type=str, required=True, help="Path to experiment config")
    # parser.add_argument("--seed", type=int, required=True, help="Seed")
    return parser.parse_args()


def get_baseline_cache_path(model_name, dataset_name):
    cache_dir = Path("cache/baseline_metrics")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{model_name}_{dataset_name}.pkl"


def load_or_compute_baseline(model, dataset, device, model_name, dataset_name):
    cache_path = get_baseline_cache_path(model_name, dataset_name)
    
    if cache_path.exists():
        logger.info("Loading baseline metrics from cache...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    else:
        logger.info("Evaluating baseline model...")
        perplexity, emb = evaluate_model(model, dataset, device)
        latency = measure_latency(model, dataset, device)
        size = get_model_size(model)
        analyze = analyze_model_compression(model)
        
        results = {
            "perplexity": perplexity,
            "emb": emb,
            "latency": latency,
            "model_size_mb": size,
            "analyze": analyze
        }
        
        with open(cache_path, "wb") as f:
            pickle.dump(results, f)
        
        return results
    

if __name__ == "__main__":
    args = parse_args()

    # Load configs
    base_config = load_config(args.config)
    experiment_config = load_config(args.experiment)
    # base_config["seed"] = args.seed
    seeds = base_config.get("seeds", [])

    for seed in seeds:
        set_seed(seed)
        print('\n')
        logger.info(f"\n############### Experiment {base_config['model']} | seed {seed} ###############")
        print(*experiment_config["compression"].get("order", []))
        print()

        # Setup device
        device = torch.device(base_config["device"])
        logger.info(f"Using device: {device}")

        # Load and preprocess dataset
        tokenizer = AutoTokenizer.from_pretrained(base_config["model"])
        dataset = load_dataset(base_config["dataset"], split="test[:100]")
        dataset = dataset.map(
            lambda x: preprocess_mlm(x, tokenizer),
            batched=True,
            remove_columns=["text"]
        )
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        # Load baseline model
        model = AutoModelForMaskedLM.from_pretrained(
            base_config["model"],
            output_hidden_states=True
        ).to(device)

        # Get or compute baseline metrics
        baseline_results = load_or_compute_baseline(
            model, dataset, device,
            base_config["model"],
            base_config["dataset"]
        )

        # Apply compression
        logger.info("Applying compression...")
        compressed_model = apply_compression(model, experiment_config["compression"])
        compressed_model.to(device)

        # Evaluate compressed model
        logger.info("Evaluating compressed model...")
        compressed_perplexity, compressed_emb = evaluate_model(compressed_model, dataset, device)
        compressed_latency = measure_latency(compressed_model, dataset, device)
        compressed_size = get_model_size(compressed_model)
        compressed_baseline_analyze = analyze_model_compression(compressed_model)

        # Compare embeddings
        comparison = compare_embeddings(baseline_results["emb"], compressed_emb)

        # Prepare results
        results = {
            "baseline": {
                "perplexity": baseline_results["perplexity"],
                "latency": baseline_results["latency"],
                "model_size_mb": baseline_results["model_size_mb"],
                **baseline_results["analyze"]
            },
            "compressed": {
                "perplexity": compressed_perplexity,
                "latency": compressed_latency,
                "model_size_mb": compressed_size,
                **compressed_baseline_analyze
            },
            "comparison": comparison
        }

        # Save results
        save_results(results, {**base_config, **experiment_config}, seed)
        logger.info("Experiment completed successfully!")