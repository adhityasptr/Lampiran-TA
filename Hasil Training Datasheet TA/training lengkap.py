# ====================================================================
# SCRIPT LENGKAP UNTUK MENGUMPULKAN DATA BAB IV
# Sistem Deteksi Rambu Lalu Lintas - YOLO V8
# ====================================================================

import os
import time
import psutil
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import cv2
from PIL import Image
import torch

# ====================================================================
# 1. TRAINING DAN EVALUASI MODEL
# ====================================================================

def train_and_evaluate_model():
    """
    Function untuk training model dan mengumpulkan semua metrics
    """
    print("🚀 Memulai Training Model YOLO V8...")
    
    # Load model
    model = YOLO("yolov8n.pt")
    
    # Training dengan tracking waktu
    start_time = time.time()
    
    results = model.train(
        data="C:/File Personal/Documents/kebutuhan TA/kodingan/Datasheet TA/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device="cpu",                   
        workers=2,
        patience=20,
        optimizer="SGD",
        verbose=True,
        save=True,
        save_period=10,
        cache=True,
        name="traffic_sign_detection"  # Nama eksperimen
    )
    
    training_time = time.time() - start_time
    
    print(f"✅ Training selesai dalam {training_time/3600:.2f} jam")
    
    # Load model terbaik
    best_model = YOLO("runs/detect/traffic_sign_detection/weights/best.pt")
    
    # Evaluasi pada validation set
    val_results = best_model.val()
    
    return best_model, results, val_results, training_time

# ====================================================================
# 2. ANALISIS DATASET
# ====================================================================

def analyze_dataset(data_yaml_path):
    """
    Analisis distribusi dataset
    """
    print("📊 Menganalisis Dataset...")
    
    # Baca file data.yaml
    with open(data_yaml_path, 'r') as file:
        data_config = yaml.safe_load(file)
    
    # Ekstrak informasi dataset
    dataset_info = {
        'train_path': data_config.get('train'),
        'val_path': data_config.get('val'), 
        'test_path': data_config.get('test'),
        'num_classes': data_config.get('nc'),
        'class_names': data_config.get('names')
    }
    
    # Hitung jumlah gambar per split
    def count_images(path):
        if path and os.path.exists(path):
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            count = 0
            for ext in image_extensions:
                count += len(list(Path(path).glob(f"/*{ext}")))
            return count
        return 0
    
    dataset_stats = {
        'train_images': count_images(dataset_info['train_path']),
        'val_images': count_images(dataset_info['val_path']),
        'test_images': count_images(dataset_info['test_path']),
        'total_classes': dataset_info['num_classes'],
        'class_names': dataset_info['class_names']
    }
    
    # Buat tabel distribusi dataset
    df_dataset = pd.DataFrame([
        ['Training', dataset_stats['train_images']],
        ['Validation', dataset_stats['val_images']],
        ['Testing', dataset_stats['test_images']],
        ['Total', sum([dataset_stats['train_images'], 
                      dataset_stats['val_images'], 
                      dataset_stats['test_images']])]
    ], columns=['Split', 'Jumlah Gambar'])
    
    print("Dataset Distribution:")
    print(df_dataset.to_string(index=False))
    
    return dataset_stats, df_dataset

# ====================================================================
# 3. PERFORMANCE METRICS EXTRACTION
# ====================================================================

def extract_training_metrics(results_path="runs/detect/traffic_sign_detection"):
    """
    Ekstrak metrics dari hasil training
    """
    print("📈 Mengekstrak Training Metrics...")
    
    results_file = os.path.join(results_path, "results.csv")
    
    if os.path.exists(results_file):
        # Baca hasil training
        df_results = pd.read_csv(results_file)
        
        # Ekstrak metrics penting
        final_metrics = {
            'train_loss': df_results['train/box_loss'].iloc[-1],
            'train_cls_loss': df_results['train/cls_loss'].iloc[-1],
            'train_dfl_loss': df_results['train/dfl_loss'].iloc[-1],
            'val_loss': df_results['val/box_loss'].iloc[-1],
            'val_cls_loss': df_results['val/cls_loss'].iloc[-1],
            'val_dfl_loss': df_results['val/dfl_loss'].iloc[-1],
            'precision': df_results['metrics/precision(B)'].iloc[-1],
            'recall': df_results['metrics/recall(B)'].iloc[-1],
            'map50': df_results['metrics/mAP50(B)'].iloc[-1],
            'map50_95': df_results['metrics/mAP50-95(B)'].iloc[-1]
        }
        
        # Plot training curves
        plt.figure(figsize=(15, 10))
        
        # Loss curves
        plt.subplot(2, 3, 1)
        plt.plot(df_results['epoch'], df_results['train/box_loss'], label='Train Box Loss')
        plt.plot(df_results['epoch'], df_results['val/box_loss'], label='Val Box Loss')
        plt.title('Box Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Classification loss
        plt.subplot(2, 3, 2)
        plt.plot(df_results['epoch'], df_results['train/cls_loss'], label='Train Cls Loss')
        plt.plot(df_results['epoch'], df_results['val/cls_loss'], label='Val Cls Loss')
        plt.title('Classification Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # mAP curves
        plt.subplot(2, 3, 3)
        plt.plot(df_results['epoch'], df_results['metrics/mAP50(B)'], label='mAP@0.5')
        plt.plot(df_results['epoch'], df_results['metrics/mAP50-95(B)'], label='mAP@0.5:0.95')
        plt.title('Mean Average Precision')
        plt.xlabel('Epoch')
        plt.ylabel('mAP')
        plt.legend()
        plt.grid(True)
        
        # Precision & Recall
        plt.subplot(2, 3, 4)
        plt.plot(df_results['epoch'], df_results['metrics/precision(B)'], label='Precision')
        plt.plot(df_results['epoch'], df_results['metrics/recall(B)'], label='Recall')
        plt.title('Precision & Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        
        # Learning rate
        if 'lr/pg0' in df_results.columns:
            plt.subplot(2, 3, 5)
            plt.plot(df_results['epoch'], df_results['lr/pg0'])
            plt.title('Learning Rate')
            plt.xlabel('Epoch')
            plt.ylabel('LR')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return final_metrics, df_results
    
    return None, None

# ====================================================================
# 4. MODEL PERFORMANCE TESTING
# ====================================================================

def test_model_performance(model, test_data_path):
    """
    Test performa model pada test dataset
    """
    print("🧪 Testing Model Performance...")
    
    # Test pada dataset test
    if test_data_path and os.path.exists(test_data_path):
        results = model.val(data=test_data_path)
        
        test_metrics = {
            'test_map50': results.box.map50,
            'test_map50_95': results.box.map,
            'test_precision': results.box.mp,
            'test_recall': results.box.mr
        }
        
        return test_metrics
    
    return None

# ====================================================================
# 5. INFERENCE SPEED TESTING
# ====================================================================

def benchmark_inference_speed(model, num_tests=100):
    """
    Benchmark kecepatan inference
    """
    print("⚡ Benchmarking Inference Speed...")
    
    # Buat dummy image untuk testing
    dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Warm up
    for _ in range(10):
        model.predict(dummy_image, verbose=False)
    
    # Benchmark
    inference_times = []
    
    for i in range(num_tests):
        start_time = time.time()
        results = model.predict(dummy_image, verbose=False)
        end_time = time.time()
        
        inference_times.append((end_time - start_time) * 1000)  # Convert to ms
    
    speed_stats = {
        'mean_inference_time': np.mean(inference_times),
        'std_inference_time': np.std(inference_times),
        'min_inference_time': np.min(inference_times),
        'max_inference_time': np.max(inference_times),
        'fps': 1000 / np.mean(inference_times)
    }
    
    return speed_stats, inference_times

# ====================================================================
# 6. MODEL SIZE ANALYSIS
# ====================================================================

def analyze_model_size(model_path):
    """
    Analisis ukuran model
    """
    print("📏 Menganalisis Model Size...")
    
    if os.path.exists(model_path):
        # Ukuran file
        file_size_bytes = os.path.getsize(model_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
        
        # Load model untuk parameter count
        model = torch.load(model_path, map_location='cpu')
        
        # Hitung parameter
        if isinstance(model, dict) and 'model' in model:
            model_state = model['model']
            total_params = sum(p.numel() for p in model_state.parameters())
        else:
            total_params = "Unknown"
        
        model_info = {
            'file_size_mb': file_size_mb,
            'file_size_bytes': file_size_bytes,
            'total_parameters': total_params
        }
        
        return model_info
    
    return None

# ====================================================================
# 7. HARDWARE MONITORING
# ====================================================================

def monitor_hardware_during_inference(model, duration_seconds=60):
    """
    Monitor penggunaan hardware selama inference
    """
    print("🖥 Monitoring Hardware Performance...")
    
    # Lists untuk menyimpan data monitoring
    cpu_usage = []
    memory_usage = []
    timestamps = []
    
    # Dummy image untuk testing
    dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    start_time = time.time()
    
    while time.time() - start_time < duration_seconds:
        # Monitor hardware
        cpu_percent = psutil.cpu_percent(interval=None)
        memory_info = psutil.virtual_memory()
        memory_percent = memory_info.percent
        
        # Run inference
        model.predict(dummy_image, verbose=False)
        
        # Store data
        cpu_usage.append(cpu_percent)
        memory_usage.append(memory_percent)
        timestamps.append(time.time() - start_time)
        
        time.sleep(0.1)  # 100ms interval
    
    hardware_stats = {
        'avg_cpu_usage': np.mean(cpu_usage),
        'max_cpu_usage': np.max(cpu_usage),
        'avg_memory_usage': np.mean(memory_usage),
        'max_memory_usage': np.max(memory_usage)
    }
    
    # Plot hardware usage
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(timestamps, cpu_usage)
    plt.title('CPU Usage During Inference')
    plt.xlabel('Time (seconds)')
    plt.ylabel('CPU Usage (%)')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(timestamps, memory_usage)
    plt.title('Memory Usage During Inference')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Memory Usage (%)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('hardware_monitoring.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return hardware_stats

# ====================================================================
# 8. GENERATE COMPREHENSIVE REPORT
# ====================================================================

def generate_bab4_report():
    """
    Generate laporan lengkap untuk BAB IV
    """
    print("📋 Generating Comprehensive BAB IV Report...")
    
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'training_config': {},
        'dataset_stats': {},
        'training_metrics': {},
        'model_info': {},
        'performance_metrics': {},
        'hardware_stats': {}
    }
    
    # Training Configuration
    report['training_config'] = {
        'model_variant': 'YOLOv8n',
        'epochs': 100,
        'batch_size': 16,
        'image_size': 640,
        'optimizer': 'SGD',
        'device': 'CPU',
        'workers': 2,
        'patience': 20
    }
    
    print("📊 1. Analyzing Dataset...")
    dataset_stats, df_dataset = analyze_dataset("C:/File Personal/Documents/kebutuhan TA/kodingan/Datasheet TA/data.yaml")
    report['dataset_stats'] = dataset_stats
    
    print("🚀 2. Training Model...")
    model, training_results, val_results, training_time = train_and_evaluate_model()
    
    print("📈 3. Extracting Training Metrics...")
    final_metrics, df_results = extract_training_metrics()
    if final_metrics:
        report['training_metrics'] = final_metrics
    
    print("📏 4. Analyzing Model Size...")
    model_info = analyze_model_size("runs/detect/traffic_sign_detection/weights/best.pt")
    if model_info:
        report['model_info'] = model_info
    
    print("⚡ 5. Benchmarking Inference Speed...")
    speed_stats, inference_times = benchmark_inference_speed(model)
    report['performance_metrics']['speed'] = speed_stats
    
    print("🧪 6. Testing Model Performance...")
    test_metrics = test_model_performance(model, dataset_stats.get('test_path'))
    if test_metrics:
        report['performance_metrics']['test'] = test_metrics
    
    print("🖥 7. Monitoring Hardware...")
    hardware_stats = monitor_hardware_during_inference(model)
    report['hardware_stats'] = hardware_stats
    
    # Save report
    import json
    with open('bab4_report.json', 'w') as f:
        json.dump(report, f, indent=4, default=str)
    
    print("✅ Report Generated: bab4_report.json")
    
    return report

# ====================================================================
# 9. CREATE TABLES FOR BAB IV
# ====================================================================

def create_bab4_tables(report):
    """
    Buat tabel-tabel untuk BAB IV
    """
    print("📊 Creating Tables for BAB IV...")
    
    # Tabel 1: Training Configuration
    training_config_df = pd.DataFrame([
        ['Model Architecture', report['training_config']['model_variant']],
        ['Epochs', report['training_config']['epochs']],
        ['Batch Size', report['training_config']['batch_size']],
        ['Image Size', f"{report['training_config']['image_size']}x{report['training_config']['image_size']}"],
        ['Optimizer', report['training_config']['optimizer']],
        ['Device', report['training_config']['device']],
        ['Workers', report['training_config']['workers']],
        ['Patience', report['training_config']['patience']]
    ], columns=['Parameter', 'Value'])
    
    # Tabel 2: Model Performance
    if 'training_metrics' in report:
        performance_df = pd.DataFrame([
            ['mAP@0.5', f"{report['training_metrics']['map50']:.3f}"],
            ['mAP@0.5:0.95', f"{report['training_metrics']['map50_95']:.3f}"],
            ['Precision', f"{report['training_metrics']['precision']:.3f}"],
            ['Recall', f"{report['training_metrics']['recall']:.3f}"],
            ['F1-Score', f"{2 * report['training_metrics']['precision'] * report['training_metrics']['recall'] / (report['training_metrics']['precision'] + report['training_metrics']['recall']):.3f}"]
        ], columns=['Metric', 'Value'])
    
    # Tabel 3: Hardware Performance
    if 'hardware_stats' in report:
        hardware_df = pd.DataFrame([
            ['Average CPU Usage', f"{report['hardware_stats']['avg_cpu_usage']:.1f}%"],
            ['Maximum CPU Usage', f"{report['hardware_stats']['max_cpu_usage']:.1f}%"],
            ['Average Memory Usage', f"{report['hardware_stats']['avg_memory_usage']:.1f}%"],
            ['Maximum Memory Usage', f"{report['hardware_stats']['max_memory_usage']:.1f}%"]
        ], columns=['Metric', 'Value'])
    
    # Tabel 4: Speed Performance
    if 'performance_metrics' in report and 'speed' in report['performance_metrics']:
        speed_df = pd.DataFrame([
            ['Mean Inference Time', f"{report['performance_metrics']['speed']['mean_inference_time']:.1f} ms"],
            ['Standard Deviation', f"{report['performance_metrics']['speed']['std_inference_time']:.1f} ms"],
            ['Minimum Inference Time', f"{report['performance_metrics']['speed']['min_inference_time']:.1f} ms"],
            ['Maximum Inference Time', f"{report['performance_metrics']['speed']['max_inference_time']:.1f} ms"],
            ['Average FPS', f"{report['performance_metrics']['speed']['fps']:.1f}"]
        ], columns=['Metric', 'Value'])
    
    # Save all tables
    with pd.ExcelWriter('bab4_tables.xlsx') as writer:
        training_config_df.to_excel(writer, sheet_name='Training_Config', index=False)
        if 'performance_df' in locals():
            performance_df.to_excel(writer, sheet_name='Model_Performance', index=False)
        if 'hardware_df' in locals():
            hardware_df.to_excel(writer, sheet_name='Hardware_Performance', index=False)
        if 'speed_df' in locals():
            speed_df.to_excel(writer, sheet_name='Speed_Performance', index=False)
    
    print("✅ Tables saved to: bab4_tables.xlsx")

# ====================================================================
# 10. MAIN EXECUTION
# ====================================================================

if _name_ == "_main_":
    print("🎯 STARTING BAB IV DATA COLLECTION")
    print("="*50)
    
    try:
        # Generate comprehensive report
        report = generate_bab4_report()
        
        # Create tables
        create_bab4_tables(report)
        
        print("\n" + "="*50)
        print("✅ BAB IV DATA COLLECTION COMPLETED!")
        print("="*50)
        print("\nFiles Generated:")
        print("📄 bab4_report.json - Comprehensive report")
        print("📊 bab4_tables.xlsx - All tables for BAB IV")
        print("📈 training_curves.png - Training progress charts")
        print("🖥 hardware_monitoring.png - Hardware usage charts")
        print("\nNow you have all the data needed for BAB IV! 🎉")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please check your file paths and try again.")