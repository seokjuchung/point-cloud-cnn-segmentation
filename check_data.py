#!/usr/bin/env python3
"""
Analyze class distribution in training labels
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_class_distribution(label_file):
    """Analyze and visualize class distribution in the dataset"""
    print(f"📊 Analyzing class distribution in: {label_file}")
    
    # Load labels
    with h5py.File(label_file, 'r') as f:
        labels = f['labels'][:]
    
    print(f"📋 Dataset info:")
    print(f"   Number of samples: {len(labels)}")
    
    # Count points per class
    total_points = 0
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    samples_with_class = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    
    # Track samples that contain each class
    for i, sample_labels in enumerate(labels):
        total_points += len(sample_labels)
        
        # Count points per class
        for class_id in range(5):
            count = np.sum(sample_labels == class_id)
            class_counts[class_id] += count
            if count > 0:
                samples_with_class[class_id] += 1
    
    print(f"   Total points across all samples: {total_points:,}")
    print()
    
    # Display results
    print("🎯 CLASS DISTRIBUTION ANALYSIS")
    print("=" * 70)
    print(f"{'Class':<8} {'Points':<12} {'Percentage':<12} {'Samples':<10} {'Sample %':<10}")
    print("-" * 70)
    
    ratios = {}
    for class_id in range(5):
        count = class_counts[class_id]
        percentage = 100 * count / total_points
        sample_count = samples_with_class[class_id]
        sample_percentage = 100 * sample_count / len(labels)
        
        ratios[class_id] = {
            'count': count,
            'percentage': percentage,
            'samples_with_class': sample_count,
            'sample_percentage': sample_percentage
        }
        
        print(f"Class {class_id}  {count:>10,}  {percentage:>10.4f}%  {sample_count:>8}   {sample_percentage:>8.1f}%")
    
    print("-" * 70)
    
    # Calculate ratios between classes
    print("\n📊 CLASS IMBALANCE RATIOS")
    print("-" * 40)
    
    # Most common class as reference (usually Class 0)
    max_class = max(class_counts.keys(), key=lambda k: class_counts[k])
    max_count = class_counts[max_class]
    
    print(f"Reference: Class {max_class} ({max_count:,} points)")
    print()
    
    for class_id in range(5):
        if class_id != max_class:
            ratio = max_count / class_counts[class_id] if class_counts[class_id] > 0 else float('inf')
            print(f"Class {max_class} : Class {class_id} = {ratio:>8.1f} : 1")
    
    # Special focus on rare classes (2 and 3)
    print(f"\n🔍 RARE CLASSES ANALYSIS")
    print("-" * 30)
    
    class2_count = class_counts[2]
    class3_count = class_counts[3]
    
    if class2_count > 0 and class3_count > 0:
        ratio_2_to_3 = class2_count / class3_count
        print(f"Class 2 : Class 3 = {ratio_2_to_3:.2f} : 1")
        print(f"Combined rare classes (2+3): {class2_count + class3_count:,} points ({100*(class2_count + class3_count)/total_points:.4f}%)")
    else:
        print("One or both rare classes have zero points")
    
    # Common classes (0, 1, 4) analysis
    common_classes = [0, 1, 4]
    common_total = sum(class_counts[c] for c in common_classes)
    rare_total = class_counts[2] + class_counts[3]
    
    print(f"\n🏗️ AUTOENCODER TRAINING DATA ANALYSIS")
    print("-" * 45)
    print(f"Common classes (0,1,4): {common_total:,} points ({100*common_total/total_points:.2f}%)")
    print(f"Rare classes (2,3): {rare_total:,} points ({100*rare_total/total_points:.4f}%)")
    
    if rare_total > 0:
        common_to_rare_ratio = common_total / rare_total
        print(f"Common : Rare ratio = {common_to_rare_ratio:.1f} : 1")
    
    print(f"\n💡 IMPLICATIONS FOR MODEL TRAINING:")
    print("-" * 40)
    print(f"✅ Multi-class segmentation: Needs heavy class weighting")
    print(f"✅ Binary classification: Class 2 vs others = extreme imbalance")
    print(f"✅ Autoencoder approach: Perfect for filtering common classes")
    print(f"✅ Hybrid approach: Stage 2 will have balanced Class 2 vs 3")
    
    return ratios

def main():
    label_file = "/nevis/houston/home/sc5303/INSS_2025/point-cloud-cnn-segmentation/data/train_label_1e4.h5"
    
    # Check if file exists
    if not os.path.exists(label_file):
        print(f"❌ File not found: {label_file}")
        return
    
    # Analyze class distribution
    ratios = analyze_class_distribution(label_file)

if __name__ == "__main__":
    main()
