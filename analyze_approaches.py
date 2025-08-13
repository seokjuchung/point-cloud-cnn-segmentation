#!/usr/bin/env python3
"""
Analysis comparing different approaches for Class 2 detection:
1. Multi-class segmentation (original approach)
2. Binary classification  
3. Autoencoder anomaly detection

This script helps understand which approach works best for your specific use case.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

def load_results():
    """Load results from different approaches"""
    results = {}
    
    # Try to load multi-class results
    try:
        if os.path.exists('inference_results/metrics.json'):
            with open('inference_results/metrics.json', 'r') as f:
                results['multiclass'] = json.load(f)
                print("✅ Loaded multi-class results")
    except:
        print("❌ Multi-class results not found")
    
    # Try to load binary classification results
    try:
        if os.path.exists('inference_results_weighted/weighted_metrics.json'):
            with open('inference_results_weighted/weighted_metrics.json', 'r') as f:
                results['binary_weighted'] = json.load(f)
                print("✅ Loaded binary weighted results")
    except:
        print("❌ Binary weighted results not found")
    
    # Try to load autoencoder results
    try:
        if os.path.exists('anomaly_results/anomaly_detection_results.json'):
            with open('anomaly_results/anomaly_detection_results.json', 'r') as f:
                results['autoencoder'] = json.load(f)
                print("✅ Loaded autoencoder results")
    except:
        print("❌ Autoencoder results not found")
    
    return results

def analyze_class2_performance(results):
    """Compare Class 2 detection performance across methods"""
    print("\n🎯 === CLASS 2 DETECTION PERFORMANCE COMPARISON ===")
    
    comparison_data = {}
    
    # Multi-class approach
    if 'multiclass' in results:
        mc = results['multiclass']
        if 'per_class_metrics' in mc and 'class_2' in mc['per_class_metrics']:
            class2_metrics = mc['per_class_metrics']['class_2']
            comparison_data['Multi-class\nSegmentation'] = {
                'precision': class2_metrics.get('precision', 0),
                'recall': class2_metrics.get('recall', 0),
                'f1_score': class2_metrics.get('f1_score', 0),
                'support': class2_metrics.get('support', 0),
                'approach': 'supervised',
                'training_data': 'all classes'
            }
    
    # Binary weighted approach
    if 'binary_weighted' in results:
        bw = results['binary_weighted']
        if 'per_class_metrics' in bw and 'class_2' in bw['per_class_metrics']:
            class2_metrics = bw['per_class_metrics']['class_2']
            comparison_data['Binary Weighted\nClassification'] = {
                'precision': class2_metrics.get('precision', 0),
                'recall': class2_metrics.get('recall', 0),
                'f1_score': class2_metrics.get('f1_score', 0),
                'support': class2_metrics.get('support', 0),
                'approach': 'supervised',
                'training_data': 'binary labels'
            }
    
    # Autoencoder approach
    if 'autoencoder' in results:
        ae = results['autoencoder']
        # Use F1 optimal threshold performance
        if 'thresholds' in ae and 'F1 optimal' in ae['thresholds']:
            f1_metrics = ae['thresholds']['F1 optimal']
            comparison_data['Autoencoder\nAnomaly Detection'] = {
                'precision': f1_metrics.get('precision', 0),
                'recall': f1_metrics.get('recall', 0),
                'f1_score': f1_metrics.get('f1', 0),
                'support': ae.get('anomaly_samples', 0),
                'approach': 'unsupervised',
                'training_data': 'non-Class-2 only',
                'roc_auc': ae.get('roc_auc', 0),
                'pr_auc': ae.get('pr_auc', 0)
            }
    
    # Display comparison table
    print(f"\n{'Method':<25} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10} {'Approach':<12}")
    print("-" * 85)
    
    for method, metrics in comparison_data.items():
        method_clean = method.replace('\n', ' ')
        print(f"{method_clean:<25} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} "
              f"{metrics['f1_score']:<10.3f} {metrics['support']:<10} {metrics['approach']:<12}")
    
    # Create comparison visualization
    if comparison_data:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        methods = list(comparison_data.keys())
        precisions = [comparison_data[m]['precision'] for m in methods]
        recalls = [comparison_data[m]['recall'] for m in methods]
        f1_scores = [comparison_data[m]['f1_score'] for m in methods]
        
        # Colors for different approaches
        colors = []
        for m in methods:
            if comparison_data[m]['approach'] == 'unsupervised':
                colors.append('orange')  # Autoencoder
            elif 'Binary' in m:
                colors.append('green')   # Binary
            else:
                colors.append('blue')    # Multi-class
        
        # Precision comparison
        axes[0, 0].bar(methods, precisions, color=colors, alpha=0.7)
        axes[0, 0].set_title('Precision for Class 2 Detection')
        axes[0, 0].set_ylabel('Precision')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].set_ylim(0, 1)
        for i, v in enumerate(precisions):
            axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
        
        # Recall comparison
        axes[0, 1].bar(methods, recalls, color=colors, alpha=0.7)
        axes[0, 1].set_title('Recall for Class 2 Detection')
        axes[0, 1].set_ylabel('Recall')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].set_ylim(0, 1)
        for i, v in enumerate(recalls):
            axes[0, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
        
        # F1 Score comparison
        axes[1, 0].bar(methods, f1_scores, color=colors, alpha=0.7)
        axes[1, 0].set_title('F1 Score for Class 2 Detection')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].set_ylim(0, 1)
        for i, v in enumerate(f1_scores):
            axes[1, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
        
        # Precision-Recall scatter plot
        axes[1, 1].scatter(recalls, precisions, c=colors, s=100, alpha=0.7)
        for i, method in enumerate(methods):
            axes[1, 1].annotate(method.replace('\n', ' '), 
                               (recalls[i], precisions[i]),
                               xytext=(5, 5), textcoords='offset points', 
                               fontsize=10, ha='left')
        axes[1, 1].set_xlabel('Recall')
        axes[1, 1].set_ylabel('Precision')
        axes[1, 1].set_title('Precision vs Recall for Class 2')
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('class2_detection_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig('class2_detection_comparison.pdf', bbox_inches='tight')
        plt.show()
        
        print(f"\n📊 Comparison plots saved as class2_detection_comparison.*")
    
    return comparison_data

def analyze_training_requirements(results):
    """Compare training requirements and computational costs"""
    print("\n⚙️ === TRAINING REQUIREMENTS COMPARISON ===")
    
    requirements = {
        'Multi-class Segmentation': {
            'training_data': 'All 5 classes labeled',
            'model_complexity': 'High (5-class decoder)',
            'training_time': 'Long (class imbalance issues)',
            'memory_usage': 'High (large model)',
            'inference_speed': 'Moderate',
            'tuning_difficulty': 'Hard (5 thresholds)'
        },
        'Binary Classification': {
            'training_data': 'Binary labels (Class 2 vs others)',
            'model_complexity': 'Medium (2-class decoder)',
            'training_time': 'Medium (weighted loss)',
            'memory_usage': 'Medium',
            'inference_speed': 'Fast',
            'tuning_difficulty': 'Easy (1 threshold)'
        },
        'Autoencoder Anomaly Detection': {
            'training_data': 'Only non-Class-2 data (unsupervised)',
            'model_complexity': 'Medium (reconstruction)',
            'training_time': 'Medium (stable training)',
            'memory_usage': 'Medium',
            'inference_speed': 'Fast',
            'tuning_difficulty': 'Easy (reconstruction threshold)',
            'special_advantage': 'Works on unlabeled data!'
        }
    }
    
    print(f"\n{'Approach':<30} {'Training Data':<25} {'Complexity':<15} {'Tuning':<12}")
    print("-" * 85)
    for approach, req in requirements.items():
        print(f"{approach:<30} {req['training_data']:<25} {req['model_complexity']:<15} {req['tuning_difficulty']:<12}")
    
    return requirements

def provide_recommendations():
    """Provide recommendations based on different scenarios"""
    print("\n💡 === RECOMMENDATIONS FOR CLASS 2 DETECTION ===")
    
    scenarios = {
        "🎯 **Best Overall Performance**": {
            "recommendation": "Autoencoder Anomaly Detection",
            "reasons": [
                "✅ No need for Class 2 labels during training",
                "✅ Naturally handles extreme class imbalance", 
                "✅ Can process completely unlabeled data",
                "✅ Easy threshold tuning",
                "✅ Interpretable results (reconstruction error)"
            ]
        },
        "📊 **Supervised Learning Preferred**": {
            "recommendation": "Binary Weighted Classification", 
            "reasons": [
                "✅ Direct optimization for Class 2 detection",
                "✅ Faster than multi-class approach",
                "✅ Better handling of class imbalance",
                "✅ Single threshold to tune"
            ]
        },
        "🔬 **Research/Analysis**": {
            "recommendation": "Multi-class Segmentation",
            "reasons": [
                "✅ Provides complete semantic understanding",
                "✅ Can analyze relationships between all classes",
                "✅ Useful for understanding overall data structure"
            ]
        },
        "🚀 **Production Deployment**": {
            "recommendation": "Autoencoder (primary) + Binary (backup)",
            "reasons": [
                "✅ Autoencoder works on unlabeled data",
                "✅ Binary classifier as supervised validation",
                "✅ Fast inference for both approaches",
                "✅ Easy to interpret and debug"
            ]
        }
    }
    
    for scenario, info in scenarios.items():
        print(f"\n{scenario}")
        print(f"   Recommendation: {info['recommendation']}")
        for reason in info['reasons']:
            print(f"   {reason}")

def main():
    print("📊 Class 2 Detection Approach Comparison")
    
    # Load results from different approaches
    results = load_results()
    
    if not results:
        print("\n❌ No results found. Please run the following scripts first:")
        print("   1. python simple_train.py && python inference.py")
        print("   2. python binary_train_class2.py && python binary_inference.py")  
        print("   3. python autoencoder_train.py && python autoencoder_inference.py")
        return
    
    # Analyze Class 2 performance
    comparison_data = analyze_class2_performance(results)
    
    # Analyze training requirements
    requirements = analyze_training_requirements(results)
    
    # Provide recommendations
    provide_recommendations()
    
    # Summary
    print(f"\n🏆 === SUMMARY ===")
    print(f"For your specific use case (0.002% Class 2 points):")
    print(f"")
    print(f"1. **AUTOENCODER** is likely the best approach:")
    print(f"   - Trains only on abundant non-Class-2 data")
    print(f"   - No labeling needed for new data")
    print(f"   - Natural anomaly detection for rare events")
    print(f"")
    print(f"2. **BINARY CLASSIFICATION** as alternative:")
    print(f"   - If you prefer supervised learning")
    print(f"   - Faster training than multi-class")
    print(f"   - Better class imbalance handling")
    print(f"")
    print(f"3. **MULTI-CLASS** for research only:")
    print(f"   - Complete semantic understanding")
    print(f"   - But struggles with extreme imbalance")

if __name__ == "__main__":
    main()
