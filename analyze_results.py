#!/usr/bin/env python3
"""
Analysis and visualization of inference results
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_results():
    # Load metrics
    with open('inference_results/metrics.json', 'r') as f:
        metrics = json.load(f)
    
    # Extract confusion matrix
    cm = np.array(metrics['confusion_matrix'])
    
    print("🎯 INFERENCE RESULTS SUMMARY")
    print("=" * 50)
    print(f"📊 Total voxels processed: {metrics['total_voxels']:,}")
    print(f"📈 Overall accuracy (5-class semantic segmentation): {metrics['overall_accuracy']:.3f}")
    print()
    
    # Analyze class distribution in ground truth and predictions
    print("📋 CLASS DISTRIBUTION ANALYSIS")
    print("-" * 40)
    class_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']
    
    gt_counts = cm.sum(axis=1)  # Ground truth counts
    pred_counts = cm.sum(axis=0)  # Prediction counts
    total_samples = gt_counts.sum()
    
    print("Ground Truth Distribution:")
    for i, (name, count) in enumerate(zip(class_names, gt_counts)):
        percentage = (count / total_samples) * 100
        print(f"  {name}: {count:10,} ({percentage:5.2f}%)")
    
    print("\nPrediction Distribution:")
    for i, (name, count) in enumerate(zip(class_names, pred_counts)):
        percentage = (count / total_samples) * 100
        print(f"  {name}: {count:10,} ({percentage:5.2f}%)")
    print()
    
    # Per-class performance
    print("📊 PER-CLASS PERFORMANCE")
    print("-" * 40)
    
    for i, class_name in enumerate(class_names):
        gt_count = gt_counts[i]
        correct_pred = cm[i, i]  # Diagonal element
        
        if gt_count > 0:
            recall = correct_pred / gt_count
            precision = cm[i, i] / pred_counts[i] if pred_counts[i] > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            print(f"{class_name}: Recall={recall:.3f}, Precision={precision:.3f}, F1={f1:.3f} (GT: {gt_count:,})")
        else:
            print(f"{class_name}: No ground truth samples")
    print()
    
    # Key insights
    print("💡 KEY INSIGHTS")
    print("-" * 40)
    
    # Most predicted class
    most_pred_idx = np.argmax(pred_counts)
    print(f"• Most predicted class: {class_names[most_pred_idx]} ({pred_counts[most_pred_idx]:,} voxels)")
    
    # Most common ground truth class
    most_common_idx = np.argmax(gt_counts)
    print(f"• Most common ground truth class: {class_names[most_common_idx]} ({gt_counts[most_common_idx]:,} voxels)")
    
    # Class imbalance
    max_class = np.argmax(gt_counts)
    min_class_counts = gt_counts[gt_counts > 0]  # Only classes with samples
    if len(min_class_counts) > 1:
        min_count = np.min(min_class_counts)
        imbalance_ratio = gt_counts[max_class] / min_count
        print(f"• Class imbalance ratio: {imbalance_ratio:.1f}:1")
    
    # Overall model performance assessment
    if metrics['overall_accuracy'] > 0.8:
        print("• Model shows excellent semantic segmentation performance!")
    elif metrics['overall_accuracy'] > 0.6:
        print("• Model shows good semantic segmentation performance")
    else:
        print("• Model performance could be improved")
    
    print()
    print("🎉 5-Class semantic segmentation analysis completed!")
    print(f"✨ Overall accuracy: {metrics['overall_accuracy']:.1%}")
    
    # Create detailed visualization
    create_detailed_plots(cm, class_names)

def create_detailed_plots(cm, class_names):
    """Create detailed visualization plots"""
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Confusion matrix (counts)
    ax1 = plt.subplot(2, 3, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title('Confusion Matrix (Counts)')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    
    # 2. Normalized confusion matrix
    ax2 = plt.subplot(2, 3, 2)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_title('Normalized Confusion Matrix')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    
    # 3. Class distribution comparison
    ax3 = plt.subplot(2, 3, 3)
    gt_counts = cm.sum(axis=1)
    pred_counts = cm.sum(axis=0)
    
    x = np.arange(len(class_names))
    width = 0.35
    
    ax3.bar(x - width/2, gt_counts, width, label='Ground Truth', alpha=0.7)
    ax3.bar(x + width/2, pred_counts, width, label='Predictions', alpha=0.7)
    
    ax3.set_xlabel('Classes')
    ax3.set_ylabel('Count (Log Scale)')
    ax3.set_title('Ground Truth vs Predictions')
    ax3.set_xticks(x)
    ax3.set_xticklabels(class_names, rotation=45)
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Per-class recall
    ax4 = plt.subplot(2, 3, 4)
    recalls = []
    for i in range(len(cm)):
        gt_count = cm[i].sum()
        correct_pred = cm[i, i]
        recall = correct_pred / gt_count if gt_count > 0 else 0
        recalls.append(recall)
    
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'orange']
    bars = ax4.bar(class_names, recalls, alpha=0.7, color=colors)
    ax4.set_xlabel('Classes')
    ax4.set_ylabel('Recall')
    ax4.set_title('Per-Class Recall')
    ax4.set_ylim(0, 1)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, recall in zip(bars, recalls):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{recall:.3f}', ha='center', va='bottom')
    
    # 5. Per-class precision
    ax5 = plt.subplot(2, 3, 5)
    precisions = []
    for i in range(len(cm)):
        pred_count = cm[:, i].sum()  # Total predictions for this class
        correct_pred = cm[i, i]
        precision = correct_pred / pred_count if pred_count > 0 else 0
        precisions.append(precision)
    
    bars = ax5.bar(class_names, precisions, alpha=0.7, color=colors)
    ax5.set_xlabel('Classes')
    ax5.set_ylabel('Precision')
    ax5.set_title('Per-Class Precision')
    ax5.set_ylim(0, 1)
    ax5.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, precision in zip(bars, precisions):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{precision:.3f}', ha='center', va='bottom')
    
    # 6. Sample distribution pie chart
    ax6 = plt.subplot(2, 3, 6)
    gt_counts = cm.sum(axis=1)
    wedges, texts, autotexts = ax6.pie(gt_counts, labels=class_names, autopct='%1.1f%%',
                                      colors=colors, startangle=90)
    ax6.set_title('Ground Truth Class Distribution')
    
    plt.tight_layout()
    plt.savefig('inference_results/detailed_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('inference_results/detailed_analysis.pdf', bbox_inches='tight')
    print("📊 Detailed analysis plots saved to inference_results/detailed_analysis.*")

if __name__ == "__main__":
    analyze_results()
