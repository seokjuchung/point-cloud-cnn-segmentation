def compute_accuracy(predictions, labels):
    correct = (predictions == labels).sum()
    total = labels.size
    accuracy = correct / total
    return accuracy

def compute_iou(predictions, labels, num_classes):
    iou = []
    for cls in range(num_classes):
        intersection = ((predictions == cls) & (labels == cls)).sum()
        union = ((predictions == cls) | (labels == cls)).sum()
        iou.append(intersection / union if union > 0 else 0)
    return iou

def compute_mean_iou(iou):
    return sum(iou) / len(iou) if len(iou) > 0 else 0