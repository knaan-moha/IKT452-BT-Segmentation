import torch




def calculate_metrics(predication, target, smooth= 1e-6,  threshold = 0.5): 
    """
    This function calculates the segmentation metrics: pixel accuracy, precision, recall, F1-score, IoU, and dice coefficient
    
    arg:
    predication: the predicated mask
    target: the ground truth mask
    threshold: the threshold to binarize the predicated mask
    
    returns: 
    tuple of metrics: 
    pixel accuracy: the pixel accuracy of the predicated mask 
    precision: the precision predicated mask 
    recall: the recall of the predicated mask 
    F1-score: the F1-score of the predicated mask
    IoU: the Intersection over Union of the predicated mask
    dice: the dice coefficient of the predicated mask
    """
    
    #* Binarize the predicated mask and targets 
    
    predication = (predication > threshold).float()
    target = (target > threshold).float()
    
    #* calculating the pixel accuracy
    
    pixel_acc = torch.sum(predication == target).item() /torch.numel(predication) #* 
    
    #* calculating the precision 
    
    true_positive = torch.sum(predication * target)
    false_positive = torch.sum(predication * (1 - target))
    false_negative = torch.sum((1 - predication) * target)
    precision = true_positive / (true_positive + false_positive + smooth)
    
    #* calculating the recall 
    recall = true_positive / (true_positive + false_negative + smooth)
    
    #* calculating the F1-score 
    
    f1_score = 2 * (precision * recall) /  (precision + recall + smooth)
    
    #* calculating the IoU 
    intersection = torch.sum(predication * target)
    union = torch.sum(predication) + torch.sum(target) - intersection
    iou  = intersection / (union + smooth)
    
    #* calculating the dice coefficient 
    
    dice = (2 * intersection + smooth) / (torch.sum(predication) + torch.sum(target) + smooth)
    
    return pixel_acc, precision.item(), recall.item(), f1_score.item(), iou.item(), dice.item()    
    
    