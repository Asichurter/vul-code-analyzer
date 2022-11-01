import torch
import torch.nn.functional as F

def binary_focal_loss(probs, labels, loss_tensor, gamma=2, class_weight=None):
    """
    Binary alpha-balanced focal loss.
    https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf.

    :param probs: Predicted probabilities ranged (0,1)
    :param labels: Real binary labels with either 0 or 1
    :param loss_tensor: Non-reduced loss tensor, such as output of F.binary_cross_entropy(probs, labels, reduction='none')
    :param gamma: Exponential factor of focal loss
    :param class_weight: Balanced class weights, must be in shape (2,)
    """
    modulating_factor = (probs - labels) ** gamma
    label_adapt_term = ((-1) ** labels) ** gamma      # For gamma is even, no need for this term
    non_balanced_focal_loss = label_adapt_term * modulating_factor * loss_tensor
    assert torch.all(non_balanced_focal_loss > 0), 'Fatal error of focal loss, negative loss term found (maybe a bug)'

    if class_weight is not None:
        class_weight_alphas = torch.index_select(class_weight, 0, labels.long())
    else:
        class_weight_alphas = 1

    return torch.mean(non_balanced_focal_loss * class_weight_alphas)

logits = torch.Tensor([4, -4, 2, 0.5])
labels = torch.LongTensor([1, 1, 1, 0]).float()
probs = torch.sigmoid(logits)
loss_tensor = F.binary_cross_entropy(probs, labels, reduction='none')

loss1 = binary_focal_loss(probs, labels, loss_tensor, gamma=2, class_weight=torch.FloatTensor([0.9, 0.1]))
loss2 = binary_focal_loss(probs, labels, loss_tensor, gamma=3)