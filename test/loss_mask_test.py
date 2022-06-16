import torch

pred = torch.randn((2,3,3)).softmax(dim=-1)
label = torch.LongTensor([[-1,0,1],[-1,0,1]])
loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1)

loss_full = loss_func(pred, label)
loss_12 = loss_func(pred[0:], label[0:])
loss_01 = loss_func(pred[:2], label[:2])
loss_1 = loss_func(pred[1:2], label[1:2])