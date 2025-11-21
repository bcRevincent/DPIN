import torch
import torch.nn as nn
import numpy as np
# from tools.utils import compute_loss_acc
from collections import defaultdict


def compute_average_class_top1_accuracy(pred, label):
    all_label = np.array(label)
    all_pred = np.array(pred)
    all_class = sorted(list(set(all_label.tolist())))
    acc_per_class = []
    for cls in all_class:
        idx = all_label == cls
        aca = np.mean(all_label[idx] == all_pred[idx])
        acc_per_class.append(aca)
    ACA = np.mean(acc_per_class)

    return ACA

def evaluate(loader, map_id, attr, swa_net, net, bias, config):
    # map_id = torch.from_numpy(map_id).long()
    layer_pred, layer_label = [], []
    vision_pred, semantic_pred = [], []
    with torch.no_grad():
        swa_net.eval()
        attr = attr.to(config['device'])
        for batch_x, batch_y in loader:
            layer_output = swa_net(batch_x.to(config['device']), attr, 'eval')
            pred = map_id[torch.argmax(layer_output - bias, dim=-1).cpu().numpy()]
            layer_pred.extend(pred.tolist())
            layer_label.extend(batch_y.cpu().numpy().tolist())

    fusion_acc = compute_average_class_top1_accuracy(layer_pred, layer_label)

    return fusion_acc
