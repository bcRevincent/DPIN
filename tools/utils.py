import logging
import torch
import torch.nn as nn
from collections import defaultdict, Counter
import torch.nn.functional as F
from click.core import batch
from openpyxl.styles.builtins import output, total
from torch.utils.tensorboard import SummaryWriter
import os
import shutil


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.handlers.clear()

    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def get_writer(config):
    if os.path.exists(config['writer_path']):
        shutil.rmtree(config['writer_path'])
    os.mkdir(config['writer_path'])
    writer = SummaryWriter(config['writer_path'])

    return writer

def margin_loss(p_s, p_v, p_gt, margin=1.0):
    p_s = F.normalize(p_s, p=2, dim=1)
    p_v = F.normalize(p_v, p=2, dim=1)
    p_gt = F.normalize(p_gt, p=2, dim=1)

    dist_gt2s = torch.cdist(p_gt, p_s, p=2)
    dist_gt2v = torch.cdist(p_gt, p_v, p=2)
    positive_pair = torch.diag(dist_gt2s) + torch.diag(dist_gt2v)
    positive_pair = positive_pair.unsqueeze(1).repeat(1, p_s.size(0))
    negative_pair = dist_gt2s + dist_gt2v

    margin_dist = positive_pair - negative_pair + margin


    margin_dist = margin_dist - torch.diag(margin_dist).diag_embed()
    margin_dist[margin_dist > margin] = positive_pair[margin_dist > margin]

    margin_dist = torch.max(margin_dist, torch.zeros_like(margin_dist))

    return margin_dist.mean()


def group_infoNCE_loss(q, k, y):
    q = F.normalize(q, dim=1) # n x d
    k = F.normalize(k, dim=1) # b x d

    similarity_scores = torch.matmul(q, k.t())  # n x b
    temperature = 0.07 # AWA2 and SUN 0.07
    logits = similarity_scores / temperature

    N = q.size(0)
    labels = torch.arange(N).to(logits.device)
    x_idx = torch.arange(y.size(0)).to(logits.device)

    batch_y = y.unsqueeze(0).repeat(N, 1)
    positive_logits = logits.clone()
    positive_logits = positive_logits.t()[x_idx, y]

    mask_logits = logits.clone()
    mask_logits[labels.view(-1, 1) == y] = -1e8
    negative_logits = mask_logits[y]


    cat_logits = torch.cat((positive_logits.view(-1, 1), negative_logits), dim=-1)
    soft_labels = torch.zeros(k.size(0), dtype=torch.long).to(logits.device)

    loss = F.cross_entropy(cat_logits, soft_labels)

    return loss


def js_div(p_output, q_output, get_softmax=True):
    """
    Function that measures JS divergence between target and output logits:
    """
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')
    if get_softmax:
        p_output = F.softmax(p_output)
        q_output = F.softmax(q_output)
    log_mean_output = ((p_output + q_output) / 2).log()

    return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output)) / 2


def compute_loss_acc(prob, construct_output, batch_y, graph_loss_func, net, config):
    cls_loss = graph_loss_func(prob, batch_y.long())
    cls_acc = torch.eq(torch.argmax(prob[-1], dim=-1), batch_y.long()).float().mean().cpu().data

    construct_loss = F.mse_loss(construct_output[0], construct_output[1])

    group_contrastive_loss = group_infoNCE_loss(construct_output[0], construct_output[2], batch_y)
    proto_margin_loss = margin_loss(net.semantic_proto, net.visual_proto, construct_output[1], config['margin_dist'])
    proto_gt_sim = torch.cosine_similarity(construct_output[1].unsqueeze(1), construct_output[1].unsqueeze(0), dim=-1)
    proto_gt_sim = torch.softmax(proto_gt_sim / 0.07, dim=1)

    edge_loss = F.mse_loss(net.visual_edge, net.semantic_edge)

    # compute total loss
    total_loss = (config['lambda_project_loss'] * construct_loss +
                  config['lambda_contrastive_loss'] * group_contrastive_loss) + config['lambda_margin_loss'] * proto_margin_loss
    total_loss += config['lambda_cls_loss'] * cls_loss + config['lambda_edge_loss'] * edge_loss


    return total_loss, cls_acc