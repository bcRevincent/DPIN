import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import numpy as np
import random
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel, SWALR
from models.network import GraphPropagation
from tools.utils import get_logger, compute_loss_acc, get_writer
from dataset.dataset import CUB, AWA2, SUN
from dataset import Sampler
from eval import evaluate
import os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from thop import profile
from thop.utils import clever_format
import itertools
import warnings
# os.environ["OMP_NUM_THREADS"] = '1'
# os.environ['PYDEVD_USE_CYTHON'] = 'NO'
warnings.filterwarnings("ignore")


def train_model(config):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(config['manual_seed'])
    np.random.seed(config['manual_seed'])
    torch.manual_seed(config['manual_seed'])
    torch.cuda.manual_seed(config['manual_seed'])
    print(torch.cuda.is_available())
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    config['device'] = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    # config['device'] = torch.device("cpu")
    print(config['device'])

    writer = get_writer(config)
    logger = get_logger(config['log_dir'])
    config['logger'] = logger
    logger.info("\n".join(str(key) + ': ' + str(val) for key, val in config.items()))
    logger.info("Start training on " + config['dataset_name'])

    if config['dataset_name'] == 'AWA2':
        all_data = AWA2(dataroot=config['dataset_root'])
    elif config['dataset_name'] == 'CUB':
        all_data = CUB(dataroot=config['dataset_root'])
    elif config['dataset_name'] == 'SUN':
        all_data = SUN(dataroot=config['dataset_root'])
    else:
        print("DataSet is not found!")
        return
    train_dataset = TensorDataset(all_data.train_x, all_data.train_y)
    train_loader = DataLoader(train_dataset, batch_size=config['train_batch_size'], shuffle=True, drop_last=True)
    test_unseen_dataset = TensorDataset(all_data.test_unseen_x, all_data.test_unseen_y)
    test_unseen_loader = DataLoader(test_unseen_dataset, batch_size=config['test_batch_size'], shuffle=False)
    test_seen_dataset = TensorDataset(all_data.test_seen_x, all_data.test_seen_y)
    test_seen_loader = DataLoader(test_seen_dataset, batch_size=config['test_batch_size'], shuffle=False)
    all_attr = all_data.attr
    seen_attr = all_data.train_attr_pro
    unseen_attr = all_data.test_att_pro


    kmeans = KMeans(n_clusters=config['k_cluster'], random_state=1337).fit(all_attr.cpu().data.numpy())
    config['clusters'] = torch.from_numpy(kmeans.cluster_centers_).float().to(config['device'])
    net = GraphPropagation(
        config=config,
    ).to(config['device'])

    if config['dataset_name'] == 'CUB':
        flops, params = profile(net, inputs=(torch.rand(2, config['visual_feat_num']).to(config['device']), torch.rand(2, config['semantic_feat_num']).to(config['device'])),
                            verbose=False)
        logger.info(f"GFLOPS: {flops / 1e9}")
        logger.info(f"Params: {params / 1e6}")


    writer.add_graph(net, input_to_model=[torch.rand(32, config['visual_feat_num']).to(config['device']), torch.rand(32, config['semantic_feat_num']).to(config['device'])])

    bias = torch.zeros((1, all_attr.size(0)))
    for i, _id in enumerate(all_data.train_id.numpy()):
        bias[0, _id] = config['gamma']
    bias = bias.to(config['device'])

    loss_func = nn.CrossEntropyLoss()
    if config['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(params=net.parameters(), lr=config['lr'], weight_decay=config['weight_decay'], momentum=config['momentum'])
    elif config['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(params=net.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'RMSProp':
        optimizer = torch.optim.RMSprop(params=net.parameters(), lr=config['lr'], weight_decay=config['weight_decay'], momentum=config['momentum'], alpha=config['alpha'])
    else:
        optimizer = torch.optim.AdamW(params=net.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])


    swa_model = AveragedModel(net)
    swa_scheduler = SWALR(optimizer, swa_lr=config['swa_lr'], anneal_epochs=config['swa_annealing_epoch'], anneal_strategy=config['swa_annealing_strategy'])

    last_accuracy = [0.0] * 4
    best_H = [0.0] * 3
    best_e = 0
    for e in range(config['epoch']):
        train_loader = DataLoader(train_dataset, batch_size=config['train_batch_size'], shuffle=True, drop_last=True)
        for step, (batch_x, batch_y) in enumerate(train_loader):
        # for step, batch in enumerate(sampler):
        #     batch_x = all_data.train_x[batch]
        #     batch_y = all_data.train_y[batch]

            net.train()
            optimizer.zero_grad()
            sample_labels = []
            for label in batch_y.numpy():
                if label not in sample_labels:
                    sample_labels.append(label)

            semantic_proto = all_attr[np.array(sample_labels)].to(config['device'])

            visual_proto = []
            for cls in sample_labels:
                proto = batch_x[batch_y == cls].mean(dim=0)
                visual_proto.append(proto)
            visual_proto = torch.stack(visual_proto, dim=0).to(config['device'])

            sample_labels = np.array(sample_labels)

            re_batch_labels = []
            for label in batch_y.cpu().numpy():
                index = np.argwhere(sample_labels == label)
                re_batch_labels.append(index[0][0])
            re_batch_labels = torch.LongTensor(re_batch_labels).to(config['device'])

            batch_x = batch_x.to(config['device'])
            prob = net(batch_x, semantic_proto, 'train')
            loss, accuracy = compute_loss_acc(prob, [net.proj_visual_proto, visual_proto, batch_x], re_batch_labels, loss_func, net, config)
            loss.backward()

            if config['grad_clip_value'] > 0:
                torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=config['grad_clip_value']) ### very good!

            optimizer.step()

            if step % 100 == 0:
                logger.info("Episode: %d  Training loss: %.4f" % (e + 1, loss.cpu().data.numpy()))
                logger.info("{:.4f}".format(accuracy.cpu().data.numpy()))

            writer.add_scalar('total_loss', loss.cpu().data.numpy(), e * len(train_loader) + step)

        swa_model.update_parameters(net)

        if e > config['swa_annealing_epoch']:
            for para in optimizer.param_groups:
                para['lr'] *= config['lr_decay']
        else:
            swa_scheduler.step()

        zsl_accuracy = evaluate(test_unseen_loader, all_data.test_id, unseen_attr, swa_model, net, 0, config)
        gzsl_unseen_accuracy = evaluate(test_unseen_loader, np.arange(all_attr.size(0)),  all_attr, swa_model, net, bias, config)
        gzsl_seen_accuracy = evaluate(test_seen_loader, np.arange(all_attr.size(0)),  all_attr, swa_model, net, bias, config)
        H = 2 * gzsl_seen_accuracy * gzsl_unseen_accuracy / (gzsl_unseen_accuracy + gzsl_seen_accuracy)

        logger.info('Episode: %d zsl: %.4f  gzsl: seen=%.4f, unseen=%.4f, h=%.4f' % (e + 1,
                                                                                     zsl_accuracy,
                                                                                     gzsl_seen_accuracy,
                                                                                     gzsl_unseen_accuracy, H))
        if zsl_accuracy > last_accuracy[0]:
            best_e = e
            last_accuracy = [zsl_accuracy, gzsl_seen_accuracy, gzsl_unseen_accuracy, H]
            if not config['ablation']:
                torch.save(net, config['dataset_name'] + '_vit_net_czsl.pt')
                torch.save(swa_model, config['dataset_name'] + '_swa_vit_net_czsl.pt')
        if H > best_H[-1]:
            best_e = e
            best_H = [gzsl_seen_accuracy, gzsl_unseen_accuracy, H]
            if not config['ablation']:
                torch.save(net, config['dataset_name'] + '_vit_net.pt')
                torch.save(swa_model, config['dataset_name'] + '_swa_vit_net.pt')

        logger.info('BEST: zsl=%.4f, gzsl: seen=%.4f, unseen=%.4f, h=%.4f' % (
        last_accuracy[0], best_H[0], best_H[1], best_H[2]))
        logger.info('----------------------------------------------------')

        writer.add_scalar('zsl', last_accuracy[0], e)
        writer.add_scalar('gzsl', best_H[0], e)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], e)
        print('lr=', optimizer.param_groups[0]['lr'])

        if e - best_e >= 20:
            break
    writer.close()



