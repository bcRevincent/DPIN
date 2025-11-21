import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from pyexpat import features



class ExpertModule(nn.Module):

    def __init__(self, field_center, config):
        super(ExpertModule, self).__init__()
        self.field_center = field_center
        input_size = field_center.shape[0]
        self.fc = nn.Sequential(
            nn.Linear(input_size, config['visual_feat_num']),
            nn.ReLU(),
        )

    def forward(self, semantic_vec):
        input_offsets = semantic_vec - self.field_center
        response = self.fc(input_offsets)

        return response


class CooperationModule(nn.Module):

    def __init__(self, field_centers, config=None):
        super(CooperationModule, self).__init__()
        self.k_filed_centers = field_centers
        self.individuals = nn.ModuleList([])
        for i in range(field_centers.shape[0]):
            self.individuals.append(ExpertModule(field_centers[i], config))


    def forward(self, semantic_vec):

        responses = [indiv(semantic_vec) for indiv in self.individuals]
        feature_anchor = sum(responses)

        return feature_anchor


class UpdateSemanticNode(nn.Module):
    def __init__(self, in_feature, config=None):
        super(UpdateSemanticNode, self).__init__()
        self.config = config
        self.trans = nn.Sequential(
            nn.Linear(in_feature, in_feature),
            nn.LayerNorm(in_feature),
            nn.ReLU(),
        )

    def forward(self, semantic_proto, last_visual_edge, semantic_edge):
        n_proto, n_feat = semantic_proto.size()
        # N * N  x  N * F -> N * F
        extra_semantic_proto = torch.matmul(last_visual_edge, semantic_proto)
        semantic_proto = torch.matmul(semantic_edge, semantic_proto)
        # N * 2F
        if self.config['dist_injection']:
            cat_semantic_proto = semantic_proto + extra_semantic_proto
        else:
            cat_semantic_proto = semantic_proto
        new_semantic_proto = self.trans(cat_semantic_proto) + semantic_proto

        return new_semantic_proto


class UpdateVisualNode(nn.Module):
    def __init__(self, in_feature, config=None):
        super(UpdateVisualNode, self).__init__()
        self.config = config
        self.trans = nn.Sequential(
            nn.Linear(in_feature, in_feature),
            nn.LayerNorm(in_feature),
            nn.ReLU(),
        )

    def forward(self, visual_proto, last_semantic_edge, visual_edge):
        n_proto, n_feat = visual_proto.size()
        # N * N  x  N * F -> N * F
        extra_visual_proto = torch.matmul(last_semantic_edge.float(), visual_proto.float())
        visual_proto = torch.matmul(visual_edge.float(), visual_proto.float())
        # N * 2F
        if self.config['dist_injection']:
            cat_visual_proto = visual_proto + extra_visual_proto
        else:
            cat_visual_proto = visual_proto
        new_visual_proto = self.trans(cat_visual_proto) + visual_proto

        return new_visual_proto


class UpdateSemanticEdge(nn.Module):
    def __init__(self, in_feature, temperature=10.0, config=None):
        super(UpdateSemanticEdge, self).__init__()
        self.config = config
        self.trans = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_feature, in_feature // 16),
            nn.InstanceNorm1d(in_feature // 16),
            nn.ReLU(),
            nn.Linear(in_feature // 16, 1),
            nn.Tanh(),
        )
        self.temperature = temperature

    def forward(self, semantic_proto, last_edge, distance_metric='l2'):
        n_proto, feat_dim = semantic_proto.size()
        vp_i = semantic_proto.unsqueeze(1)
        vp_j = semantic_proto.unsqueeze(0)
        feat = (vp_i - vp_j) ** 2
        feat = feat.view(n_proto * n_proto, feat_dim)
        current_edge = self.trans(feat)
        current_edge = current_edge.view(n_proto, n_proto)
        new_edge = current_edge * (last_edge + 1e-8)
        norm_edge = torch.softmax(new_edge / self.temperature, dim=-1)

        return norm_edge


class UpdateVisualEdge(nn.Module):
    def __init__(self, in_feature, temperature=10.0, config=None):
        super(UpdateVisualEdge, self).__init__()
        self.config = config
        self.trans = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_feature, in_feature // 16),
            nn.InstanceNorm1d(in_feature // 16),
            nn.ReLU(),
            nn.Linear(in_feature // 16, 1),
            nn.Tanh(),
        )
        self.temperature = temperature

    def forward(self, visual_proto, last_edge, distance_metric='l2'):
        n_proto, feat_dim = visual_proto.size()
        vp_i = visual_proto.unsqueeze(1)
        vp_j = visual_proto.unsqueeze(0)
        feat = (vp_i - vp_j) ** 2
        feat = feat.view(n_proto * n_proto, feat_dim)
        current_edge = self.trans(feat.float())
        current_edge = current_edge.view(n_proto, n_proto)
        new_edge = current_edge * (last_edge + 1e-8)
        norm_edge = torch.softmax(new_edge / self.temperature, dim=-1)

        return norm_edge


class FusionLayer(nn.Module):
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(FusionLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat

        self.w_omega = nn.Parameter(torch.FloatTensor(in_feat, out_feat))
        self.u_omega = nn.Parameter(torch.FloatTensor(out_feat, 1))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w_omega)
        torch.nn.init.xavier_uniform_(self.u_omega)

    def forward(self, emb1, emb2):
        emb = []
        emb.append(torch.unsqueeze(torch.squeeze(emb1), dim=1))
        emb.append(torch.unsqueeze(torch.squeeze(emb2), dim=1))
        self.emb = torch.cat(emb, dim=1)
        self.v = F.tanh(torch.matmul(self.emb, self.w_omega))

        self.vu = torch.matmul(self.v, self.u_omega)

        self.alpha = F.softmax(torch.squeeze(self.vu) / 0.01 + 1e-8)

        emb_combined = torch.matmul(torch.transpose(self.emb, 1, 2), torch.unsqueeze(self.alpha, -1))

        out_emb = torch.squeeze(emb_combined)

        return out_emb


class GraphPropagation(nn.Module):
    def __init__(self, config):
        super(GraphPropagation, self).__init__()
        self.visual_feat_num = config['visual_feat_num']
        self.semantic_feat_num = config['semantic_feat_num']
        self.visual_temperature = config['visual_temperature']
        self.semantic_temperature = config['semantic_temperature']
        self.layers = config['layer']
        self.trans_feat_num = config['trans_feat_num']
        self.config = config

        self.cm = CooperationModule(config['clusters'], config)

        update_visual_edge = UpdateVisualEdge(
            in_feature=self.trans_feat_num,
            temperature=self.visual_temperature,
            config=config
        )
        update_visual_node = UpdateVisualNode(
            in_feature=self.trans_feat_num,
            config=config
        )
        update_semantic_edge = UpdateSemanticEdge(
            in_feature=self.trans_feat_num,
            temperature=self.semantic_temperature,
            config=config
        )
        update_semantic_node = UpdateSemanticNode(
            in_feature=self.trans_feat_num,
            config=config
        )

        self.add_module('update_visual_edge', update_visual_edge)
        self.add_module('update_visual_node', update_visual_node)
        self.add_module('update_semantic_edge', update_semantic_edge)
        self.add_module('update_semantic_node', update_semantic_node)

        self.semantic2visual = nn.Sequential(
            nn.Linear(self.trans_feat_num, self.visual_feat_num),
            nn.BatchNorm1d(self.visual_feat_num),
            nn.LeakyReLU(0.2),
            nn.Linear(self.visual_feat_num, self.visual_feat_num),
            nn.InstanceNorm1d(self.visual_feat_num),
            nn.LeakyReLU(0.2),
        )

        self.S2Q = nn.Sequential(
            nn.Linear(self.visual_feat_num, self.visual_feat_num),
            nn.LayerNorm(self.visual_feat_num),
            nn.LeakyReLU(0.2),
        )
        self.S2K = nn.Sequential(
            nn.Linear(self.visual_feat_num, self.visual_feat_num),
            nn.LayerNorm(self.visual_feat_num),
            nn.LeakyReLU(0.2),
        )


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

        self.fusion = FusionLayer(
            in_feat=self.trans_feat_num,
            out_feat=self.trans_feat_num,
        )


    def init_visual_edge(self, prototype):
        proto_i = prototype.unsqueeze(1).float()
        proto_j = prototype.unsqueeze(0).float()
        edge = torch.cosine_similarity(proto_i, proto_j, dim=-1).to(self.config['device'])
        edge = torch.softmax(edge / self.config['init_temp'], dim=-1)

        return edge

    def init_semantic_edge(self, attribute):
        proto_i = attribute.unsqueeze(1).float()
        proto_j = attribute.unsqueeze(0).float()
        edge = torch.cosine_similarity(proto_i, proto_j, dim=-1)
        edge = torch.softmax(edge / self.config['init_temp'], dim=-1)

        return edge

    def propagate_proto(self, visual_proto, semantic_proto, attribute):
        visual_edge = self.init_visual_edge(visual_proto)
        semantic_edge = self.init_semantic_edge(attribute)

        visual_proto = self._modules['update_visual_node'](visual_proto, semantic_edge, visual_edge)
        visual_edge = self._modules['update_visual_edge'](visual_proto, visual_edge)

        if self.config['anchor']:
            semantic_proto = semantic_proto
            semantic_edge = semantic_edge
        else:
            # update node in semantic graph
            semantic_proto = self._modules['update_semantic_node'](semantic_proto, visual_edge, semantic_edge)
            # update edge in semantic graph
            semantic_edge = self._modules['update_semantic_edge'](semantic_proto, semantic_edge)


        self.semantic_proto = semantic_proto
        self.visual_proto = visual_proto
        self.semantic_edge = semantic_edge
        self.visual_edge = visual_edge

    def S2V(self, attr, ref_attr):
        s2v = self.semantic2visual(ref_attr)
        if self.config['plain_proj']:
            return s2v

        s2q = self.S2Q(ref_attr - s2v)
        s2k = self.S2K(ref_attr - s2v)

        attr = F.normalize(attr, dim=1)
        ref_sim = F.normalize(ref_attr, dim=1)
        gt_sim = torch.matmul(attr, attr.t())
        s2q = F.normalize(s2q, dim=1)
        s2k = F.normalize(s2k, dim=1)
        cut_sim = torch.matmul(s2q, s2k.t())

        sim = torch.softmax((gt_sim + cut_sim) / 0.07, dim=1)

        self.s2v_gt_sim = gt_sim
        self.s2v_cut_sim = cut_sim
        self.s2v_sim = sim

        self.init_v_edge = sim

        return torch.matmul(sim, s2v)


    def forward(self, img_feat, attribute, mode='train'):
        semantic_proto = self.cm(attribute)
        proj_visual_proto = self.S2V(attribute, ref_attr=semantic_proto)


        self.proj_visual_proto = proj_visual_proto
        self.init_s_proto = semantic_proto


        self.propagate_proto(proj_visual_proto, semantic_proto, attribute)

        layer_result = []

        visual_proto = self.visual_proto
        semantic_proto = self.semantic_proto

        if self.config['pred_mode'] == 'fusion':
            proto_feat = self.fusion(visual_proto, semantic_proto)
            prob = self.get_prototype_prediction(img_feat, proto_feat)
        elif self.config['pred_mode'] == 'vision':
            prob = self.get_prototype_prediction(img_feat, visual_proto)
        elif self.config['pred_mode'] == 'attribute':
            prob = self.get_prototype_prediction(img_feat, semantic_proto)
        else:
            prob = self.get_prototype_prediction(img_feat, (visual_proto + semantic_proto) / 2.0)


        return prob


    def get_proposing_prototype(self, attribute):
        semantic_proto = self.cm(attribute)
        proj_visual_proto = self.S2V(attribute, semantic_proto)
        self.proj_visual_proto = proj_visual_proto
        self.propagate_proto(proj_visual_proto, semantic_proto, attribute)

        visual_proto = self.visual_proto
        semantic_proto = self.semantic_proto
        proto_feat = self.fusion(visual_proto, semantic_proto)

        return proto_feat


    def get_prototype_prediction(self, instance, prototype):
        proto_feat = F.normalize(prototype, p=0, dim=1)
        img_feat = F.normalize(instance, p=0, dim=1)
        prob = torch.cosine_similarity(img_feat.unsqueeze(1), proto_feat.unsqueeze(0), dim=-1)
        prob = prob * self.config['scale']

        torch.save(prototype.cpu(), 'prototype.pt')

        return prob

