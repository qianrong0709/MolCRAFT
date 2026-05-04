import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import radius_graph, knn_graph
from torch_scatter import scatter_softmax, scatter_sum

from core.models.common import GaussianSmearing, MLP, batch_hybrid_edge_connection, outer_product

# class ED2HAttLayer(nn.Module):
#     """
#     Point-Cloud-to-Hidden cross-attention:
#     Query  : node hidden feature h_i (pocket or ligand)
#     Key    : ED embedding f_j
#     Value  : ED embedding f_j
#     Geometry: relative coordinate (ED_xyz - node_xyz)
#     """
#
#     def __init__(self, h_dim, ed_dim, hidden_dim=64, heads=4):
#         """
#         h_dim: 外部的特征维度，就是输入特征h_i的维度
#         hidden_dim：模型内部自己的中间层维度
#         """
#         super().__init__()
#         self.heads = heads
#         self.scale = (hidden_dim // heads) ** -0.5
#
#         # Q: from node feature (dynamic)
#         self.to_q = nn.Linear(h_dim, hidden_dim, bias=False)
#
#         # K,V: from ED embedding (static)
#         self.to_k = nn.Linear(ed_dim, hidden_dim, bias=False)
#         self.to_v = nn.Linear(ed_dim, hidden_dim, bias=False)
#
#         # geometry embedding
#         self.geo_mlp = nn.Sequential(
#             nn.Linear(3, hidden_dim),
#             nn.SiLU(),
#             nn.Linear(hidden_dim, hidden_dim)
#         )
#
#         # Final projection
#         self.out_proj = nn.Linear(hidden_dim, h_dim)
#
#     def forward(self, node_h, node_x, ed_embed, ed_xyz, neighbor_idx):
#         """
#         node_h: (N, h_dim)
#         node_x: (N, 3)
#         ed_embed: (M, ed_dim)
#         ed_xyz: (M, 3)
#         neighbor_idx: list of tensors, each is neighbors of node i in ED cloud
#         """
#         N, H = node_h.size()
#         M, D = ed_embed.size()
#
#         out_h = torch.zeros_like(node_h)
#
#         # Pre-compute keys and values
#         K_all = self.to_k(ed_embed)  # (M, hidden_dim)
#         V_all = self.to_v(ed_embed)  # (M, hidden_dim)
#
#         for i in range(N):
#             q = self.to_q(node_h[i]).unsqueeze(0)  # (1, hidden_dim)
#
#             # find ED neighbors
#             nei = neighbor_idx[i]    # tensor(K,)
#             k = K_all[nei]           # (K, hidden_dim)
#             v = V_all[nei]           # (K, hidden_dim)
#
#             # geometry term
#             rel = ed_xyz[nei] - node_x[i].unsqueeze(0)  # (K, 3)
#             geo = self.geo_mlp(rel)                     # (K, hidden_dim)
#
#             # attention score
#             attn_score = (q * k).sum(-1) * self.scale    # (K,)
#             attn_score = attn_score + geo.sum(-1)        # inject geometry
#
#             # softmax
#             attn = F.softmax(attn_score, dim=0).unsqueeze(1)  # (K,1)
#
#             # weighted sum
#             E_i = (attn * v).sum(0)   # (hidden_dim,)
#             out_h[i] = node_h[i] + self.out_proj(E_i)
#
#         return out_h


class BaseX2HAttLayer(nn.Module):
    """
    带有结构感知能力的多头注意力模块
    将与坐标相关的信息（如距离、边特征）与节点特征一起，生成新的节点表示（H ← X）
    即从几何结构 X → 节点特征 H 的更新，故名 X2H
    从 坐标空间（X）到节点特征空间（H） 的消息传播过程

    节点对之间的几何信息（如距离、边特征、r_feat）
    节点自身特征 h[i], h[j]
    输入边权重 e_w（可以是 learnable，也可以是外部提供）

    """
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, edge_feat_dim, r_feat_dim,
                 act_fn='relu', norm=True, ew_net_type='r', out_fc=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.act_fn = act_fn
        self.edge_feat_dim = edge_feat_dim  # 边特征维度
        self.r_feat_dim = r_feat_dim        # 几何特征维度，就是两个节点之间的距离变换成向量
        self.ew_net_type = ew_net_type      # 控制 e_w 权重来源（r: 依赖距离特征；m: 依赖消息本身）
        self.out_fc = out_fc                # 最后是否加入额外融合层（FC+残差）

        # attention key func
        # key/value 网络的输入维度（两个点的特征 + 边特征 + 几何特征）
        kv_input_dim = input_dim * 2 + edge_feat_dim + r_feat_dim

        # K/V/Q 的 MLP 模块（注意力核心）
        self.hk_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)

        # attention value func
        self.hv_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)

        # attention query func
        self.hq_func = MLP(input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)     # 输入是单个节点的特征 h[i]

        # 边权重网络（控制每条边信息强度）
        if ew_net_type == 'r':
            self.ew_net = nn.Sequential(nn.Linear(r_feat_dim, 1), nn.Sigmoid())
        elif ew_net_type == 'm':
            self.ew_net = nn.Sequential(nn.Linear(output_dim, 1), nn.Sigmoid())

        # 输出融合 FC
        if self.out_fc:
            self.node_output = MLP(2 * hidden_dim, hidden_dim, hidden_dim, norm=norm, act_fn=act_fn)

    def forward(self, h, r_feat, edge_feat, edge_index, e_w=None):
        N = h.size(0)               # 节点数
        src, dst = edge_index       # 边的起点 dst，终点 src（注意是 message 从 src 到 dst）
        hi, hj = h[dst], h[src]     # hi 是接收方，hj 是发送方（信息从 j → i），两个端点的节点特征

        # multi-head attention
        # decide inputs of k_func and v_func
        # 构造 attention 的 K/V 输入
        kv_input = torch.cat([r_feat, hi, hj], -1)
        if edge_feat is not None:
            kv_input = torch.cat([edge_feat, kv_input], -1) # shape 为 [E, kv_input_dim]，其中 E 是边数

        # compute k
        # 得到每条边上的 key 和 value 表示（带有结构和边语义信息）
        k = self.hk_func(kv_input).view(-1, self.n_heads, self.output_dim // self.n_heads)
        # compute v
        v = self.hv_func(kv_input)      # 经过 MLP 后 reshape 成 [E, n_heads, d_per_head]，每条边都含有多个头的 k 和 v 向量

        # 计算边权重 e_w
        if self.ew_net_type == 'r':
            e_w = self.ew_net(r_feat)
        elif self.ew_net_type == 'm':
            e_w = self.ew_net(v[..., :self.hidden_dim])
        elif e_w is not None:
            e_w = e_w.view(-1, 1)
        else:
            e_w = 1.

        # 用边权重修正 v
        v = v * e_w
        v = v.view(-1, self.n_heads, self.output_dim // self.n_heads)

        # compute q
        # 得到每个节点的 query（q），用自身特征 h 生成
        q = self.hq_func(h).view(-1, self.n_heads, self.output_dim // self.n_heads)

        # compute attention weights
        # 注意力打分（点积注意力），并在 dst 上 softmax 聚合
        alpha = scatter_softmax((q[dst] * k / np.sqrt(k.shape[-1])).sum(-1), dst, dim=0,
                                dim_size=N)  # [num_edges, n_heads]

        # perform attention-weighted message-passing
        # 注意力加权聚合
        m = alpha.unsqueeze(-1) * v  # (E, heads, H_per_head)
        output = scatter_sum(m, dst, dim=0, dim_size=N)  # (N, heads, H_per_head)
        output = output.view(-1, self.output_dim)

        # 输出层和残差连接
        if self.out_fc:
            output = self.node_output(torch.cat([output, h], -1))

        output = output + h     # 残差连接
        return output   # 得到 [N, output_dim] 的新节点特征


class BaseH2XAttLayer(nn.Module):
    """
        向量注意力机制
        输入：节点特征 H，结构特征 rel_x、r_feat 等，
        输出：每个节点的坐标更新向量 Δx。
        用 scalar 权重 × rel_x 方向向量，控制坐标的变化方向和幅度
        这个结构其实在模仿力场或势能梯度更新过程


    """
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, edge_feat_dim, r_feat_dim,
                 act_fn='relu', norm=True, ew_net_type='r'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.edge_feat_dim = edge_feat_dim
        self.r_feat_dim = r_feat_dim
        self.act_fn = act_fn
        self.ew_net_type = ew_net_type

        kv_input_dim = input_dim * 2 + edge_feat_dim + r_feat_dim

        self.xk_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        self.xv_func = MLP(kv_input_dim, self.n_heads, hidden_dim, norm=norm, act_fn=act_fn)
        self.xq_func = MLP(input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        if ew_net_type == 'r':
            self.ew_net = nn.Sequential(nn.Linear(r_feat_dim, 1), nn.Sigmoid())

    def forward(self, h, rel_x, r_feat, edge_feat, edge_index, e_w=None):
        N = h.size(0)
        src, dst = edge_index
        hi, hj = h[dst], h[src]

        # multi-head attention
        # decide inputs of k_func and v_func
        kv_input = torch.cat([r_feat, hi, hj], -1)
        if edge_feat is not None:
            kv_input = torch.cat([edge_feat, kv_input], -1)

        k = self.xk_func(kv_input).view(-1, self.n_heads, self.output_dim // self.n_heads)
        v = self.xv_func(kv_input)  # 这里的 v 是不同于 X2H 的 —— 不是向量，而是一个 标量权重 v（每个 head 一维），它稍后会作用在 rel_x 上
        if self.ew_net_type == 'r':
            e_w = self.ew_net(r_feat)
        elif self.ew_net_type == 'm':
            e_w = 1.
        elif e_w is not None:
            e_w = e_w.view(-1, 1)
        else:
            e_w = 1.
        v = v * e_w

        v = v.unsqueeze(-1) * rel_x.unsqueeze(1)  # (xi - xj) [n_edges, n_heads, 3]， 两者相乘后，就得到每条边在几何空间中的“向量更新”，也就是 Δx_ij
        q = self.xq_func(h).view(-1, self.n_heads, self.output_dim // self.n_heads)

        # Compute attention weights
        alpha = scatter_softmax((q[dst] * k / np.sqrt(k.shape[-1])).sum(-1), dst, dim=0, dim_size=N)  # (E, heads)

        # Perform attention-weighted message-passing
        m = alpha.unsqueeze(-1) * v  # (E, heads, 3)
        output = scatter_sum(m, dst, dim=0, dim_size=N)  # (N, heads, 3)
        return output.mean(1)  # [num_nodes, 3]


class AttentionLayerO2TwoUpdateNodeGeneral(nn.Module):
    def __init__(self, hidden_dim, n_heads, num_r_gaussian, edge_feat_dim,
                 # ed_embed_dim=None,   # 新增
                 act_fn='relu', norm=True,
                 num_x2h=1, num_h2x=1, r_min=0., r_max=10., num_node_types=8,
                 ew_net_type='r', x2h_out_fc=True, sync_twoup=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.edge_feat_dim = edge_feat_dim
        self.num_r_gaussian = num_r_gaussian
        self.norm = norm
        self.act_fn = act_fn
        self.num_x2h = num_x2h
        self.num_h2x = num_h2x
        self.r_min, self.r_max = r_min, r_max
        self.num_node_types = num_node_types
        self.ew_net_type = ew_net_type
        self.x2h_out_fc = x2h_out_fc
        self.sync_twoup = sync_twoup

        self.distance_expansion = GaussianSmearing(self.r_min, self.r_max, num_gaussians=num_r_gaussian)

        # self.ed2h = ED2HAttLayer(hidden_dim, ed_dim=ed_embed_dim)

        self.x2h_layers = nn.ModuleList()
        for i in range(self.num_x2h):
            self.x2h_layers.append(
                BaseX2HAttLayer(hidden_dim, hidden_dim, hidden_dim, n_heads, edge_feat_dim,
                                r_feat_dim=num_r_gaussian * 4,  # 为什么是 * 4 ？？？
                                act_fn=act_fn, norm=norm,
                                ew_net_type=self.ew_net_type, out_fc=self.x2h_out_fc)
            )
        self.h2x_layers = nn.ModuleList()
        for i in range(self.num_h2x):
            self.h2x_layers.append(
                BaseH2XAttLayer(hidden_dim, hidden_dim, hidden_dim, n_heads, edge_feat_dim,
                                r_feat_dim=num_r_gaussian * 4,
                                act_fn=act_fn, norm=norm,
                                ew_net_type=self.ew_net_type)
            )

    def forward(self, h, x, edge_attr, edge_index, mask_ligand,
                # ed_embed=None, ed_xyz=None, ed_neighbor=None,   # 新增
                e_w=None, fix_x=False):
        """
        edge_attr: 边的属性
        edge_index: 邻接对
        fix_x: 是否冻结位置（不更新 x）
        """
        src, dst = edge_index
        if self.edge_feat_dim > 0:
            edge_feat = edge_attr  # shape: [#edges_in_batch, #bond_types]
        else:
            edge_feat = None

        rel_x = x[dst] - x[src] # 计算每条边的向量
        dist = torch.norm(rel_x, p=2, dim=-1, keepdim=True) # 计算每条边的欧式距离

        # # new ed点云更新 h 特征
        # if ed_embed is not None:
        #     h = self.pc2h(h, x, ed_embed, ed_xyz, ed_neighbor)


        # X2H（用 x 更新 h）
        h_in = h
        # 4 separate distance embedding for p-p, p-l, l-p, l-l
        for i in range(self.num_x2h):
            dist_feat = self.distance_expansion(dist)   # 距离变成向量
            dist_feat = outer_product(edge_attr, dist_feat)   # r_feat 距离特征
            h_out = self.x2h_layers[i](h_in, dist_feat, edge_feat, edge_index, e_w=e_w) # node feature
            h_in = h_out
        x2h_out = h_in

        # H2X（用 h 更新 x）
        new_h = h if self.sync_twoup else x2h_out
        for i in range(self.num_h2x):
            dist_feat = self.distance_expansion(dist)
            dist_feat = outer_product(edge_attr, dist_feat)     # 还是r_feat
            delta_x = self.h2x_layers[i](new_h, rel_x, dist_feat, edge_feat, edge_index, e_w=e_w)   # 当前原子节点的坐标调整量
            if not fix_x:
                x = x + delta_x * mask_ligand[:, None]  # only ligand positions will be updated
            rel_x = x[dst] - x[src]
            dist = torch.norm(rel_x, p=2, dim=-1, keepdim=True)

        return x2h_out, x   # 更新后的 node embedding 和坐标，后面整个模型再去预测 μ, σ² 或 logits 等用于 BFN 的分布参数


class UniTransformerO2TwoUpdateGeneral(nn.Module):
    def __init__(self, num_blocks, num_layers, hidden_dim, n_heads=1, knn=32,
                 num_r_gaussian=50, edge_feat_dim=0, num_node_types=8, act_fn='relu', norm=True,
                 cutoff_mode='radius', ew_net_type='r',
                 num_init_x2h=1, num_init_h2x=0, num_x2h=1, num_h2x=1, r_max=10., x2h_out_fc=True, sync_twoup=False, name='unio2net'
                 ):
        super().__init__()
        self.name = name
        # Build the network
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.num_r_gaussian = num_r_gaussian    # 用于 将距离特征 r 映射为高维连续特征表示 的维度数量，常用于图神经网络中对连续距离的编码，特别是在分子建模中
        self.edge_feat_dim = edge_feat_dim
        self.act_fn = act_fn
        self.norm = norm
        self.num_node_types = num_node_types
        # radius graph / knn graph
        self.cutoff_mode = cutoff_mode  # [radius, none]
        self.knn = knn
        self.ew_net_type = ew_net_type  # [r, m, none] 如何根据原子对之间的信息（如距离、边特征等）生成 e_w（边权），并在 attention 中作为权重因子使用

        self.num_x2h = num_x2h
        self.num_h2x = num_h2x
        self.num_init_x2h = num_init_x2h
        self.num_init_h2x = num_init_h2x
        self.r_max = r_max
        self.x2h_out_fc = x2h_out_fc        # 在 X2H attention 输出之后，是否还加一个额外的全连接（Fully Connected）层
        self.sync_twoup = sync_twoup        # X2H 输出的 h，是否立刻用于接下来的 H2X 坐标更新
        self.distance_expansion = GaussianSmearing(0., r_max, num_gaussians=num_r_gaussian) # 把一维的距离（如原子之间的距离）转换成一个高维向量特征。这种变换称为 高斯展开（Gaussian Smearing）
        if self.ew_net_type == 'global':
            self.edge_pred_layer = MLP(num_r_gaussian, 1, hidden_dim)

        self.init_h_emb_layer = self._build_init_h_layer()      # 初始的节点特征 embedding 层（H embedding）
        self.base_block = self._build_share_blocks()            # Transformer 主干结构（通常是多层 AttentionLayerO2TwoUpdateNodeGeneral）

    def __repr__(self):
        """
        norm: 归一化方式
        ew_net_type: 边的加权网络类型？？？
        init_h_emb_layer: 初始节点 embedding 层的结构
        edge_pred_layer: 可选）是否包含边预测头（如果有，则打印结构，否则为 None）
        """
        return f'UniTransformerO2(num_blocks={self.num_blocks}, num_layers={self.num_layers}, n_heads={self.n_heads}, ' \
               f'act_fn={self.act_fn}, norm={self.norm}, cutoff_mode={self.cutoff_mode}, ew_net_type={self.ew_net_type}, ' \
               f'init h emb: {self.init_h_emb_layer.__repr__()} \n' \
               f'base block: {self.base_block.__repr__()} \n' \
               f'edge pred layer: {self.edge_pred_layer.__repr__() if hasattr(self, "edge_pred_layer") else "None"}) '

    def _build_init_h_layer(self):
        """
        初始层：构建初始注意力层
        第一次将节点（atom-level）输入特征映射到模型的“高维语义空间”中
        原始输入（如原子种类、配体/蛋白标识、位置信息等）是低维离散/连续特征，需要先通过一个专门的注意力层做一次映射
        """
        layer = AttentionLayerO2TwoUpdateNodeGeneral(
            self.hidden_dim, self.n_heads, self.num_r_gaussian, self.edge_feat_dim, act_fn=self.act_fn, norm=self.norm,
            num_x2h=self.num_init_x2h, num_h2x=self.num_init_h2x, r_max=self.r_max, num_node_types=self.num_node_types,
            ew_net_type=self.ew_net_type, x2h_out_fc=self.x2h_out_fc, sync_twoup=self.sync_twoup
        )
        return layer

    def _build_share_blocks(self):
        """
        堆叠的共享层
        构建主干的 GNN 块（也就是 Graph Transformer 中的“图层”），
        会被重复使用 num_layers 次，每一层负责对图节点进行更新（包括 feature 和位置的更新）
        这部分就像 transformer 的 encoder 中的多层结构，只不过这里是处理图数据和坐标的
        ：可以理解为这一部分是整个 UniTransformer 模型的 “主干骨架”，反复 message passing 和坐标更新，每层都使用 attention
        """
        # Equivariant layers
        base_block = []
        for l_idx in range(self.num_layers):
            layer = AttentionLayerO2TwoUpdateNodeGeneral(
                self.hidden_dim, self.n_heads, self.num_r_gaussian, self.edge_feat_dim, act_fn=self.act_fn,
                norm=self.norm,
                num_x2h=self.num_x2h, num_h2x=self.num_h2x, r_max=self.r_max, num_node_types=self.num_node_types,
                ew_net_type=self.ew_net_type, x2h_out_fc=self.x2h_out_fc, sync_twoup=self.sync_twoup
            )
            base_block.append(layer)
        return nn.ModuleList(base_block)       # 专门存放“神经网络层”的列表

    # 根据不同的策略构建图中的边连接，x是所有节点的坐标
    def _connect_edge(self, x, mask_ligand, batch):
        if self.cutoff_mode == 'radius':    # 半径邻接图，表示如果两个节点之间的距离小于某个阈值 r（self.r），就认为它们之间有一条边
            edge_index = radius_graph(x, r=self.r, batch=batch, flow='source_to_target')
        elif self.cutoff_mode == 'knn':     # K 近邻图，每个节点连接最近的 k 个邻居（按距离排序）不管距离多远，只要是最近的就连
            edge_index = knn_graph(x, k=self.knn, batch=batch, flow='source_to_target')
        elif self.cutoff_mode == 'hybrid':      # 自定义混合图连接方式
            edge_index = batch_hybrid_edge_connection(
                x, k=self.knn, mask_ligand=mask_ligand, batch=batch, add_p_index=True)
        else:
            raise ValueError(f'Not supported cutoff mode: {self.cutoff_mode}')
        return edge_index   # edge_index 是一个 [2, E(边的总数)] 的张量，表示所有的边连接（src → dst）

    @staticmethod
    def _build_edge_type(edge_index, mask_ligand):
        """
        边关系类型构建，4种
        配体-配体：0
        配体-蛋白：1
        蛋白-配体：2
        蛋白-蛋白：3

        """
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index)
        n_src = mask_ligand[src] == 1   # =1，就是配体，赋值true
        n_dst = mask_ligand[dst] == 1
        edge_type[n_src & n_dst] = 0
        edge_type[n_src & ~n_dst] = 1       # ～按位取反；&，与运算
        edge_type[~n_src & n_dst] = 2
        edge_type[~n_src & ~n_dst] = 3
        edge_type = F.one_hot(edge_type, num_classes=4)
        return edge_type    # edge_feat

    def forward(self, h, x, mask_ligand, batch,
                # ed_embed=None, ed_xyz=None, ed_neighbor=None,  # <<< 新增！
                return_all=False, fix_x=False):  # fix_x: 是否禁止更新 x（冻结原子坐标）

        all_x = [x]
        all_h = [h]

        for b_idx in range(self.num_blocks):
            edge_index = self._connect_edge(x, mask_ligand, batch)  # 构建图结构
            src, dst = edge_index

            # edge type (dim: 4)
            edge_type = self._build_edge_type(edge_index, mask_ligand)  # 构建边类型

            # 得到边权重
            if self.ew_net_type == 'global':
                dist = torch.norm(x[dst] - x[src], p=2, dim=-1, keepdim=True)   # 对每个 [dx, dy, dz] 向量求 L2 范数，也就是欧几里得距离，结果是形状 [E, 1]，每一行是边的距离
                dist_feat = self.distance_expansion(dist)   # 距离变换成向量，形状是 [E, num_r_gaussian]
                logits = self.edge_pred_layer(dist_feat)    # logits 是一个 [E, 1] 的张量，每一行表示一条边的「未归一化权重（logits）」
                e_w = torch.sigmoid(logits)                 # 归一化，sigmoid 把 logits 映射到 0~1，用作边的 attention 权重
            else:
                e_w = None

            for l_idx, layer in enumerate(self.base_block):
                h, x = layer(h, x, edge_type, edge_index, mask_ligand,
                             # ed_embed=ed_embed,
                             # ed_xyz=ed_xyz,
                             # ed_neighbor=ed_neighbor,
                             e_w=e_w, fix_x=fix_x)
            all_x.append(x)
            all_h.append(h)

        outputs = {'x': x, 'h': h}
        if return_all:
            outputs.update({'all_x': all_x, 'all_h': all_h})
        return outputs
