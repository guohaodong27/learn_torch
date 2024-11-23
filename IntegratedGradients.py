import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch, DataLoader
from torch_geometric.nn import GATConv
from captum.attr import IntegratedGradients
from torch_geometric.datasets import Planetoid
import numpy as np


# 定义图注意力网络模型
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=2):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# 生成合成数据：10个图，每个图包含5个节点
dataset = Planetoid(root='.', name='Cora')
dataloader = DataLoader(dataset[0], batch_size=3)


# 定义批处理大小
batch_size = 2

# 初始化模型
model = GAT(in_channels=1, hidden_channels=8, out_channels=2, heads=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
# 简单训练过程
model.train()
for epoch in range(100):
    total_loss = 0
    # 按批处理遍历数据
    for i in range(0, len(dataloader), batch_size):
        batch = Batch.from_data_list(data_list[i:i + batch_size])
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        loss = F.nll_loss(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

# 使用 Integrated Gradients 计算节点重要性
model.eval()
ig = IntegratedGradients(model)

# 选择需要解释的节点索引（例如，批处理中的前4个节点）
# 由于每个批次有2个图，每个图5个节点，共10个节点
selected_node_indices = [0, 1, 5, 6]  # 选择第1图的节点0、1和第2图的节点0、1

# 定义基线（这里使用零向量作为基线）
baseline = torch.zeros_like(model.conv1.att_src.grad)

# 计算每个选定节点的特征重要性
for node_idx in selected_node_indices:
    # 获取节点特征并确保需要梯度
    x_node = data_list[node_idx // 5].x[node_idx % 5].unsqueeze(0).unsqueeze(0)  # [1, 1, 1]


    # 定义前向传播函数，仅输出目标节点的预测
    def forward_func(x, edge_index):
        out = model(x, edge_index)
        return out[node_idx].unsqueeze(0)  # 目标节点的输出


    # 计算 Integrated Gradients
    test = data_list[0]
    attribution, delta = ig.attribute(test.x,  n_steps=1, additional_forward_args=test.edge_index, return_convergence_delta=True)

    # 打印节点重要性
    print(f"\n节点 {node_idx} 的特征重要性:")
    print(attribution)

    # 规范化
    importance = attribution.squeeze().detach().numpy()
    importance_normalized = (importance - importance.min()) / (
            importance.max() - importance.min()) if importance.max() != importance.min() else importance
    print("规范化后的重要性:")
    print(importance_normalized)
