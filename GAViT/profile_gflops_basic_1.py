import torch
import torch.nn as nn
import torch.nn.functional as F
from model.R2Former_basic_1 import R2Former
from torch_cluster import knn_graph

# ---------------------------
# FLOPs 计算函数
# ---------------------------
def linear_flops(in_features, out_features, batch=1):
    return batch * in_features * out_features * 2

def matmul_flops(M, N, K, batch=1):
    return batch * M * N * (2*K - 1)

def gat_flops(in_dim, out_dim, num_nodes, heads=2, batch=1):
    linear = linear_flops(in_dim, out_dim, batch=batch*num_nodes)
    attn = num_nodes * out_dim * heads * 2
    return linear + attn

def transformer_block_flops(num_tokens, dim):
    return 4 * num_tokens * dim * dim

def knn_flops(num_nodes, k, feat_dim):
    return num_nodes * k * (2*feat_dim + 1)

# ---------------------------
# 初始化模型和输入
# ---------------------------
B = 1
num_rerank = 250
feat_dim = 131
decoder_embed_dim = 32
decoder_num_heads = 4
num_corr = 3
topk = 3

model = R2Former(
    img_size=(224,224),
    decoder_embed_dim=decoder_embed_dim,
    decoder_depth=2,
    decoder_num_heads=decoder_num_heads,
    decoder_mlp_ratio=4.,
    embed_dim=decoder_embed_dim,
    num_classes=1,
    num_patches=16,
    input_dim=256,
    num_corr=num_corr
).cuda().eval()

x_rerank = torch.randn(B, num_rerank, feat_dim).cuda()
y_rerank = torch.randn(B, num_rerank, feat_dim).cuda()
x_global = torch.randn(B, 256).cuda()
y_global = torch.randn(B, 256).cuda()

# ---------------------------
# FLOPs 计算，严格对应 Basic1 forward
# ---------------------------
rerank_token_dim = feat_dim - 3

# 1. correlation矩阵
flops_corr = matmul_flops(num_rerank, num_rerank, rerank_token_dim, batch=B)

# 2. pair_head
N_select = num_rerank * num_corr * 2
flops_pair = linear_flops(7, decoder_embed_dim, batch=B*N_select*num_corr)

# 3. Transformer blocks (Basic1: Transformer先)
num_tokens_trans = N_select + 1  # +cls_token
flops_trans = sum([transformer_block_flops(num_tokens_trans, decoder_embed_dim) for _ in model.blocks])

# 4. knn_graph (GAT前 token reshape)
num_gat_nodes = B * N_select * num_corr
flops_knn = knn_flops(num_gat_nodes, topk, decoder_embed_dim)

# 5. GAT layers
flops_gat = sum([gat_flops(decoder_embed_dim, decoder_embed_dim, num_gat_nodes, heads=2) for gat in model.gat_layers])

# 6. decoder_pred
flops_pred = linear_flops(decoder_embed_dim, 1, batch=num_gat_nodes)

# 7. 总和
total_flops = flops_corr + flops_pair + flops_trans + flops_knn + flops_gat + flops_pred

print(f"Basic1 strict FLOPs (B={B}): {total_flops/1e9:.6f} GFLOPs")
print(f"Correlation FLOPs: {flops_corr/1e6:.2f} MFLOPs")
print(f"Pair_head FLOPs: {flops_pair/1e6:.2f} MFLOPs")
print(f"Transformer blocks FLOPs: {flops_trans/1e6:.2f} MFLOPs")
print(f"knn_graph FLOPs: {flops_knn/1e6:.2f} MFLOPs")
print(f"GAT layers FLOPs: {flops_gat/1e6:.2f} MFLOPs")
print(f"Decoder_pred FLOPs: {flops_pred/1e6:.2f} MFLOPs")

# 8. 参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"Total params: {total_params/1e3:.1f} K")
