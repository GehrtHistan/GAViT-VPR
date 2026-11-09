import torch
import torch.nn.functional as F
from torch_geometric.nn import knn_graph
from model.R2Former_basic_4 import R2Former

# -----------------------------
# FLOPs 计算函数
def count_mlp_flops(input, output):
    N, in_dim = input.shape
    out_dim = output.shape[1]
    return 2 * N * in_dim * out_dim

def count_transformer_flops(x, num_heads):
    B, N, D = x.shape
    flops_attn = 2 * B * N * N * D      # Q*K^T + softmax*V
    flops_proj = 2 * B * N * D * D      # 输出投影
    flops_ffn = 2 * B * N * D * D * 2   # 两层 FFN
    return flops_attn + flops_proj + flops_ffn

def count_gat_flops(x, edge_index, heads=2):
    N, D = x.shape
    E = edge_index.shape[1]
    return 2 * E * D * heads

# -----------------------------
# 参数
B = 1
num_corr = 3
decoder_embed_dim = 32
N_rerank = 250
input_dim = 131

# 随机输入
x_rerank = torch.rand(B, N_rerank, input_dim).cuda()
y_rerank = torch.rand(B, N_rerank, input_dim).cuda()
x_global = torch.rand(B, 256).cuda()
y_global = torch.rand(B, 256).cuda()

model = R2Former().cuda()
model.eval()

# ==============================
# 1. Correlation FLOPs
x_token = F.normalize(x_rerank[:, :, 3:], p=2, dim=2)
y_token = F.normalize(y_rerank[:, :, 3:], p=2, dim=2)
correlation = torch.matmul(x_token, y_token.permute(0,2,1))
corr_flops = 2 * B * N_rerank * N_rerank * (input_dim - 3)
print(f"Correlation FLOPs: {corr_flops/1e6:.2f} MFLOPs")

# 2. Pair_head FLOPs
pair_matrix = torch.rand(B * N_rerank * num_corr, 7).cuda()
pair_output = model.pair_head(pair_matrix)
pair_flops = count_mlp_flops(pair_matrix, pair_output)
print(f"Pair_head FLOPs: {pair_flops/1e6:.2f} MFLOPs")

# 3. TransformerBlock1 FLOPs
N_select = N_rerank
x_tb1 = pair_output.view(B, N_select * num_corr, decoder_embed_dim)
x_tb1 = torch.cat([model.cls_token.expand(B, -1, -1), x_tb1], dim=1)
tb1_flops = count_transformer_flops(x_tb1, num_heads=4)
print(f"TransformerBlock1 FLOPs: {tb1_flops/1e6:.2f} MFLOPs")

# 4. TransformerBlock2 FLOPs (Basic4 中先于 GAT)
x_tb2 = x_tb1[:, 1:, :]  # 去掉 cls_token
x_tb2 = torch.cat([model.cls_token_2.expand(B,-1,-1), x_tb2], dim=1)
tb2_flops = count_transformer_flops(x_tb2, num_heads=4)
print(f"TransformerBlock2 FLOPs: {tb2_flops/1e6:.2f} MFLOPs")

# 5. GAT FLOPs
x_gat = x_tb2[:, 1:, :].reshape(B * N_select * num_corr, decoder_embed_dim)
batch = torch.arange(B * N_select).repeat_interleave(num_corr).cuda()
edge_index = knn_graph(x_gat, k=3, batch=batch, loop=False)
gat_flops = count_gat_flops(x_gat, edge_index, heads=2) * 2  # 两层 GAT
print(f"GAT layers FLOPs: {gat_flops/1e6:.2f} MFLOPs")

# 6. Decoder_pred FLOPs
x_dec = x_gat.view(B, N_select, num_corr, decoder_embed_dim).mean(dim=2)  # mean pooling
dec_flops = count_mlp_flops(x_dec.reshape(-1, decoder_embed_dim), torch.rand(B*N_select,1).cuda())
print(f"Decoder_pred FLOPs: {dec_flops/1e6:.2f} MFLOPs")

# ==============================
total_flops = (corr_flops + pair_flops + tb1_flops + tb2_flops + gat_flops + dec_flops)/1e9
print(f"Basic4 strict FLOPs (B=1): {total_flops:.6f} GFLOPs")
