import torch
import torch.nn.functional as F
from torch_geometric.nn import knn_graph
from model.R2Former_basic_2 import R2Former

def count_mlp_flops(input, output):
    N, in_dim = input.shape
    out_dim = output.shape[1]
    return 2 * N * in_dim * out_dim

def count_transformer_flops(x, num_heads):
    B, N, D = x.shape
    flops_attn = 2 * B * N * N * D
    flops_proj = 2 * B * N * D * D
    flops_ffn = 2 * B * N * D * D * 2
    return flops_attn + flops_proj + flops_ffn

def count_gat_flops(x, edge_index, heads=2):
    N, D = x.shape
    E = edge_index.shape[1]
    return 2 * E * D * heads

# -------------------------------
B = 1
num_corr = 3
decoder_embed_dim = 32
N_rerank = 250
input_dim = 131

x_rerank = torch.rand(B, N_rerank, input_dim).cuda()
y_rerank = torch.rand(B, N_rerank, input_dim).cuda()
x_global = torch.rand(B, 256).cuda()
y_global = torch.rand(B, 256).cuda()

model = R2Former().cuda()
model.eval()

# 1. Correlation FLOPs
x_token = F.normalize(x_rerank[:, :, 3:], p=2, dim=2)
y_token = F.normalize(y_rerank[:, :, 3:], p=2, dim=2)
correlation = torch.matmul(x_token, y_token.permute(0, 2, 1))
corr_flops = 2 * B * N_rerank * N_rerank * (input_dim-3)
print(f"Correlation FLOPs: {corr_flops/1e6:.2f} MFLOPs")

# 2. Pair_head
N_select = N_rerank
pair_matrix = torch.rand(B * N_select * num_corr, 7).cuda()
pair_output = model.pair_head(pair_matrix)
pair_flops = count_mlp_flops(pair_matrix, pair_output)
print(f"Pair_head FLOPs: {pair_flops/1e6:.2f} MFLOPs")

# 3. TransformerBlock1
x_tb1 = torch.rand(B, N_select*num_corr+1, decoder_embed_dim).cuda()
tb1_flops = count_transformer_flops(x_tb1, num_heads=4)
print(f"TransformerBlock1 FLOPs: {tb1_flops/1e6:.2f} MFLOPs")

# 4. GAT
x_gat = torch.rand(B * N_select * num_corr, decoder_embed_dim).cuda()
batch = torch.arange(B * N_select).repeat_interleave(num_corr).cuda()
edge_index = knn_graph(x_gat, k=3, batch=batch, loop=False)
gat_flops = count_gat_flops(x_gat, edge_index, heads=2) * 2
print(f"GAT layers FLOPs: {gat_flops/1e6:.2f} MFLOPs")

# 5. TransformerBlock2
x_tb2 = torch.rand(B, N_select+1, decoder_embed_dim).cuda()
tb2_flops = count_transformer_flops(x_tb2, num_heads=4)
print(f"TransformerBlock2 FLOPs: {tb2_flops/1e6:.2f} MFLOPs")

# 6. Decoder_pred
x_dec = torch.rand(B, N_select, decoder_embed_dim).cuda()
dec_flops = count_mlp_flops(x_dec.reshape(-1, decoder_embed_dim), torch.rand(B*N_select,1).cuda())
print(f"Decoder_pred FLOPs: {dec_flops/1e6:.2f} MFLOPs")

# Total
total_flops = (corr_flops + pair_flops + tb1_flops + gat_flops + tb2_flops + dec_flops)/1e9
print(f"Basic2 strict FLOPs (B=1): {total_flops:.6f} GFLOPs")
