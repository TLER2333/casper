import math
import torch
from torch import nn
import torch.nn.functional as F
from .base_model import BaseModel
from .attention import TransformerEncoder

class GNNLayer(nn.Module):
    def __init__(self, d_model, expansion=2):
        super().__init__()
        self.lin1 = nn.Linear(d_model, d_model * expansion)
        self.lin2 = nn.Linear(d_model * expansion, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, adj_weights):
        h = torch.bmm(adj_weights, x)
        h = self.norm1(x + self.dropout(h))
        h2 = self.lin2(self.act(self.lin1(h)))
        return self.norm2(h + self.dropout(h2))

class CriticalSubgraphExtractor(nn.Module):
    def __init__(self, d_model=256, hidden_dim=256):
        super().__init__()
        self.node_scorer = nn.Sequential(
            nn.Linear(d_model * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.n_heads = 4
        self.head_dim = d_model // self.n_heads
        self.edge_W = nn.Linear(d_model, d_model)
        self.att_src = nn.Parameter(torch.randn(1, self.n_heads, 1, self.head_dim))
        self.att_dst = nn.Parameter(torch.randn(1, self.n_heads, 1, self.head_dim))
        self.leaky_relu = nn.LeakyReLU(0.2)
        
        self.temp = nn.Parameter(torch.ones(1))
        
        self.gnn_layers = nn.ModuleList([GNNLayer(d_model) for _ in range(3)])
        
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, emb_pert, emb_ctrl, adj_matrix):
        B, N, D = emb_pert.shape
        emb_diff = emb_pert - emb_ctrl
        
        node_input = torch.cat([emb_pert, emb_diff], dim=-1)
        node_scores = self.node_scorer(node_input) 
        
        h_trans = self.edge_W(emb_diff).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        score_src = (h_trans * self.att_src).sum(dim=-1, keepdim=True) 
        score_dst = (h_trans * self.att_dst).sum(dim=-1, keepdim=True) 
        
        e_logits = self.leaky_relu(score_src + score_dst.transpose(-1, -2)) / (self.temp.abs() + 1e-6)
        
        mask = (adj_matrix.unsqueeze(1) <= 0)
        e_logits = e_logits.masked_fill(mask, -1e9)
        edge_scores = torch.sigmoid(e_logits).mean(dim=1) * adj_matrix
        
        h_struct = emb_pert + 0.1 * emb_diff 
        
        for gnn in self.gnn_layers:
            row_sum = edge_scores.sum(dim=-1, keepdim=True)
            norm_adj = edge_scores / (row_sum + 1e-8)
            h_struct = gnn(h_struct, norm_adj)
            
        return node_scores, edge_scores, h_struct

class RawCountEncoding(nn.Module):
    def __init__(self, num_buckets=2048, max_value=20000, out_dim=256, dropout=0.1):
        super(RawCountEncoding, self).__init__()
        self.num_buckets = num_buckets
        self.max_value = max_value
        self.out_dim = out_dim
        self.fc = nn.Sequential(
            nn.Embedding(self.num_buckets, self.out_dim), nn.Dropout(dropout)
        )

    @staticmethod
    def _rawcount_bucket(rawcount, num_buckets, max_value):
        max_exact = num_buckets // 2
        is_small = rawcount < max_exact

        val_if_large = (
            max_exact
            + (
                torch.log(rawcount / max_exact)
                / math.log(max_value / max_exact)
                * (num_buckets - max_exact)
            ).long()
        )
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1)
        )

        return torch.where(is_small, rawcount, val_if_large)

    def forward(self, rawcount):
        bucket = self._rawcount_bucket(
            rawcount,
            num_buckets=self.num_buckets,
            max_value=self.max_value,
        )
        x = self.fc(bucket)
        return x

class Casper(nn.Module):
    def __init__(self, args):
        super(Casper, self).__init__()
        self.args = args
        self.d_model = args.d_model
        self.n_cls = args.n_cls
        self.z_dim = args.z_dim

        self.embed_exp = nn.Sequential(nn.Linear(1, self.d_model), nn.Dropout(0.1))
        self.embed_ratio = nn.Sequential(
            nn.Embedding(21, self.d_model), nn.Dropout(0.1)
        )
        self.embed_gene = nn.Sequential(
            nn.Embedding(40002, self.d_model, padding_idx=40000), nn.Dropout(0.1)
        )
        self.embed_rawcount = RawCountEncoding(
            num_buckets=2048, max_value=20000, out_dim=self.d_model, dropout=0.1
        )
        self.embed_in_degree = nn.Embedding(20000, self.d_model, padding_idx=0)
        self.embed_out_degree = nn.Embedding(20000, self.d_model, padding_idx=0)
        self.graph_attn_bias = nn.Embedding(10, 16)

        self.encoder = TransformerEncoder(
            n_layer=6,
            d_model=self.d_model,
            d_key=self.d_model,
            n_head=self.d_model // 16,
            dim_feedforward=self.d_model * 4,
            dropout=0.1
        )

        self.subgraph_extractor = CriticalSubgraphExtractor(d_model=self.d_model, hidden_dim=self.d_model)
        
        self.mu_cls = nn.Linear(512, self.z_dim)
        self.logvar_cls = nn.Linear(512, self.z_dim)

        self.gate_generator = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.z_dim),
            nn.Sigmoid()
        )
        
        self.fc_sub_only = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self.n_cls)
        )
        
        self.fc_final = nn.Sequential(
            nn.Linear(self.z_dim + 256, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, self.n_cls)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _kl_divergence(self, mu, logvar):
        return 0.5 * torch.mean(torch.sum(mu.pow(2) + logvar.exp() - logvar - 1, dim=1))

    def _encode_graph(self, data_subset):
        src = data_subset["ds_norm"].unsqueeze(-1)
        rawcount = data_subset["ds"]
        u_token = data_subset["u_token"]
        in_degree = data_subset["in_degree"]
        out_degree = data_subset["out_degree"]
        mask_attn = data_subset["mask_attn"]
        ratio = data_subset["ratio"]

        cls_id = torch.ones(
            (u_token.shape[0], 1),
            dtype=u_token.dtype,
            device=u_token.device,
        )
        cls_id[:] = 40001
        cls_emb = self.embed_gene(cls_id) + self.embed_ratio(ratio)

        cls_mask1 = torch.ones(
            (mask_attn.shape[0], 1, mask_attn.shape[2]),
            dtype=mask_attn.dtype, device=mask_attn.device
        ) * 6
        mask_attn = torch.cat([cls_mask1, mask_attn], dim=1)

        cls_mask2 = torch.ones(
            (mask_attn.shape[0], mask_attn.shape[1], 1),
            dtype=mask_attn.dtype, device=mask_attn.device
        ) * 6
        mask_attn = torch.cat([cls_mask2, mask_attn], dim=2)
        mask_attn[mask_attn == -1] = 5

        in_degree = in_degree.unsqueeze(0) if in_degree.dim() == 1 else in_degree
        out_degree = out_degree.unsqueeze(0) if out_degree.dim() == 1 else out_degree

        in_degree_embed = self.embed_in_degree(in_degree)
        out_degree_embed = self.embed_out_degree(out_degree)
        
        mask_attn_embed = self.graph_attn_bias(mask_attn).view(
            mask_attn.shape[0], 16, mask_attn.shape[1], mask_attn.shape[2]
        )

        src = (
            self.embed_exp(src)
            + self.embed_gene(u_token)
            + self.embed_rawcount(rawcount)
            + in_degree_embed
            + out_degree_embed
        )

        src = torch.cat([cls_emb, src], dim=1)
        u_token_cat = torch.cat([cls_id, u_token], dim=1)
        mask = u_token_cat == 40000

        src, _ = self.encoder(src, mask, mask_attn_embed)
        return src

    def forward(self, data, epoch):
        mask_dis_long = data["dis_matrix"].long()
        data_pert = {
            "ratio": data["ratio"],
            "u_token": data["u_token"],
            "ds": data["ds"],
            "ds_norm": data["ds_norm"],
            "mask_attn": mask_dis_long,
            "in_degree": data["in_degree"],
            "out_degree": data["out_degree"]
        }

        data_ctrl = {
            "ratio": data["ratio"],
            "u_token": data["u_token"],
            "ds": data["ds_control"],
            "ds_norm": data["ds_control_norm"],
            "mask_attn": mask_dis_long,
            "in_degree": data["in_degree"],
            "out_degree": data["out_degree"]
        }

        src_pert = self._encode_graph(data_pert)
        src_ctrl = self._encode_graph(data_ctrl)
        
        h_cls_cat = torch.cat([src_pert[:, 0], src_ctrl[:, 0]], dim=1)
        emb_pert = src_pert[:, 1:]
        emb_ctrl = src_ctrl[:, 1:]
        
        mu_c = self.mu_cls(h_cls_cat)
        logvar_c = torch.clamp(self.logvar_cls(h_cls_cat), -10, 5)
        z_cls = self.reparameterize(mu_c, logvar_c)
        
        node_scores, _, h_struct = self.subgraph_extractor(emb_pert, emb_ctrl, data["adj_matrix"])
        
        mask_valid = (data["u_token"] != 40000).float().unsqueeze(-1)
        w_crit = node_scores * mask_valid
        h_sub_crit = (h_struct * w_crit).sum(dim=1) / (w_crit.sum(dim=1) + 1e-8)
        
        sub_gate = self.gate_generator(h_sub_crit)
        z_cls_gated = z_cls * sub_gate 
        
        final_feat = torch.cat([z_cls_gated, h_sub_crit], dim=1)
        pred_final = self.fc_final(final_feat)
        
        pred_sub_only = self.fc_sub_only(h_sub_crit)
        
        w_triv = (1 - node_scores) * mask_valid
        h_sub_triv = (emb_pert * w_triv).sum(dim=1) / (w_triv.sum(dim=1) + 1e-8)
        pred_triv = self.fc_sub_only(h_sub_triv)

        out = {
            "pred": pred_final,
            "pred_sub": pred_sub_only,
            "pred_triv": pred_triv,
            "mu_cls": mu_c,
            "logvar_cls": logvar_c,
            "node_scores": node_scores,
        }
        
        out.update(self._compute_loss(out, data, epoch))
        out.update(self._compute_metric(out, data))
        return out

    def _compute_loss(self, out, data, epoch=0):
        target = data["perturb"]
        loss_ce = F.cross_entropy(out["pred"], target, label_smoothing=self.args.label_smoothing)
        loss_sub_ce = F.cross_entropy(out["pred_sub"], target, label_smoothing=self.args.label_smoothing)
        
        kl_cls = self._kl_divergence(out["mu_cls"], out["logvar_cls"])
        
        uni_target = torch.full_like(out["pred_triv"], 1.0 / self.n_cls)
        loss_triv = F.kl_div(F.log_softmax(out["pred_triv"], dim=1), uni_target, reduction='batchmean')
        
        target_k = self.args.target_k
        current_k = out["node_scores"].sum(dim=1)

        diff = F.relu(current_k - target_k)
        loss_sparse_val = (diff / target_k).mean() 
        full_prune_epoch = self.args.full_prune_epoch
        max_sparse_weight = self.args.max_sparse_weight
        
        if epoch < full_prune_epoch:
            progress = epoch / full_prune_epoch
            w_sparse = max_sparse_weight * progress
        else:
            w_sparse = max_sparse_weight

        total_loss = loss_ce + loss_sub_ce  + w_sparse * loss_sparse_val + self.args.beta * kl_cls + self.args.triv_weight * loss_triv

        return {"update_loss": total_loss}
    

    def _compute_metric(self, out, data):
            def accuracy(output, target, topk=(1,)):
                maxk = max(topk)
                batch_size = target.size(0)
                
                _, pred = output.topk(maxk, 1, True, True)
                pred = pred.t()
                
                correct = pred.eq(target.view(1, -1).expand_as(pred))
                
                res = []
                for k in topk:
                    correct_k = correct[:k].reshape(-1).float().sum(0)
                    res.append(correct_k.mul_(1.0 / batch_size))
                return res

            with torch.no_grad():
                metrics = {}
                logits = out["pred"]
                targets = data["perturb"]
                
                if targets.dim() == 1:
                    targets = targets.unsqueeze(1)
                
                B, Num_Labels = targets.shape
                
                topk_list = [1, 5, 10, 50, 100]
                acc_accumulators = {k: 0.0 for k in topk_list} 
                rank_accumulator = 0.0
                
                sorted_indices = torch.argsort(logits, dim=1, descending=True)
                N_vocab = logits.shape[1]

                for i in range(Num_Labels):
                    current_target = targets[:, i]
                    acc_results = accuracy(logits, current_target, topk_list)
                    
                    for k_val, score in zip(topk_list, acc_results):
                        acc_accumulators[k_val] += score.item()
                    
                    target_expand = current_target.view(-1, 1)
                    rank_pos = (sorted_indices == target_expand).nonzero(as_tuple=False)[:, 1]
                    rank_score = 1.0 - rank_pos.float() / (N_vocab - 1)
                    rank_accumulator += rank_score.mean().item()

                for k in topk_list:
                    metrics[f"top{k}_acc"] = torch.tensor(acc_accumulators[k] / Num_Labels)
                
                metrics["rank_acc"] = torch.tensor(rank_accumulator / Num_Labels)

                return metrics