from utils import endc, bold, red
import einops
import math
from transformers import GPT2TokenizerFast
from dataclasses import dataclass
import wandb
import torch as t
import torch.nn as nn

device = t.device("cuda" if t.cuda.is_available() else "cpu")

@dataclass
class modelConfig:
    d_model: int = 512
    d_vocab: int = 50304 # 50304
    init_range: float = 0.02
    n_ctx: int = 512
    d_head: int = 64
    d_mlp: int = 2048
    n_heads: int = 8
    n_layers: int = 8
    attention_type: str = "MHA"
    head_group_size: int =2

@dataclass
class trainingConfig:
    train_tokens: int = 300_000
    test_tokens: int = 10_000
    batch_size: int = 16
    epochs: int = 1
    lr: float = 5e-4
    weight_decay: float = 2e-3
    wandb_project: str = "gqa_project"
    wandb_name: str = None
    save_name: str = None

# This is standard, mostly unoptimized multihead causal self-attention
class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: modelConfig):
        super().__init__()
        self.cfg = cfg

        # These matrices are how we trasform the residual stream into keys, queries, and values
        self.W_Q = nn.Parameter(t.empty(cfg.d_model, cfg.n_heads, cfg.d_head))
        self.W_K = nn.Parameter(t.empty(cfg.d_model, cfg.n_heads, cfg.d_head))
        self.W_V = nn.Parameter(t.empty(cfg.d_model, cfg.n_heads, cfg.d_head))
        nn.init.xavier_uniform_(self.W_Q) # initialize the weights of the input projections
        nn.init.xavier_uniform_(self.W_K) 
        nn.init.xavier_uniform_(self.W_V) 

        # output projection from final averaged value vectors (length d_head) back to normal residual vectors (length d_model)
        self.W_out = nn.Parameter(t.empty(cfg.d_head, cfg.n_heads, cfg.d_model))
        self.b_out = nn.Parameter(t.zeros(cfg.d_model))
        nn.init.xavier_uniform_(self.W_out) # initialize the output projection 

    def forward(self, resid: t.Tensor):
        # get some input shapes.
        batch, seq, _ = resid.shape

        # do the key, query, and value projections for every head simultaneously.
        #qkv = einops.einsum(resid, self.W_in, "batch seq d_model, d_model kqv n_heads d_head -> batch seq kqv n_heads d_head") + self.b_in
        q = einops.einsum(resid, self.W_Q, "batch seq d_model, d_model n_heads d_head -> batch seq n_heads d_head")
        k = einops.einsum(resid, self.W_K, "batch seq d_model, d_model n_heads d_head -> batch seq n_heads d_head")
        v = einops.einsum(resid, self.W_V, "batch seq d_model, d_model n_heads d_head -> batch seq n_heads d_head")

        # This does the dot product between all possible pairs of query and value vectors.
        # We use different dimension names 'qseq' and 'kseq' to refer to the query sequence dimension and the key sequence dimension, but here keys and queries come from the same sequence.
        # This is what makes it 'self-attention'.
        # We divide by the sqrt of the d_head dimension for numerical stability reasons, as recommended in 'Attention Is All You Need'
        scores = einops.einsum(q, k, "batch qseq n_heads d_head, batch kseq n_heads d_head -> batch n_heads qseq kseq") / math.sqrt(self.cfg.d_head) 

        # This sets values above the main diagonal to a large negative number. This means the model can't move information backwards (from a later token to an earlier one).  This is what makes it 'causal'.
        causal_mask = t.triu(t.ones(seq, seq, device=device), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask.to(scores.device), -1e6)

        # softmax along rows turns these into the attention probabilities. large negative values go to approximately 0.
        probs = scores.softmax(dim=-1)
        
        # This produces a weighted sum of value vectors, where the weight of the value vector from sequence position i in the sum for sequence position j is just probs[j, i].
        z = einops.einsum(probs, v, "batch n_heads qseq kseq, batch kseq n_heads d_head -> batch qseq n_heads d_head")

        # We then project each head back to the residual space from whence it came, and add all the projected outputs together.
        out = einops.einsum(z, self.W_out, "batch seq n_heads d_head, d_head n_heads d_model -> batch seq d_model") + self.b_out
        return out

# This is MQA, where ALL heads in the layer share the same values and keys. The only that that differs between heads is the query.
class MultiQueryAttention(nn.Module):
    def __init__(self, cfg: modelConfig):
        super().__init__()
        self.cfg = cfg
        
        self.W_Q = nn.Parameter(t.empty(cfg.d_model, cfg.n_heads, cfg.d_head)) # We still have a query for each head, 
        self.W_K = nn.Parameter(t.empty(cfg.d_model, cfg.d_head)) # but only one key,
        self.W_V = nn.Parameter(t.empty(cfg.d_model, cfg.d_head)) # and one value.
        nn.init.xavier_uniform_(self.W_Q) # initialize the weights of the input projections
        nn.init.xavier_uniform_(self.W_K) 
        nn.init.xavier_uniform_(self.W_V) 

        nn.init.xavier_uniform_(self.W_in) # initialize the input projection 

        # output projection from final averaged value vectors (length d_head) back to normal residual vectors (length d_model)
        self.W_out = nn.Parameter(t.empty(cfg.d_head, cfg.n_heads, cfg.d_model))
        self.b_out = nn.Parameter(t.zeros(cfg.d_model))
        nn.init.xavier_uniform_(self.W_out) # initialize the output projection 

    def forward(self, resid: t.Tensor):
        batch, seq, _ = resid.shape

        q = einops.einsum(resid, self.W_Q, "batch seq d_model, d_model n_heads d_head -> batch seq n_heads d_head")
        
        # The rest is identical, except that the keys and queries no longer have a n_heads dimension
        k = einops.einsum(resid, self.W_K, "batch seq d_model, d_model d_head -> batch seq d_head")
        v = einops.einsum(resid, self.W_V, "batch seq d_model, d_model d_head -> batch seq d_head")
        scores = einops.einsum(q, k, "n_heads batch qseq d_head, batch kseq d_head -> batch n_heads qseq kseq") / math.sqrt(self.cfg.d_head) 
        causal_mask = t.triu(t.ones(seq, seq, device=device), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask, -1e6)
        probs = scores.softmax(dim=-1)
        z = einops.einsum(probs, v, "batch n_heads qseq kseq, batch kseq d_head -> batch qseq n_heads d_head")
        out = einops.einsum(z, self.W_out, "batch seq n_heads d_head, d_head n_heads d_model -> batch seq d_model") + self.b_out
        return out

# This is GQA, which lets us select the best tradeoff between MHA, where each head has its own keys and values, and MQA, where all heads share 1 set of keys and 1 seq of values.
# In GQA we can select a group size. Heads in different groups have different keys and different values, and heads in the same group have the same keys and the same values.
class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg: modelConfig):
        super().__init__()
        self.cfg = cfg
        self.group_size = self.cfg.head_group_size 
        
        # group size is the number of heads per group, so we will only have n_heads/group_size sets of keys. Same for values
        self.n_groups = self.cfg.n_heads//self.group_size

        self.W_Q = nn.Parameter(t.empty(cfg.d_model, cfg.n_heads, cfg.d_head))
        self.W_K = nn.Parameter(t.empty(cfg.d_model, self.n_groups, cfg.d_head))
        self.W_V = nn.Parameter(t.empty(cfg.d_model, self.n_groups, cfg.d_head))
        nn.init.xavier_uniform_(self.W_Q) # initialize the weights of the input projections
        nn.init.xavier_uniform_(self.W_K) 
        nn.init.xavier_uniform_(self.W_V) 

        self.W_out = nn.Parameter(t.empty(cfg.d_head, cfg.n_heads, cfg.d_model))
        self.b_out = nn.Parameter(t.zeros(cfg.d_model))
        nn.init.xavier_uniform_(self.W_out) # initialize the output projection 

    def forward(self, resid: t.Tensor):
        batch, seq, _ = resid.shape

        q = einops.einsum(resid, self.W_Q, "batch seq d_model, d_model n_heads d_head -> batch seq n_heads d_head")
        # We split the head dimension for the queries into two dimensions: one for the groups and one for the heads within the group
        q = q.reshape(batch, seq, self.n_groups, self.group_size, self.cfg.d_head) 
        # group dimension replaces head dimension for keys and values
        k = einops.einsum(resid, self.W_K, "batch seq d_model, d_model group d_head -> batch seq group d_head")
        v = einops.einsum(resid, self.W_V, "batch seq d_model, d_model group d_head -> batch seq group d_head")

        scores = einops.einsum(q, k, "batch qseq group group_head d_head, batch kseq group d_head -> batch group group_head qseq kseq") / math.sqrt(self.cfg.d_head) 

        causal_mask = t.triu(t.ones(seq, seq, device=device), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask.to(scores.device), -1e6)
        probs = scores.softmax(dim=-1)
        z = einops.einsum(probs, v, "batch group group_head qseq kseq, batch kseq group d_head -> batch qseq group group_head d_head")

        # concatenate the head outputs of heads in different groups, creating an output of the same shape as the queries
        z = z.reshape(batch, seq, self.cfg.n_heads, self.cfg.d_head)

        # normal projection to residual space.
        out = einops.einsum(z, self.W_out, "batch seq n_heads d_head, d_head n_heads d_model -> batch seq d_model") + self.b_out
        return out

class tblock(nn.Module):
    def __init__(self, cfg: modelConfig, targs: trainingConfig):
        super().__init__()
        self.cfg, self.targs = cfg, targs
        self.ln1 = nn.LayerNorm((cfg.d_model))
        if cfg.attention_type == "MHA":
            self.attn = MultiHeadAttention(cfg)
        elif self.cfg.attention_type == "MQA":
            self.attn = MultiQueryAttention(cfg)
        elif self.cfg.attention_type == "GQA":
            self.attn = GroupedQueryAttention(cfg)
        else:
            assert False, f"{bold+red}attention_type was '{cfg.attention_type}'. Expected one of ['MHA', 'MQA', 'GQA']{endc}"
        self.ln2 = nn.LayerNorm((cfg.d_model))
        self.mlp1 = nn.Linear(cfg.d_model, cfg.d_mlp)
        self.act = nn.GELU()
        self.mlp2 = nn.Linear(cfg.d_mlp, cfg.d_model)

    def forward(self, x: t.Tensor):
        normed = self.ln1(x)
        attn_out = self.attn.forward(normed)
        
        post_attn = x + attn_out
        mlp_out = self.ln2(post_attn)
        mlp_out = self.act(self.mlp1(mlp_out))
        mlp_out = self.mlp2(mlp_out)
        post_mlp = post_attn + mlp_out
        return post_mlp

class gpt2(nn.Module):
    def __init__(self, cfg, targs):
        super().__init__()
        self.cfg, self.targs = cfg, targs

        self.E = nn.Embedding(cfg.d_vocab, cfg.d_model)
        self.PE = nn.Embedding(cfg.n_ctx, cfg.d_model)
        self.blocks = nn.Sequential(*[tblock(cfg, targs) for i in range(cfg.n_layers)])
        self.UE = nn.Linear(cfg.d_model, cfg.d_vocab, bias=False)
        self.ln_final = nn.LayerNorm((cfg.d_model))

        self.to(device)
        self.device = device
        
        self.tk = GPT2TokenizerFast.from_pretrained('gpt2')
        self.tk.add_special_tokens({'pad_token': '<PAD>'})
        self.opt = t.optim.AdamW(self.parameters(), targs.lr, betas=(0.9, 0.95), weight_decay=targs.weight_decay, fused=True)
        self.scheduler = t.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=1e4, eta_min=0)

    def forward(self, x: t.Tensor):
        if isinstance(x, str): x = self.tokenize(x)['input_ids'].to(device)
        resid = self.E(x) + self.PE(t.arange(0, x.shape[-1], device=device, requires_grad=False).unsqueeze(0))
        resid = self.blocks(resid)
        logits = self.UE(self.ln_final(resid))
        return logits

    def loss(self, logits: t.Tensor, labels: t.tensor):
        logprobs = logits.log_softmax(dim=-1)
        correct_logprobs = logprobs[:, :-1].gather(dim=-1, index=labels[:, 1:].unsqueeze(-1)).squeeze(-1)
        return -(correct_logprobs.mean())

    def trainstep(self, loss: t.Tensor):
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.scheduler.step()

    def tokenize(self, prompt): return self.tk(prompt, return_tensors='pt')
    def decode(self, tokens): return self.tk.batch_decode(tokens)
    def accuracy(self, logits: t.Tensor, tokens: t.Tensor):
        preds = logits.squeeze().argmax(dim=-1)
        return (preds[...,:-1]==tokens.squeeze()[...,1:]).float().mean()
    def yap(self, _prompt, ntok=30, show=False):
        out = _prompt
        prompt = self.tokenize(_prompt)['input_ids'].to(device).squeeze()
        for i in range(ntok):
            logits = self.forward(prompt).squeeze()
            nexttok = self.sample(logits)
            prompt = t.cat([prompt, t.tensor([nexttok], device=device)], dim=-1)
            out += self.tk.decode(nexttok)
            if show:
                print(out)
                print()
        return out
    def sample(self, logits: t.Tensor, k=5, temp=0):
        vals, indices = logits.squeeze()[-1].topk(k)
        if temp != 0: vals /= temp
        idx = t.distributions.categorical.Categorical(logits=vals).sample().item()
        return indices[idx].item()
    def log_completion(self, completion):
        table = wandb.Table(data=[[completion]], columns=['completion'])
        wandb.log({"completion": table})
    def load(self, path):
        self.load_state_dict(t.load(path).state_dict())

