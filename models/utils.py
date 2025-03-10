import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from transformers import T5Tokenizer, T5EncoderModel
from diffusers.models.embeddings import PixArtAlphaTextProjection

from timm.models.vision_transformer import Block, Attention, LayerScale
from timm.layers import Mlp, DropPath


class T5Wrapper(nn.Module):
    def __init__(self, mt5_model_name, cache_dir):
        super(T5Wrapper, self).__init__()
        self.model_name = mt5_model_name
        print(f'Loading T5 model from {self.model_name}...')
        self.text_enc = T5EncoderModel.from_pretrained(self.model_name, cache_dir=cache_dir).eval()

    def forward(self, input_ids, attention_mask):
        # Ensure input_ids and attention_mask are torch tensors and on the correct device
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids)
        if not isinstance(attention_mask, torch.Tensor):
            attention_mask = torch.tensor(attention_mask)
        
        # Detach the encoder output so that gradients are not computed for frozen model
        text_encoder_embs = self.text_enc(input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state']
        return text_encoder_embs.detach()


class T5_Embedding(nn.Module):
    def __init__(self, mt5_model_name, cache_dir, max_length=128):
        super(T5_Embedding, self).__init__()
        self.max_length = max_length
        # Load pre-trained MT5 model
        self.tokenizer = T5Tokenizer.from_pretrained(mt5_model_name, cache_dir=cache_dir)
        self.text_encoder = T5Wrapper(mt5_model_name, cache_dir)

        # Freeze all the parameters of the T5 model
        for param in self.text_encoder.parameters():
            param.requires_grad = False
    
    def forward(self, texts):
        
        text_inputs = self.tokenizer(
            texts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(texts, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            # print(
            #     "The following part of your input was truncated because CLIP can only handle sequences up to"
            #     f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            # )

        text_input_ids = text_input_ids.cuda()
        prompt_attention_mask = text_inputs.attention_mask.cuda()

        text_embeddings = self.text_encoder( # B, max_length, vacub_length
            text_input_ids,
            attention_mask=prompt_attention_mask,
        ) 

        return text_embeddings
    

class Text_Embedding(nn.Module):
    def __init__(self, caption_channels, encoder_embed_dim, text_num_heads, text_depth, mlp_ratio=4.0, 
                 norm_layer=nn.LayerNorm, proj_dropout=0.0, attn_dropout=0.0, grad_checkpointing=False):
        super(Text_Embedding, self).__init__()
        
        self.text_proj = PixArtAlphaTextProjection(in_features=caption_channels, hidden_size=encoder_embed_dim)

        self.grad_checkpointing = grad_checkpointing
        self.text_aligner = nn.ModuleList([
            Block(encoder_embed_dim, text_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(text_depth)])
        self.text_aligner_norm = norm_layer(encoder_embed_dim)

    def forward(self, text_embeddings):

        text_embeddings = self.text_proj(text_embeddings)

        # apply Transformer blocks
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.text_aligner:
                text_embeddings = checkpoint(block, text_embeddings)
        else:
            for block in self.text_aligner:
                text_embeddings = block(text_embeddings)
        
        text_embeddings = self.text_aligner_norm(text_embeddings)
        
        return text_embeddings


class Cross_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.query_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim*2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, key_value):
        B, N, C = query.shape
        q = self.query_linear(query).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(key_value).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention_Block(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm, 
            is_cross=False
    ):
        super().__init__()
        self.is_cross = is_cross
        if self.is_cross:
            self.normq = norm_layer(dim)
            self.normkv = norm_layer(dim)
            self.attn = Cross_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        else:
            self.norm1 = norm_layer(dim)
            self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=proj_drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, q, k_v=None):
        if self.is_cross:
            q = self.normq(q)
            k_v = self.normkv(k_v)
            x = q + self.drop_path1(self.ls1(self.attn(q, k_v)))
        else:
            x = self.norm1(q)
            x = x + self.drop_path1(self.ls1(self.attn(x)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x