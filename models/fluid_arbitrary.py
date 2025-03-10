from functools import partial

import numpy as np
from tqdm import tqdm
import scipy.stats as stats
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from timm.models.vision_transformer import Block

import os, sys
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from models.diffloss import DiffLoss
from models.utils import Text_Embedding, Attention_Block


def mask_by_order(mask_len, order, bsz, seq_len):
    masking = torch.zeros(bsz, seq_len).cuda()
    masking = torch.scatter(masking, dim=-1, index=order[:, :mask_len.long()], src=torch.ones(bsz, seq_len).cuda()).bool()
    return masking


class FLUID(nn.Module):
    """ 
        Finetune Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=256, vae_stride=16, patch_size=1, text_depth=6, 
                 encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
                 decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
                 cross_embed_dim=1024, cross_depth=16, cross_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 caption_channels=4096, max_length=512,
                 interpolate_offset=0.1,
                 vae_embed_dim=16,
                 mask_ratio_min=0.7,
                 text_drop_prob=0.1,
                 attn_dropout=0.1,
                 proj_dropout=0.1,
                 diffloss_d=3,
                 diffloss_w=1024,
                 num_sampling_steps='100',
                 diffusion_batch_mul=4,
                 grad_checkpointing=False,
                 ):
        super().__init__()

        # --------------------------------------------------------------------------
        # VAE and patchify specifics
        self.vae_embed_dim = vae_embed_dim

        self.img_size = img_size
        self.vae_stride = vae_stride
        self.patch_size = patch_size
        self.seq_h = self.seq_w = img_size // vae_stride // patch_size
        self.seq_len = 256 # self.seq_h * self.seq_w # start from 256px ckpt
        self.token_embed_dim = vae_embed_dim * patch_size**2
        self.grad_checkpointing = grad_checkpointing
        self.caption_channels = caption_channels
        self.max_length = max_length
        self.cross_embed_dim = cross_embed_dim
        self.interpolate_offset = interpolate_offset

        # --------------------------------------------------------------------------
        # Fluid cross attn specifics (Text Embedding)
        self.last_embed = nn.Linear(decoder_embed_dim, cross_embed_dim, bias=True)
        self.last_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len, cross_embed_dim))

        self.text_drop_prob = text_drop_prob
        # Fake text embedding for CFG's unconditional generation
        self.fake_latent = nn.Parameter(torch.zeros(1, 1, cross_embed_dim))

        self.text_emb = Text_Embedding(caption_channels, encoder_embed_dim, text_num_heads=encoder_num_heads, text_depth=text_depth, 
                                       mlp_ratio=mlp_ratio, norm_layer=norm_layer, proj_dropout=proj_dropout, attn_dropout=attn_dropout, 
                                       grad_checkpointing=grad_checkpointing)
        
        # Cross-attention layer (projecting text embedding into the vision token space)
        self.cross_attn_blocks =  nn.ModuleList([
            Attention_Block(dim=cross_embed_dim, num_heads=cross_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer, 
                                proj_drop=proj_dropout, attn_drop=attn_dropout, is_cross=True) for _ in range(cross_depth)])
        self.cross_norm = norm_layer(cross_embed_dim)

        # --------------------------------------------------------------------------
        # Fluid variant masking ratio, a left-half truncated Gaussian centered at 100% masking ratio with std 0.25
        self.mask_ratio_generator = stats.truncnorm((mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25)

        # --------------------------------------------------------------------------
        # Fluid encoder specifics
        self.z_proj = nn.Linear(self.token_embed_dim, encoder_embed_dim, bias=True)
        self.z_proj_ln = nn.LayerNorm(encoder_embed_dim, eps=1e-6)
        self.encoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len, encoder_embed_dim))

        self.encoder_blocks = nn.ModuleList([
            Block(encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(encoder_depth)])
        self.encoder_norm = norm_layer(encoder_embed_dim)

        # --------------------------------------------------------------------------
        # Fluid decoder specifics
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.initialize_weights()

        # --------------------------------------------------------------------------
        # Diffusion Loss
        self.diffloss = DiffLoss(
            target_channels=self.token_embed_dim,
            z_channels=decoder_embed_dim,
            width=diffloss_w,
            depth=diffloss_d,
            num_sampling_steps=num_sampling_steps,
            grad_checkpointing=grad_checkpointing
        )
        self.diffusion_batch_mul = diffusion_batch_mul

    def initialize_weights(self):
        # parameters
        torch.nn.init.normal_(self.fake_latent, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.encoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.last_pos_embed_learned, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
    
    def patchify(self, x):
        bsz, c, h, w = x.shape
        p = self.patch_size
        h_, w_ = h // p, w // p

        x = x.reshape(bsz, c, h_, p, w_, p)
        x = torch.einsum('nchpwq->nhwcpq', x)
        x = x.reshape(bsz, h_ * w_, c * p ** 2)
        return x  # [n, l, d]

    def unpatchify(self, x):
        bsz = x.shape[0]
        p = self.patch_size
        c = self.vae_embed_dim
        h_, w_ = self.seq_h, self.seq_w

        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum('nhwcpq->nchpwq', x)
        x = x.reshape(bsz, c, h_ * p, w_ * p)
        return x  # [n, c, h, w]

    def sample_orders(self, bsz):
        # generate a batch of random generation orders
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(self.seq_len)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).cuda().long()
        return orders

    def random_masking(self, x, orders):
        # generate token mask
        bsz, seq_len, embed_dim = x.shape
        mask_rate = self.mask_ratio_generator.rvs(1)[0]
        num_masked_tokens = int(np.ceil(seq_len * mask_rate))
        mask = torch.zeros(bsz, seq_len, device=x.device)
        mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens],
                             src=torch.ones(bsz, seq_len, device=x.device))
        return mask

    def interpolate_pos_encoding(self, x, pos_embed):
        previous_dtype = x.dtype
        npatch = x.shape[1]
        N = pos_embed.shape[1]
        if npatch == N and self.seq_w == self.seq_h:
            return pos_embed
        pos_embed = pos_embed.float()
        dim = x.shape[-1]
        w0 = self.seq_w
        h0 = self.seq_h
        M = int(math.sqrt(N))  # Recover the number of patches in each dimension
        assert N == M * M
        kwargs = {}
        if self.interpolate_offset:
            # Historical kludge: add a small number to avoid floating point error in the interpolation, see https://github.com/facebookresearch/dino/issues/8
            # Note: still needed for backward-compatibility, the underlying operators are using both output size and scale factors
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sx, sy)
        else:
            # Simply specify an output size instead of a scale factor
            kwargs["size"] = (w0, h0)
        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            **kwargs,
        )
        assert (w0, h0) == pos_embed.shape[-2:]
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return pos_embed.to(previous_dtype)
    
    def forward_fluid_encoder(self, x, mask):
        x = self.z_proj(x)

        bsz, _, embed_dim = x.shape

        # encoder position embedding
        # x = x + self.encoder_pos_embed_learned
        encoder_pos_embed = self.interpolate_pos_encoding(x, self.encoder_pos_embed_learned)
        x = x + encoder_pos_embed
        x = self.z_proj_ln(x)

        # dropping
        # x = x[(1 - mask).nonzero(as_tuple=True)].reshape(bsz, -1, embed_dim)
        x = x[torch.logical_not(mask).nonzero(as_tuple=True)].reshape(bsz, -1, embed_dim)

        # apply Transformer blocks
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.encoder_blocks:
                x = checkpoint(block, x)
        else:
            for block in self.encoder_blocks:
                x = block(x)
        x = self.encoder_norm(x)

        return x
    
    def forward_fluid_decoder(self, x, mask):

        x = self.decoder_embed(x)

        # pad mask tokens
        mask_tokens = self.mask_token.repeat(mask.shape[0], mask.shape[1], 1).to(x.dtype)
        x_after_pad = mask_tokens.clone()
        # x_after_pad[(1 - mask).nonzero(as_tuple=True)] = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
        x_after_pad[torch.logical_not(mask).nonzero(as_tuple=True)] = x.reshape(x.shape[0] * x.shape[1], x.shape[2])

        # decoder position embedding
        # x = x_after_pad + self.decoder_pos_embed_learned
        decoder_pos_embed = self.interpolate_pos_encoding(x_after_pad, self.decoder_pos_embed_learned)
        x = x_after_pad + decoder_pos_embed

        # apply Transformer blocks
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.decoder_blocks:
                x = checkpoint(block, x)
        else:
            for block in self.decoder_blocks:
                x = block(x)
        x = self.decoder_norm(x)
    
        return x
    
    def forward_fluid_text_decoder(self, x, text_embeddings):

        x = self.last_embed(x)
        # x = x + self.last_pos_embed_learned
        text_pos_embed = self.interpolate_pos_encoding(x, self.last_pos_embed_learned)
        x = x + text_pos_embed
        bsz, _, _ = x.shape
        # random drop text embedding during training
        if self.training:
            drop_latent_mask = torch.rand(bsz) < self.text_drop_prob
            drop_latent_mask = drop_latent_mask.unsqueeze(-1)
            drop_latent_mask = drop_latent_mask.unsqueeze(-1).cuda().to(x.dtype)
            text_embeddings = drop_latent_mask * self.fake_latent + (1 - drop_latent_mask) * text_embeddings
        
        # apply Transformer blocks
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.cross_attn_blocks:
                x = checkpoint(block, x, text_embeddings)
        else:
            for block in self.cross_attn_blocks:
                x = block(x, text_embeddings)

        x = self.cross_norm(x)

        return x

    def forward_loss(self, z, target, mask):
        bsz, seq_len, _ = target.shape
        target = target.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        z = z.reshape(bsz*seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        mask = mask.reshape(bsz*seq_len).repeat(self.diffusion_batch_mul)
        loss = self.diffloss(z=z, target=target, mask=mask)
        return loss
    
    def forward(self, imgs, texts, height=512, width=512):
        assert height % (self.vae_stride * self.patch_size) == 0 and width % (self.vae_stride * self.patch_size) == 0
        self.seq_h, self.seq_w = height // self.vae_stride // self.patch_size, width // self.vae_stride // self.patch_size
        self.seq_len = self.seq_h * self.seq_w
        # patchify and mask (drop) tokens
        x = self.patchify(imgs)
        gt_latents = x.clone().detach()
        orders = self.sample_orders(bsz=x.size(0))

        mask = self.random_masking(x, orders)

        # mae encoder
        x = self.forward_fluid_encoder(x, mask)

        # mae decoder
        x = self.forward_fluid_decoder(x, mask)

        # Tokenize and encode the input text
        text_embeddings = self.text_emb(texts)
        z = self.forward_fluid_text_decoder(x, text_embeddings)

        # diffloss
        loss = self.forward_loss(z=z, target=gt_latents, mask=mask)

        return loss
    
    def sample_tokens(self, bsz, num_iter=64, cfg=1.0, cfg_schedule="linear", texts=None, temperature=1.0, height=512, width=512, progress=False):

        assert height % (self.vae_stride * self.patch_size) == 0 and width % (self.vae_stride * self.patch_size) == 0
        self.seq_h, self.seq_w = height // self.vae_stride // self.patch_size, width // self.vae_stride // self.patch_size
        self.seq_len = self.seq_h * self.seq_w

        # init and sample generation orders
        mask = torch.ones(bsz, self.seq_len).cuda()
        tokens = torch.zeros(bsz, self.seq_len, self.token_embed_dim).cuda()
        orders = self.sample_orders(bsz)

        indices = list(range(num_iter))
        if progress:
            indices = tqdm(indices)
        if texts is not None:
            text_embeddings_temp = self.text_emb(texts)
        # generate latents
        for step in indices:
            cur_tokens = tokens.clone()

            # text_embedding and CFG
            if texts is None:
                text_embeddings = self.fake_latent.expand(bsz, self.max_length, self.cross_embed_dim)
            else:
                text_embeddings = text_embeddings_temp
            if not cfg == 1.0:
                tokens = torch.cat([tokens, tokens], dim=0)
                text_embeddings = torch.cat([text_embeddings, self.fake_latent.expand(bsz, self.max_length, self.cross_embed_dim)], dim=0)
                mask = torch.cat([mask, mask], dim=0)

            # mae encoder
            x = self.forward_fluid_encoder(tokens, mask)

            # mae decoder
            x = self.forward_fluid_decoder(x, mask)

            z = self.forward_fluid_text_decoder(x, text_embeddings)

            # mask ratio for the next round, following MaskGIT and MAGE.
            mask_ratio = np.cos(math.pi / 2. * (step + 1) / num_iter)
            mask_len = torch.Tensor([np.floor(self.seq_len * mask_ratio)]).cuda()

            # masks out at least one for the next iteration
            mask_len = torch.maximum(torch.Tensor([1]).cuda(),
                                     torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len))

            # get masking for next iteration and locations to be predicted in this iteration
            mask_next = mask_by_order(mask_len[0], orders, bsz, self.seq_len)

            if step >= num_iter - 1:
                mask_to_pred = mask[:bsz].bool()
            else:
                mask_to_pred = torch.logical_xor(mask[:bsz].bool(), mask_next.bool())
            mask = mask_next
            if not cfg == 1.0:
                mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)

            # sample token latents for this step
            z = z[mask_to_pred.nonzero(as_tuple=True)]
            # cfg schedule follow Muse
            if cfg_schedule == "linear":
                cfg_iter = 1 + (cfg - 1) * (self.seq_len - mask_len[0]) / self.seq_len
            elif cfg_schedule == "constant":
                cfg_iter = cfg
            else:
                raise NotImplementedError
            sampled_token_latent = self.diffloss.sample(z, temperature, cfg_iter)
            if not cfg == 1.0:
                sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)  # Remove null class samples
                mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)

            cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent
            tokens = cur_tokens.clone()

        # unpatchify
        tokens = self.unpatchify(tokens)
        return tokens


def fluid_base(**kwargs):
    model = FLUID(
        encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
        decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12,
        cross_embed_dim=768, cross_depth=12, cross_num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def fluid_large(**kwargs):
    model = FLUID(
        encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
        decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
        cross_embed_dim=1024, cross_depth=16, cross_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def fluid_huge(**kwargs):
    model = FLUID(
        encoder_embed_dim=1280, encoder_depth=20, encoder_num_heads=16,
        decoder_embed_dim=1280, decoder_depth=20, decoder_num_heads=16,
        cross_embed_dim=1280, cross_depth=20, cross_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

if __name__ == "__main__":
    mt5_cache_dir = '/data/xianfeng/code/model/google/flan-t5-xxl'
    mt5_model_name = mt5_cache_dir
    # img_size = 256
    patch_size = 2
    vae_embed_dim = 16
    vae_stride = 8
    vae_temperal_stride = 4
    diffloss_d = 8
    diffloss_w = 1280

    model = fluid_large(vae_embed_dim=vae_embed_dim, vae_stride=vae_stride, patch_size=patch_size, 
                        diffloss_d=diffloss_d, diffloss_w=diffloss_w)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {}M".format(n_params / 1e6))
    model = model.cuda()

    # generate pseudo image data (batch_size, channels, height, width)
    fake_image = torch.randn(4, 16, 32, 32).cuda()  # batch_size=4, 3 channels, 256x256 resolution

    # generate pseudo text data
    fake_texts = ["This is a test sentence.", "Another example sentence.", "Text embedding test.", "Vision and text fusion."]

    # T5 Embedding
    import os, sys
    sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
    sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))

    # from models.vae import AutoencoderKL
    # define the vae and mar model
    from diffusers.models import AutoencoderKL
    vae_path = '/data/xianfeng/code/model/stabilityai/stable-diffusion-3.5-large'
    vae = AutoencoderKL.from_pretrained(os.path.join(vae_path, "vae")).cuda().eval()
    for param in vae.parameters():
        param.requires_grad = False

    # posterior = vae.encode(fake_video).latent_dist.sample()

    from models.utils import T5_Embedding
    max_length = 512
    t5_infer = T5_Embedding(mt5_model_name, mt5_cache_dir, max_length).cuda()

    import time
    start_time = time.time()

    text_emb = t5_infer(fake_texts)
    
    x = fake_image
    # posterior = vae.encode(fake_image)

    # # normalize the std of latent to be 1. Change it if you use a different tokenizer
    # if vae.config.shift_factor is not None:
    #     x = (posterior.sample() - vae.config.shift_factor) * vae.config.scaling_factor
    # else:
    #     x = posterior.sample().mul_(vae.config.scaling_factor)

    # import pdb
    # pdb.set_trace()
    # check
    with torch.cuda.amp.autocast():
        loss = model(x, text_emb, height=256, width=256)
    
    end_time = time.time()
    print(f'cost time:{end_time-start_time}')
    import pdb
    pdb.set_trace()
    print(f"Loss: {loss.item()}")