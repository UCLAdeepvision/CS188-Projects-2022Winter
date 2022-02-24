---
layout: post
comments: true
title: Generative Adversarial Networks with Transformers
author: Sidi Lu
date: 2022-01-27
---


> While Vision Transformers have caught quite some attention in the community, it is still yet to be explored how such powerful models could work on building powerful GANs. Based on some recent progress in studying Transformers' position encoding system, we want to explore the possibility of building a vision-oriented transformer block that is simple, light-weighted yet effective for a stable training of transformer GANs.


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}
## Introduction
There's already one successful attempt at training transformer GANs ([TransGAN](https://arxiv.org/pdf/2102.07074.pdf)). However, as indicated by the paper, the proposed approach is showing the most significant shortcoming of almost all transformer-based models - it is more data-hungry compared to other architectures by a large margin. Inspired by the recent finding of an un-embedded, prior-based position encoding system ([LinearPos](https://arxiv.org/abs/2108.12409)), we hereby motivate our exploration of a simple, light-weighted yet effective design of a vision-oriented transformer block. We hope such design could alleviate the data efficiency problem of transformers whereas to maintain the merit of its ability to model long dependencies.

## Implementation
```python
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn import functional as F

from transformers.models.xlnet.modeling_xlnet import ACT2FN
from transformers.models.xlnet.modeling_xlnet import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.models.xlnet.modeling_xlnet import (
    PoolerAnswerClass,
    PoolerEndLogits,
    PoolerStartLogits,
    PreTrainedModel,
    SequenceSummary,
    apply_chunking_to_forward,
)
from transformers.models.xlnet.modeling_xlnet import logging
from transformers.models.xlnet.modeling_xlnet import XLNetConfig

class EuclideanTransformerRelativeAttention(nn.Module):
    def __init__(self, config):
        super(EuclideanTransformerRelativeAttention, self).__init__()
        self.config = config
        self.q = nn.Parameter(torch.FloatTensor(config.d_model, config.n_head, config.d_head))
        self.k = nn.Parameter(torch.FloatTensor(config.d_model, config.n_head, config.d_head))
        self.v = nn.Parameter(torch.FloatTensor(config.d_model, 4, config.n_head, config.d_head))
        self.o = nn.Parameter(torch.FloatTensor(config.d_model, config.n_head, config.d_head))
        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    @staticmethod
    def generate_bias_map(h, w, nhead, device, eps=1e-10):
        h_contribution = torch.arange(0, h, dtype=torch.float, device=device).reshape(h, 1).expand(h, w)
        w_contribution = torch.arange(0, w, dtype=torch.float, device=device).reshape(1, w).expand(h, w)
        vec_director = torch.stack([(h_contribution.reshape(h * w, 1) - (h_contribution.reshape(1, h * w))),
                                    (w_contribution.reshape(h * w, 1) - (w_contribution.reshape(1, h * w)))]) # [2, h * w, h * w]
        vec_director = (vec_director + eps) / (torch.norm(vec_director, p=1, dim=[-2, -1], keepdim=True) + eps)
        vec_director = torch.cat((torch.abs(vec_director), -torch.abs(-vec_director)), dim=0)
        h_contribution = h_contribution.reshape(h * w, 1) ** 2 + h_contribution.reshape(1, h * w) ** 2 - 2.0 * (h_contribution.reshape(h * w, 1) @ h_contribution.reshape(1, h * w))
        w_contribution = w_contribution.reshape(h * w, 1) ** 2 + w_contribution.reshape(1, h * w) ** 2 - 2.0 * (w_contribution.reshape(h * w, 1) @ w_contribution.reshape(1, h * w))
        all_dist = (h_contribution + w_contribution) ** 0.5
        all_dist = all_dist * 8
        
        m_contribution = -torch.arange(1, nhead + 1, dtype=torch.float, device=device).reshape(nhead, 1, 1) * 8 / nhead
        m_contribution = torch.exp2(m_contribution)
        bias_map = all_dist.reshape(1, h * w, h * w) * m_contribution
        return bias_map, vec_director
        
    def forward(self,
                h,
                h_pooling,
                verbose=False
                ):
        # h -> [batch_size, h, w, hidden_dim]
        # attention_mask -> [batch_size, seq_len, seq_len]
        # offset -> [batch, seq_len, seq_len]

        # value head
        # position-based key head
        batch_size, h_size, w_size, hidden_dim = h.shape
        _, pool_size, hidden_dim = h_pooling.shape
        device = h.device
        seq_len = h_size * w_size
        h = h.reshape(batch_size, h_size * w_size, hidden_dim)
        h = torch.cat((h, h_pooling), dim=1)
        n_head = self.config.n_head
        attention_mask = torch.ones(size=(1, 1, seq_len + pool_size, seq_len + pool_size), dtype=torch.float, device=device)
        attention_mask[0, 0, seq_len:, :] = 0.0
        attention_mask[0, 0, :, seq_len:] = 0.0
        attention_mask[0, 0, seq_len:, seq_len:] = torch.diag(torch.ones((pool_size,), dtype=torch.float, device=device))
        attention_mask[0, 0, seq_len:, 0:seq_len] = 1.0
        # content-stream query head
        q_head_h = torch.einsum("bih,hnd->bind", h, self.q)
        k_head_h = torch.einsum("bih,hnd->bind", h, self.k)
        v_head = torch.einsum("bih,hknd->biknd", h, self.v)

        content_interaction = torch.einsum("bind,bjnd->bnij", q_head_h, k_head_h)
        m_bias, vec_director = self.generate_bias_map(h_size, w_size, n_head, device=device)
        m_bias_ = torch.zeros(n_head, seq_len + pool_size, seq_len + pool_size, dtype=torch.float, device=device)
        vec_director_ = torch.ones(4, seq_len + pool_size, seq_len + pool_size, dtype=torch.float, device=device) / 4.0
        m_bias_[:, 0:seq_len, 0:seq_len] = m_bias
        vec_director_[:,  0:seq_len, 0:seq_len] = vec_director
        alpha = content_interaction - m_bias_
        # batch nhead seqlen seqlen

        # for numerical stability
        alpha = (alpha - (1.0 - attention_mask) * 1e30).log_softmax(dim=-1) - (1.0 - attention_mask) * 1e30

#        exp_alpha_masked = exp_alpha * attention_mask

        normalized_alpha = alpha.softmax(dim=-1) #exp_alpha_masked / (exp_alpha_masked.sum(dim=-1, keepdims=True))
        normalized_alpha_select_angle = torch.einsum("kij,bnij->bknij", vec_director_, normalized_alpha)

        reduced_v_head = torch.einsum("bknij,bjknd->bind", normalized_alpha_select_angle, v_head)

        transformed_reduced_v_head = torch.einsum("bind,hnd->bih", reduced_v_head, self.o)

        transformed_reduced_v_head = self.dropout(transformed_reduced_v_head)

        h_comp = self.layer_norm(transformed_reduced_v_head)

        return h_comp

class EuclideanTransformerFeedForward(nn.Module):
    def __init__(self, config):
        super(EuclideanTransformerFeedForward, self).__init__()
        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.layer_1 = nn.Linear(config.d_model, config.d_inner)
        self.layer_2 = nn.Linear(config.d_inner, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        if isinstance(config.ff_activation, str):
            self.activation_function = ACT2FN[config.ff_activation]
        else:
            self.activation_function = config.ff_activation

    def forward(self, inp):
        output = inp
        output = self.layer_1(output)
        output = self.activation_function(output)
        output = self.dropout(output)
        output = self.layer_2(output)
        output = self.dropout(output)
        output = self.layer_norm(output + inp)
        return output

class EuclideanTransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rel_attn = EuclideanTransformerRelativeAttention(config)
        self.ff = EuclideanTransformerFeedForward(config)
        self.dropout = nn.Dropout(config.dropout)
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1

    def forward(
        self,
        pixel_embeddings,
        semantic_embeddings,
        verbose=False
    ):
        batch_size, h_size, w_size, hidden_dim = pixel_embeddings.shape
        seq_len = h_size * w_size
        h_comp = self.rel_attn(
            h=pixel_embeddings,
            h_pooling=semantic_embeddings,
            verbose=verbose
        )
        h_comp = self.ff(h_comp)
        h_, h_pooling_ = h_comp[:, 0:seq_len, :], h_comp[:, seq_len:, :]
        pixel_embeddings, semantic_embeddings = h_.reshape(batch_size, h_size, w_size, hidden_dim), h_pooling_

        return pixel_embeddings, semantic_embeddings

class EuclideanTransformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = XLNetConfig
    base_model_prefix = "transformer"

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, EuclideanTransformerRelativeAttention):
            for param in [
                module.q,
                module.k,
                module.v,
                module.o,
            ]:
                param.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, EuclideanTransformerModel):
            module.pool_emb.data.normal_(mean=0.0, std=self.config.initializer_range)
            module.pos_emb.data.normal_(mean=0.0, std=self.config.initializer_range)


class EuclideanTransformerActivation(nn.Module):
    def __init__(self, config: XLNetConfig):
        super().__init__()
        if isinstance(config.ff_activation, str):
            self.activation_function = ACT2FN[config.ff_activation]
        else:
            self.activation_function = config.ff_activation

    def forward(self, x):
        return self.activation_function(x)


class EuclideanTransformerModel(EuclideanTransformerPreTrainedModel):
    def __init__(self, config: XLNetConfig, addon_absolute_pos=False):
        super().__init__(config)

        self.mem_len = config.mem_len
        self.reuse_len = config.reuse_len
        self.d_model = config.d_model
        self.same_length = config.same_length
        self.attn_type = config.attn_type
        self.bi_data = config.bi_data
        self.clamp_len = config.clamp_len
        self.n_layer = config.n_layer
        width = 8
        self.width = width
        self.input_proj = nn.Conv2d(in_channels=3,
                               out_channels=config.d_model,
                               kernel_size=8,
                               stride=8,
                               padding=0,
                               )
        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.pool_emb = nn.Parameter(torch.FloatTensor(1, 1, config.d_model))
        self.pos_emb = nn.Parameter(torch.FloatTensor(64 // width, 64 // width, config.d_model))

        self.layer = nn.ModuleList([EuclideanTransformerLayer(config) for _ in range(config.n_layer)])
        # self._layer = nn.ModuleList([EuclideanTransformerLayer(config)])
        # self.layer = [self._layer[0]] * config.n_layer
        self.dropout = nn.Dropout(config.dropout)

        self.init_weights()

    def forward(self,
                input_pixels
                ):
        input_pixels = self.input_proj(input_pixels)
        pixel_embeddings = input_pixels.permute(0, 2, 3, 1)

        batch_size, h_size, w_size, channel_size = pixel_embeddings.shape
        semantic_embeddings = self.pool_emb.expand(batch_size, 1, self.config.d_model)
        down_sampling = [0, 3, 8,]
        for i, layer_module in enumerate(self.layer):
            batch_size, h_size, w_size, channel_size = pixel_embeddings.shape
            pixel_embeddings, semantic_embeddings = layer_module(
                pixel_embeddings=pixel_embeddings,
                semantic_embeddings=semantic_embeddings,
            )
            # if i in down_sampling:
            #     pixel_embeddings = pixel_embeddings.reshape(batch_size, h_size//2, 2, w_size//2, 2, channel_size)
            #     pixel_embeddings = pixel_embeddings[:, :, -1, :, -1, :]

        return pixel_embeddings, semantic_embeddings
```
The 1-D Linear Bias is now extended to the 2-D world. Multiple diffusion pattern should examined, such as linear, gaussian etc.
## Demo
TBD

## Reference
[1] Jiang Y, Chang S, Wang Z. Transgan: Two transformers can make one strong gan[J]. arXiv preprint arXiv:2102.07074, 2021, 1(2): 7.

[2] Press O, Smith N A, Lewis M. Train short, test long: Attention with linear biases enables input length extrapolation[J]. arXiv preprint arXiv:2108.12409, 2021.

[3] Durall R, Frolov S, Hees J, et al. Combining transformer generators with convolutional discriminators[C]//German Conference on Artificial Intelligence (Künstliche Intelligenz). Springer, Cham, 2021: 67-79.

[4] Arjovsky M, Chintala S, Bottou L. Wasserstein generative adversarial networks[C]//International conference on machine learning. PMLR, 2017: 214-223.

[5] Gulrajani I, Ahmed F, Arjovsky M, et al. Improved training of wasserstein gans[J]. arXiv preprint arXiv:1704.00028, 2017.

## Code Repository
[1] [WGAN/WGAN-GP](https://github.com/Zeleni9/pytorch-wgan.git)

[2] [TransGAN](https://github.com/VITA-Group/TransGAN)

[3] [WGAN-T (Ours)](https://github.com/desire2020/WGAN-T)

---
