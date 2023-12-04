#!/usr/bin/env python
#-*- coding: utf-8 -*-
import torch
import torch.nn as nn
import sys

from transformers.models.gpt2.modeling_gpt2 import (
    ACT2FN,
    GPT2Attention,
    GPT2Model,
    GPT2Block,
    GPT2MLP,
    GPT2LMHeadModel,
    GPT2PreTrainedModel,
    Conv1D
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    CausalLMOutputWithCrossAttentions,
    BaseModelOutputWithPastAndCrossAttentions
)


from transformers.utils.model_parallel_utils import (
    assert_device_map, get_device_map
)

from torch.nn import CrossEntropyLoss

class UnmaskedAttention(GPT2Attention):
    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / (float(value.size(-1)) ** 0.5)

        ## remove causal mask in GPT2, because we make the attention mask
        ## in data preparation stage
        # if not self.is_cross_attention:
        #     # if only "normal" attention layer implements causal mask
            # query_length, key_length = query.size(-2), key.size(-2)
            # causal_mask = self.bias[:, :, key_length - query_length: key_length, :key_length].bool()
            # attn_weights = torch.where(causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype))

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.Softmax(dim=-1)(attn_weights)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

class MMGPT2Block(GPT2Block):
    def __init__(self, config):
        super().__init__(config)
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        if config.add_cross_attention:
            self.crossattention = UnmaskedAttention(config, is_cross_attention=True)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = GPT2MLP(inner_dim, config)

class GPT2ModelForMultiModal(GPT2PreTrainedModel):
    def __init__(self, config, img_add_pos):
        super().__init__(config)
        assert img_add_pos in ["cross_attn", "concat"]
        if img_add_pos == "cross_attn":
            config.add_cross_attention = True

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([MMGPT2Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.use_img_layernorm = config.use_img_layernorm
        self.img_add_pos = img_add_pos
        self.img_feature_type = config.img_feature_type
        self.img_layer_norm_eps = config.img_layer_norm_eps
        self.img_dim = config.img_dim
        self.no_ent_vis = config.no_ent_vis
        self.no_turn_vis = config.no_turn_vis
        self.no_vis = self.no_turn_vis and self.no_ent_vis

        if not self.no_vis:
            self.use_img_layernorm = config.use_img_layernorm
            self.img_embedding = nn.Sequential(nn.Linear(self.img_dim, self.img_dim, bias=True),
                                               nn.ReLU(),
                                               nn.Linear(self.img_dim, self.config.hidden_size, bias=True))
            if self.use_img_layernorm:
                self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.img_layer_norm_eps)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

    def parallelize(self, device_map=None):
        # Check validity of device_map
        self.device_map = (
            get_device_map(len(self.h), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.h))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        self.wte = self.wte.to(self.first_device)
        self.wpe = self.wpe.to(self.first_device)
        # Load onto devices
        for k, v in self.device_map.items():
            for block in v:
                cuda_device = "cuda:" + str(k)
                self.h[block] = self.h[block].to(cuda_device)
        # ln_f to last
        self.ln_f = self.ln_f.to(self.last_device)

    def deparallelize(self):
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        self.wte = self.wte.to("cpu")
        self.wpe = self.wpe.to("cpu")
        for index in range(len(self.h)):
            self.h[index] = self.h[index].to("cpu")
        self.ln_f = self.ln_f.to("cpu")
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)
    def forward(
        self,
        input_ids=None,
        turn_img_feats=None,
        ent_img_feats=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        ent_ids=None,
        return_dict=None,
        **kwargs,
    ):
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            ## let's only consider input_ids is not None for now
            # sent_length = input_ids.shape[-1]
            # if self.no_vis:
            #     input_shape = torch.Size([input_ids.shape[0], sent_length])
            # elif self.no_turn_vis:
            #     input_shape = torch.Size([input_ids.shape[0], sent_length + ent_img_feats.shape[-1]])
            # elif self.no_ent_vis:
            #     input_shape = torch.Size([input_ids.shape[0], sent_length + turn_img_feats.shape[-1]])
            # else:
            #     raise NotImplementedError
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        ## Image feature processing
        ## append image features to token space
        img_feat_not_none = turn_img_feats is not None or ent_img_feats is not None
        img_feat_len = 0
        img_embedding_output = None
        if not self.no_vis and img_feat_not_none:
            # ----------------------- Multi-modal knowledge -----------------------#
            if not self.no_turn_vis:
                t_img_embedding_output = self.img_embedding(turn_img_feats)
                img_feat_len += turn_img_feats.shape[1] ## [batch_size, num_img, img_dim]
                if self.use_img_layernorm:
                    t_img_embedding_output = self.LayerNorm(t_img_embedding_output)
                img_embedding_output = t_img_embedding_output
            if not self.no_ent_vis:
                e_img_embedding_output = self.img_embedding(ent_img_feats)
                img_feat_len += ent_img_feats.shape[1]
                if self.use_img_layernorm:
                    e_img_embedding_output = self.LayerNorm(e_img_embedding_output)
                img_embedding_output = e_img_embedding_output
            if not self.no_ent_vis and not self.no_turn_vis:
                img_embedding_output = torch.cat((t_img_embedding_output, e_img_embedding_output), 1)
            ## now the image feature is of [bs, img_seq_len, hidden size]
            if self.img_add_pos == "cross_attn":
                encoder_hidden_states = img_embedding_output

            # -------------------------------------------------------------------#

        input_shape = torch.Size([input_shape[0], input_shape[-1] + img_feat_len])
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = [None] * len(self.h)
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                ## if cross attention but the attention mask is None
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None


        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        ## concat image embedding features to input ids embeddings (at embedding space)
        if self.img_add_pos == "concat" and img_embedding_output is not None:
            inputs_embeds = torch.cat((img_embedding_output, inputs_embeds), 1)

        position_embeds = self.wpe(position_ids)

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0

        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states)

        # if not self.no_vis and img_feat_not_none:
        #     ## concat image embedding features to input ids embeddings
        #     if self.img_add_pos == "concat":
        #         hidden_states = torch.cat((img_embedding_output, hidden_states), 1)
        #         output_shape = hidden_states.shape
        #     else:
        #         output_shape = input_shape + (hidden_states.size(-1),)
        # else:
        #     output_shape = input_shape + (hidden_states.size(-1),)
        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)
            if self.gradient_checkpointing and self.training:

                if use_cache:
                    print(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

class GPT2LMforMMDialog(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.img_add_pos = config.img_add_pos
        ## ablation for visual knowledge
        self.no_ent_vis = config.no_ent_vis
        self.no_turn_vis = config.no_turn_vis
        self.no_vis = self.no_turn_vis and self.no_ent_vis
        self.context_len = config.context_length

        self.transformer = GPT2ModelForMultiModal(config, self.img_add_pos)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.vocab_size = config.vocab_size
        # self.ids_for_bias_vacab = torch.load('./ids_for_bias_vacab.pth')
        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.transformer.h))
        self.transformer.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.transformer.first_device)
        self.model_parallel = True

    def deparallelize(self):
        self.transformer.deparallelize()
        self.transformer = self.transformer.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        torch.cuda.empty_cache()

    def get_output_embeddings(self):
        return self.lm_head

    def _expand_for_beams(self, x, num_beams, num_fsm_states=1):
        num_expand = num_beams * num_fsm_states
        if x is None or num_expand == 1:
            return x

        input_shape = list(x.shape)
        expanded_shape = input_shape[:1] + [num_expand] + input_shape[1:]
        x = x.unsqueeze(1).expand(expanded_shape)
        # (batch_size * num_beams, ...)
        x = x.contiguous().view([input_shape[0] * num_expand] + input_shape[1:])
        return x

    def prepare_inputs_for_generation(self,
                                      input_ids,
                                      turn_img_feats=None,
                                      ent_img_feats=None,
                                      add_text_bias=False,
                                      ent_pos=None,
                                      past=None,
                                      **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        # if input_ids.size(-1) != self.context_len:
        #     input_ids = input_ids[:, -self.context_len:]
        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        img_feat_length = 0
        if self.img_add_pos=="concat":
            if not self.no_vis:
                if not self.no_ent_vis:
                    img_feat_length += ent_img_feats.shape[1]
                if not self.no_turn_vis:
                    img_feat_length += turn_img_feats.shape[1]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None

        first_decoding = input_ids.size(-1)==self.context_len
        if not first_decoding:
            turn_img_feats = ent_img_feats = None

        ## expand image feature for beam search
        num_beams = kwargs.get("num_beams", None)
        if num_beams is not None:
            turn_img_feats = self._expand_for_beams(turn_img_feats, num_beams) if turn_img_feats is not None else None
            ent_img_feats = self._expand_for_beams(ent_img_feats, num_beams) if ent_img_feats is not None else None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "turn_img_feats": turn_img_feats,
            "ent_img_feats": ent_img_feats,
            "add_text_bias": add_text_bias,
            "ent_pos": ent_pos,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def forward(
            self,
            input_ids=None,
            turn_img_feats=None,
            ent_img_feats=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            add_text_bias=False,
            ent_pos=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for language modeling.
            Note that the labels **are shifted** inside the model, i.e. you can set ``labels = input_ids``
            Indices are selected in ``[-100, 0, ..., config.vocab_size]``
            All labels set to ``-100`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``
        """
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        img_feat_length = 0
        img_feat_not_none = turn_img_feats is not None or ent_img_feats is not None
        if not self.no_vis and img_feat_not_none:
            if not self.no_turn_vis:
                img_feat_length += turn_img_feats.shape[1]
            if not self.no_ent_vis:
                img_feat_length += ent_img_feats.shape[1]

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            turn_img_feats=turn_img_feats,
            ent_img_feats=ent_img_feats,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        ## remove image mask if concat input
        trunc_length = 0
        if self.img_add_pos == "concat":
            trunc_length = img_feat_length
        hidden_states = transformer_outputs[0][:, trunc_length:, :]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        if not return_dict:
            output = (lm_logits, ) + transformer_outputs[1:]
            return ((loss, ) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )