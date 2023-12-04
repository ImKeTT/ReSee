#!/usr/bin/env python
#-*- coding: utf-8 -*-
import math, copy
import torch
import numpy as np
import torch.nn as nn

from transformers.models.t5.modeling_t5 import (
    ACT2FN,
    T5Attention,
    T5Block,
    T5Stack,
    T5Model,
    T5PreTrainedModel,
    T5LayerNorm
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    Seq2SeqLMOutput,
    CausalLMOutputWithCrossAttentions,
    BaseModelOutputWithPastAndCrossAttentions
)

from transformers.utils.model_parallel_utils import (
    assert_device_map, get_device_map
)

from torch.nn import CrossEntropyLoss

class MMT5Stack(T5Stack):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.img_dim = config.img_dim
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        self.img_add_pos = config.img_add_pos
        self.no_ent_vis = config.no_ent_vis
        self.no_turn_vis = config.no_turn_vis
        self.no_vis = self.no_turn_vis and self.no_ent_vis
        if not self.no_vis:
            if self.img_add_pos == "concat" or self.img_add_pos == "enc_cross_attn":
                if not self.is_decoder:
                    self.use_img_layernorm = config.use_img_layernorm
                    self.img_embedding = nn.Sequential(nn.Linear(self.img_dim, self.img_dim, bias=True),
                                                       nn.ReLU(),
                                                       nn.Linear(self.img_dim, self.config.hidden_size, bias=True))
                    if self.use_img_layernorm:
                        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.img_layer_norm_eps)
            else:
                if self.is_decoder:
                    self.use_img_layernorm = config.use_img_layernorm
                    self.img_embedding = nn.Sequential(nn.Linear(self.img_dim, self.img_dim, bias=True),
                                                       nn.ReLU(),
                                                       nn.Linear(self.img_dim, self.config.hidden_size, bias=True))
                    if self.use_img_layernorm:
                        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.img_layer_norm_eps)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.init_weights()
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

    def parallelize(self, device_map=None):
        # Check validity of device_map
        self.device_map = (
            get_device_map(len(self.block), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.block))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        # Load onto devices
        for k, v in self.device_map.items():
            for layer in v:
                cuda_device = "cuda:" + str(k)
                self.block[layer] = self.block[layer].to(cuda_device)

        # Set embed_tokens to first layer
        self.embed_tokens = self.embed_tokens.to(self.first_device)
        # Set final layer norm to last device
        self.final_layer_norm = self.final_layer_norm.to(self.last_device)

    def deparallelize(self):
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        for i in range(len(self.block)):
            self.block[i] = self.block[i].to("cpu")
        self.embed_tokens = self.embed_tokens.to("cpu")
        self.final_layer_norm = self.final_layer_norm.to("cpu")
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        turn_img_feats=None,
        ent_img_feats=None,
        img_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # ----------------------- Multi-modal knowledge -----------------------#
        ## add visual knowledge as encoder input
        if self.img_add_pos == "concat" or self.img_add_pos == "enc_cross_attn":
            if not self.is_decoder and not self.no_vis:
                if not self.no_turn_vis:
                    t_img_embedding_output = self.img_embedding(turn_img_feats)
                    if self.use_img_layernorm:
                        t_img_embedding_output = self.LayerNorm(t_img_embedding_output)
                    img_embedding_output = t_img_embedding_output
                if not self.no_ent_vis:
                    e_img_embedding_output = self.img_embedding(ent_img_feats)
                    if self.use_img_layernorm:
                        e_img_embedding_output = self.LayerNorm(e_img_embedding_output)
                    img_embedding_output = e_img_embedding_output
                if not self.no_ent_vis and not self.no_turn_vis:
                    img_embedding_output = torch.cat((t_img_embedding_output, e_img_embedding_output), 1)
                ## now the image feature is of [bs, img_seq_len, hidden size]
                if self.img_add_pos == "enc_cross_attn":
                    encoder_hidden_states = img_embedding_output
                    encoder_attention_mask = img_attention_mask
                elif self.img_add_pos == "concat":
                    # concatenate two embeddings
                    pass
        ## add visual knowledge as decoder input (concatenate to cross attention)
        elif self.img_add_pos == "dec_cross_attn":
            if self.is_decoder and not self.no_vis:
                assert encoder_hidden_states is not None and img_attention_mask is not None, \
                    "Encoder hidden state and image attention mask should not be None using multi-modal cross attention"
                if not self.no_turn_vis:
                    t_img_embedding_output = self.img_embedding(turn_img_feats)
                    if self.use_img_layernorm:
                        t_img_embedding_output = self.LayerNorm(t_img_embedding_output)
                    img_embedding_output = t_img_embedding_output
                if not self.no_ent_vis:
                    e_img_embedding_output = self.img_embedding(ent_img_feats)
                    if self.use_img_layernorm:
                        e_img_embedding_output = self.LayerNorm(e_img_embedding_output)
                    img_embedding_output = e_img_embedding_output
                if not self.no_ent_vis and not self.no_turn_vis:
                    img_embedding_output = torch.cat((t_img_embedding_output, e_img_embedding_output), 1)
                if encoder_attention_mask is None:
                    encoder_attention_mask = torch.ones_like(
                        encoder_hidden_states, device=encoder_hidden_states.device, dtype=torch.long
                    )
                ## concatenate visual features to encoder hidden states for cross attention
                encoder_attention_mask = torch.cat((encoder_attention_mask, img_attention_mask), 1)
                encoder_hidden_states = torch.cat((encoder_hidden_states, img_embedding_output), 1)
            # -------------------------------------------------------------------#

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, f":obj:`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length).to(inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        if not self.is_decoder and not self.no_vis:
            # ----------------------- Multi-modal knowledge -----------------------#
            if self.img_add_pos == "concat":
                inputs_embeds = torch.cat((inputs_embeds, img_embedding_output), 1)
            # -------------------------------------------------------------------#

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                layer_head_mask=layer_head_mask,
                cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions
        )


class T5ForMMDialogGeneration(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        ## Encoder input can be multi-modal with `concat` or `cross_attn` combining
        self.encoder = MMT5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = MMT5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    def deparallelize(self):
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    # @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        turn_img_feats=None,
        ent_img_feats=None,
        attention_mask=None,
        img_attention_mask=None,
        decoder_input_ids=None,
        masked_decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        masked_ids=None,
        masked_res_pos=None,
        seq_length=None,
        add_text_bias=False,
        mlm_regression=False,
        ent_pos=None,
        cap_pos=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``

        Returns:
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                turn_img_feats=turn_img_feats,
                ent_img_feats=ent_img_feats,
                img_attention_mask=img_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        hidden_states = encoder_outputs[0]
        seq_hidden_states = hidden_states[:, : seq_length, :]

        bs, _, hs = hidden_states.shape

        # ----------------------- Multi-modal task -----------------------#
        hidden_states_biased = None
        if add_text_bias and (ent_pos is not None or cap_pos is not None):
            pos = torch.zeros_like(input_ids, device=input_ids.device)
            if ent_pos is not None:
                pos += ent_pos
            if cap_pos is not None:
                pos += cap_pos
            hidden_states_biased = seq_hidden_states[pos==1, :].view(bs, -1, hs)
        # --------------------------------------------------------------- #

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            turn_img_feats=turn_img_feats,
            ent_img_feats=ent_img_feats,
            img_attention_mask=img_attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # ----------------------- Multi-modal task -----------------------#
        ## dummy input
        biased_sequence_output = None
        mlm_sequence_output = None
        if add_text_bias and hidden_states_biased is not None:
            # Decode with only textual bias
            biased_decoder_outputs = self.decoder(
                        input_ids=decoder_input_ids,
                        attention_mask=decoder_attention_mask,
                        inputs_embeds=decoder_inputs_embeds,
                        past_key_values=past_key_values,
                        encoder_hidden_states=hidden_states_biased,
                        encoder_attention_mask=None,
                        head_mask=decoder_head_mask,
                        cross_attn_head_mask=cross_attn_head_mask,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=return_dict,
                    )
            biased_sequence_output = biased_decoder_outputs[0]

        if mlm_regression and masked_decoder_input_ids is not None:
            # Decode with only textual bias
            mlm_decoder_outputs = self.decoder(
                input_ids=masked_decoder_input_ids,
                attention_mask=decoder_attention_mask,
                inputs_embeds=decoder_inputs_embeds,
                past_key_values=past_key_values,
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=attention_mask,
                head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            mlm_sequence_output = mlm_decoder_outputs[0]
        # --------------------------------------------------------------- #

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

            if add_text_bias:
                biased_sequence_output = biased_sequence_output * (self.model_dim ** -0.5)
            if mlm_regression:
                mlm_sequence_output = mlm_sequence_output * (self.model_dim ** -0.5)

        ## dummy input
        biased_lm_logits = None
        mlm_logits = None
        if add_text_bias:
            biased_lm_logits = self.lm_head(biased_sequence_output)
        if mlm_regression:
            mlm_logits = self.lm_head(mlm_sequence_output)

        lm_logits = self.lm_head(sequence_output)
        ## add textual bias from the input
        lm_logits = lm_logits + biased_lm_logits if biased_lm_logits is not None else lm_logits

        loss = None
        mlm_loss = torch.tensor(0)
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            if mlm_logits is not None:
                mlm_loss = loss_fct(mlm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        if not return_dict:
            output = (lm_logits, mlm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss, mlm_loss, ) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


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

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        turn_img_feats=None,
        ent_img_feats=None,
        img_attention_mask=None,
        **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        # num_beams = kwargs.get("num_beams", None)
        # if turn_img_feats is not None:
        #     turn_img_feats = self._expand_for_beams(turn_img_feats, num_beams)
        # if ent_img_feats is not None:
        #     ent_img_feats = self._expand_for_beams(ent_img_feats, num_beams)
        # if img_attention_mask is not None:
        #     img_attention_mask = self._expand_for_beams(img_attention_mask, num_beams)
        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "turn_img_feats": turn_img_feats,
            "ent_img_feats": ent_img_feats,
            "img_attention_mask": img_attention_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past