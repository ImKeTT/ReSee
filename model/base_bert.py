#!/usr/bin/env python3
#-*- coding: utf-8 -*-
import logging
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn.modules.loss import _Loss
from transformers.models.bert.modeling_bert import (
    BertSelfAttention,
    BertAttention,
    BertEncoder,
    BertOnlyMLMHead,
    BertOnlyNSPHead,
    BertLMPredictionHead,
    BertLayer,
    BertSelfOutput,
    BertIntermediate,
    BertOutput,
    BertPooler,
    BertPreTrainedModel,
    BertPredictionHeadTransform
)
from transformers.modeling_utils import (
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from model.modeling_utils import MultimodalTrainedModel
# from utils.cbs import ConstrainedBeamSearch, select_best_beam_with_constraints


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.max_position_embeddings = config.max_position_embeddings

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MMBertSelfAttention(BertSelfAttention):
    """
    Modified from BertSelfAttention to add support for output_hidden_states.
    """

    def __init__(self, config):
        super(MMBertSelfAttention, self).__init__(config)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
            history_state=None
    ):
        if history_state is not None:
            if encoder_hidden_states is not None:
                encoder_hidden_states = torch.cat([history_state, encoder_hidden_states], dim=1)
            else:
                kv_hidden_states = torch.cat([history_state, hidden_states], dim=1)

        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            if history_state is not None:
                key_layer = self.transpose_for_scores(self.key(kv_hidden_states))
                value_layer = self.transpose_for_scores(self.value(kv_hidden_states))
            else:
                key_layer = self.transpose_for_scores(self.key(hidden_states))
                value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class MMBertAttention(BertAttention):
    """
    Modified from BertAttention to add support for output_hidden_states.
    """

    def __init__(self, config):
        super(MMBertAttention, self).__init__(config)
        self.self = MMBertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
            history_state=None
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
            history_state
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class MMBertLayer(BertLayer):
    """
    Modified from BertLayer to add support for output_hidden_states.
    """

    def __init__(self, config):
        super(MMBertLayer, self).__init__(config)
        self.attention = MMBertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self,
                hidden_states,
                attention_mask,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                past_key_value=None,
                history_state=None,
                output_attentions=False,):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
            history_state=history_state,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

class MMBertEncoder(BertEncoder):
    """
    Modified from BertEncoder to add support for output_hidden_states.
    v4.11.0
    """
    def __init__(self, config):
        super(MMBertEncoder, self).__init__(config)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([MMBertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self,
                hidden_states,
                attention_mask,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                past_key_values=None,
                use_cache=None,
                output_attentions=False,
                output_hidden_states=False,
                encoder_history_states=None):

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            history_state = encoder_history_states[i] if encoder_history_states is not None else None
            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None
            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    history_state,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return tuple(
            v
            for v in [
                hidden_states,
                next_decoder_cache,
                all_hidden_states,
                all_self_attentions,
                all_cross_attentions,
            ]
            if v is not None
        )  # outputs, (hidden states), (attentions)


class MMBertModel(BertPreTrainedModel):
    """ Expand from BertModel to handle image region features as input
    """

    def __init__(self, config, img_add_pos):
        super(MMBertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = MMBertEncoder(config)
        self.pooler = BertPooler(config)
        self.img_add_pos=img_add_pos
        self.no_ent_vis = config.no_ent_vis
        self.no_turn_vis = config.no_turn_vis
        self.no_vis = self.no_turn_vis and self.no_ent_vis

        self.img_dim = config.img_dim
        self.img_feature_type = config.img_feature_type
        if hasattr(config, 'use_img_layernorm'):
            self.use_img_layernorm = config.use_img_layernorm
        else:
            self.use_img_layernorm = None

        if not self.no_vis:
            self.use_img_layernorm = config.use_img_layernorm
            self.img_embedding = nn.Sequential(nn.Linear(self.img_dim, self.img_dim, bias=True),
                                               nn.ReLU(),
                                               nn.Linear(self.img_dim, self.config.hidden_size, bias=True))
            if self.use_img_layernorm:
                self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.img_layer_norm_eps)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.init_weights()

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                position_ids=None,
                head_mask=None,
                turn_img_feats=None,
                ent_img_feats=None,
                img_attention_mask=None,
                inputs_embeds=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                past_key_values=None,
                output_attentions=None,
                output_hidden_states=None,
                encoder_history_states=None):

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if turn_img_feats is not None and ent_img_feats is not None and not self.no_vis:
            ## Image feature processing
            ## append image features to token space
            if not self.no_turn_vis:
                t_img_embedding_output = self.img_embedding(turn_img_feats)
                if self.use_img_layernorm:
                    t_img_embedding_output = self.LayerNorm(t_img_embedding_output)
                t_img_embedding_output = self.dropout(t_img_embedding_output)
                img_embedding_output = t_img_embedding_output
            if not self.no_ent_vis:
                e_img_embedding_output = self.img_embedding(ent_img_feats)
                if self.use_img_layernorm:
                    e_img_embedding_output = self.LayerNorm(e_img_embedding_output)
                e_img_embedding_output = self.dropout(e_img_embedding_output)
                img_embedding_output = e_img_embedding_output

            ## now the image feature is of [bs, img_seq_len, hidden size]
            if not self.no_ent_vis and not self.no_turn_vis:
                img_embedding_output = torch.cat((t_img_embedding_output, e_img_embedding_output), 1)

            if self.img_add_pos == "cross_attn":
                encoder_hidden_states = img_embedding_output
                encoder_attention_mask = img_attention_mask
            elif self.img_add_pos == "concat":
                # concatenate two embeddings
                pass
            else:
                raise NotImplementedError

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(
                    -1)  # We can specify head_mask for each layer
            # switch to float if needed + fp16 compatibility
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype)  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids,
                                           position_ids=position_ids,
                                           token_type_ids=token_type_ids,
                                           inputs_embeds=inputs_embeds,
                                           past_key_values_length=past_key_values_length,)
        ## concat image embedding features to input ids embeddings
        if not self.no_vis and self.img_add_pos == "concat":
            embedding_output = torch.cat((embedding_output, img_embedding_output), 1)

        if encoder_history_states:
            assert turn_img_feats is None and ent_img_feats is None, \
                "Cannot take image features while using encoder history states"


        encoder_outputs = self.encoder(embedding_output,
                                       extended_attention_mask,
                                       head_mask=head_mask,
                                       encoder_hidden_states=encoder_hidden_states,
                                       encoder_attention_mask=encoder_attention_mask,
                                       past_key_values=past_key_values,
                                       output_attentions=output_attentions,
                                       output_hidden_states=output_hidden_states,
                                       encoder_history_states=encoder_history_states,)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        # add hidden_states and attentions if they are here
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        return outputs


def instance_bce_with_logits(logits, labels, reduction='mean'):
    assert logits.dim() == 2
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction=reduction)
    if reduction == 'mean':
        loss *= labels.size(1)
    return loss


class ImageBertForSequenceClassification(BertPreTrainedModel):
    """
    Modified from BertForSequenceClassification to support oscar training.
    """

    def __init__(self, config):
        super(ImageBertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.loss_type = config.loss_type
        self.config = config
        if config.img_feature_dim > 0:
            self.bert = BertImgModel(config)
        else:
            self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if hasattr(config, 'classifier'):
            if not hasattr(config, 'cls_hidden_scale'):
                config.cls_hidden_scale = 2

            if config.classifier == 'linear':
                self.classifier = nn.Linear(config.hidden_size,
                                            self.config.num_labels)
            elif config.classifier == 'mlp':
                self.classifier = nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size * config.cls_hidden_scale),
                    nn.ReLU(),
                    nn.Linear(config.hidden_size * config.cls_hidden_scale, self.config.num_labels)
                )
        else:
            self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)  # original
        self.apply(self.init_weights)

    def init_code_embedding(self, em):
        self.bert.code_embeddings.weight.data = em.clone()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None, img_feats=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask, img_feats=img_feats)
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            if self.num_labels == 1:  # doing regression
                loss_fct = MSELoss()
                labels = labels.to(torch.float)
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                if self.loss_type == 'kl':
                    # KL Loss: https://github.com/uclanlp/visualbert/blob/master/pytorch_pretrained_bert/modeling.py
                    loss_fct = torch.nn.KLDivLoss(reduction="batchmean")
                    log_softmax = torch.nn.LogSoftmax(dim=-1)
                    reshaped_logits = logits.contiguous().view(-1, 3129)
                    reshaped_logits = log_softmax(reshaped_logits)
                    loss = loss_fct(reshaped_logits, labels.contiguous())
                elif self.loss_type == 'bce':  # [VQA]
                    loss = instance_bce_with_logits(logits, labels)
                else:  # cross_entropy [GQA, Retrieval, Captioning]
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs


class ImageBertForMultipleChoice(BertPreTrainedModel):
    """
    Modified from BertForMultipleChoice to support oscar training.
    """

    def __init__(self, config):
        super(ImageBertForMultipleChoice, self).__init__(config)
        self.loss_type = config.loss_type
        if config.img_feature_dim > 0:
            self.bert = BertImgModel(config)  # ImageBERT
        else:
            self.bert = BertModel(config)  # original BERT

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if hasattr(config, 'classifier'):
            if not hasattr(config, 'cls_hidden_scale'): config.cls_hidden_scale = 2
            if config.classifier == 'linear':
                self.classifier = nn.Linear(config.num_choice * config.hidden_size, self.config.num_labels)
            elif config.classifier == 'mlp':
                self.classifier = nn.Sequential(
                    nn.Linear(config.num_choice * config.hidden_size, config.hidden_size * config.cls_hidden_scale),
                    nn.ReLU(),
                    nn.Linear(config.hidden_size * config.cls_hidden_scale, self.config.num_labels)
                )
        else:
            self.classifier = nn.Linear(config.num_choice * config.hidden_size, self.config.num_labels)  # original

        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None, img_feats=None):
        num_choices = input_ids.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None

        flat_img_feats = img_feats.view(-1, img_feats.size(-2), img_feats.size(-1)) if img_feats is not None else None

        if isinstance(self.bert, BertImgModel):
            outputs = self.bert(flat_input_ids, position_ids=flat_position_ids, token_type_ids=flat_token_type_ids,
                                attention_mask=flat_attention_mask, head_mask=head_mask, img_feats=flat_img_feats)
        else:
            outputs = self.bert(flat_input_ids, position_ids=flat_position_ids, token_type_ids=flat_token_type_ids,
                                attention_mask=flat_attention_mask, head_mask=head_mask)
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)

        # reshaped_pool_output
        reshaped_pool_output = pooled_output.view(-1, self.config.num_choice * (pooled_output.shape[1]))
        logits = self.classifier(reshaped_pool_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.loss_type == 'bce':
                loss = instance_bce_with_logits(logits, labels.view(-1, self.config.num_labels))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels)
            outputs = (loss,) + outputs
        return outputs

class LabelSmoothingLoss(_Loss):
    def __init__(self, label_smoothing=0, tgt_vocab_size=0, ignore_index=0, size_average=None, reduce=None, reduction='mean'):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__(
            size_average=size_average, reduce=reduce, reduction=reduction)

        assert label_smoothing > 0
        assert tgt_vocab_size > 0

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))
        self.confidence = 1.0 - label_smoothing
        self.tgt_vocab_size = tgt_vocab_size

    def forward(self, output, target):
        assert self.tgt_vocab_size == output.size(1)
        batch_size_mul_num_pos = target.size(0)
        output = output.view(-1, self.tgt_vocab_size)
        target = target.view(-1)
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        return F.kl_div(output, model_prob.type_as(output), reduction='none').view(batch_size_mul_num_pos, -1).sum(1).mean(0)

CAP_BIAS_VOCAB_PTH="/data/tuhq/multimodal/wow/cap_ent_vocab/cap_vocab.pt"
ENT_BIAS_VOCAB_PTH="/data/tuhq/multimodal/wow/cap_ent_vocab/large_ent_vocab.pt"
class BertForMMDialog(MultimodalTrainedModel):
    """
    Bert for Multi-Modal Dialog
    """

    def __init__(self, config, img_add_pos):
        super(BertForMMDialog, self).__init__(config)
        self.config = config
        self.bert = MMBertModel(config, img_add_pos)
        # self.cls = BertOnlyMLMHead(config)
        self.transform = BertPredictionHeadTransform(config)
        self.img_add_pos = img_add_pos
        self.no_ent_vis = config.no_ent_vis
        self.no_turn_vis = config.no_turn_vis

        bert_embedding_weight = self.bert.embeddings.word_embeddings.weight
        self.decoder = nn.Linear(bert_embedding_weight.size(1),
                                 bert_embedding_weight.size(0), bias=False)
        self.bias = nn.Parameter(torch.zeros(bert_embedding_weight.size(0)))

        self.init_weights()
        self.tie_weights()

        # self.matcher = HungarianMatcher(cost_class=1)
        self.vocab_size = bert_embedding_weight.size(0)
        self.cap_bias_vocab_ids = torch.load(CAP_BIAS_VOCAB_PTH).unsqueeze(0)
        self.ent_bias_vocab_ids = torch.load(ENT_BIAS_VOCAB_PTH).unsqueeze(0)

    def get_output_embeddings(self):
        return self.decoder

    def get_input_embeddings(self):
        return self.bert.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.bert.embeddings.word_embeddings = value

    def build_smoothing_loss(self):
        if self.config.label_smoothing > 0:
            self.smoothed_lm_loss = LabelSmoothingLoss(self.config.label_smoothing,
                                                       self.vocab_size, ignore_index=0,
                                                       reduction='none')
        else:
            self.smoothed_lm_loss = None

    def forward(self, *args, **kwargs):
        is_decode = kwargs.get('is_decode', False)
        if is_decode:
            return self.generate(*args, **kwargs)
        else:
            return self.encode_forward(*args, **kwargs)

    def add_bias_logits(self, ori_logits, sequence_output, pos, vacab_ids):
        assert pos is not None
        # ------add texutal bias logits to response generation by ImKe---------
        sequence_output_add_part = sequence_output[pos == 1, :]  # [b_s* tag_num, hidden_emb_size]
        transformed_output_add_part = self.transform(sequence_output_add_part)
        add_logits_ori = self.decoder(transformed_output_add_part)  # [b_s* tag_num, vocab_size]
        add_logits = add_logits_ori.mean(0)  # [ vocab_size]

        # mask the non-tag vocab part
        add_vocab_mask = torch.zeros(self.vocab_size, dtype=torch.float, device=add_logits.device)
        add_vocab_mask[vacab_ids] = 1.0

        # add tag logit bias
        add_logits = add_vocab_mask * add_logits
        logits = ori_logits + add_logits

        return logits

    def compute_score_with_logits(self, logits, labels):
        logits = torch.max(logits, -1)[1].data  # argmax
        scores = logits == labels
        return scores

    def encode_forward(self,
                       input_ids,
                       input_ids_rmask=None,
                       turn_img_feats=None,
                       ent_img_feats=None,
                       attention_mask=None,
                       img_attention_mask=None,
                       inputs_embeds=None,
                       encoder_hidden_states=None,
                       encoder_attention_mask=None,
                       masked_res=None,
                       masked_ids=None,
                       masked_pos=None,
                       masked_res_pos=None,
                       token_type_ids=None,
                       position_ids=None,
                       head_mask=None,
                       is_training=True,
                       encoder_history_states=None,
                       add_ent_bias=False,
                       add_cap_bias=False,
                       add_ent_matching=False,
                       ent_ids=None,
                       ent_pos=None,
                       cap_pos=None,
                       ent_img_pos=None,
                       past_key_values=None,):

        outputs = self.bert(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            turn_img_feats=turn_img_feats,
            ent_img_feats=ent_img_feats,
            img_attention_mask=img_attention_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            encoder_history_states=encoder_history_states,)

        sequence_output = outputs[0][:, :masked_pos.shape[-1], :]
        if self.img_add_pos == "concat":
            img_output = outputs[0][:, masked_pos.shape[-1]:, :]
        else:
            img_output = None

        if is_training:
            if input_ids_rmask is not None:
                outputs = self.bert(
                    input_ids_rmask,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    turn_img_feats=turn_img_feats,
                    ent_img_feats=ent_img_feats,
                    img_attention_mask=img_attention_mask,
                    inputs_embeds=inputs_embeds,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    encoder_history_states=encoder_history_states, )

                masked_res_seq_output = outputs[0][:, :masked_pos.shape[-1], :]
                masked_res_seq_output = masked_res_seq_output[masked_res_pos==1, :]
            else:
                masked_res_seq_output=None

            # num_masks_in_batch * hidden_size
            sequence_output_masked = sequence_output[masked_pos==1, :]

            transformed_output_masked = self.transform(sequence_output_masked)
            class_logits = self.decoder(transformed_output_masked)

            if masked_res_seq_output is not None:
                transformed_output_res_masked = self.transform(masked_res_seq_output)
                masked_res_lm_logits = self.decoder(transformed_output_res_masked)

            ## bias loss caculation
            if add_cap_bias and cap_pos is not None:
                class_logits = self.add_bias_logits(class_logits, sequence_output, cap_pos, self.cap_bias_vocab_ids)

            if add_ent_bias and ent_pos is not None:
                class_logits = self.add_bias_logits(class_logits, sequence_output, ent_pos, self.ent_bias_vocab_ids)

            # Flatten the tokens
            if self.config.label_smoothing > 0:
                loss_fct = lambda source, target: self.smoothed_lm_loss(F.log_softmax(source, dim=-1), target)
            else:
                loss_fct = lambda source, target: F.cross_entropy(source.view(-1, source.size(-1)), target.view(-1))

            ## Entity matching loss
            if add_ent_matching and ent_ids is not None:
                assert ent_pos is not None, "Add textual entity at input for matching.."
                if self.img_add_pos=="cross_attn":
                    ent_seq_output = sequence_output[ent_pos==1, :]
                elif self.img_add_pos=="concat":
                    assert ent_img_pos is not None, "Add textual entity at input for matching.."
                    ent_seq_output = img_output[ent_img_pos==1, :]
                else:
                    raise NotImplementedError
                transformed_output_ent = self.transform(ent_seq_output)
                ent_logits = self.decoder(transformed_output_ent)
                ent_matching_loss = loss_fct(ent_logits, ent_ids)
                batch_ent_score = self.compute_score_with_logits(ent_logits.view(-1, ent_logits.size(-1)), ent_ids.view(-1))
                batch_ent_acc = torch.sum(batch_ent_score.float()) / torch.sum(ent_pos)
            else:
                ent_matching_loss = torch.tensor(0)
                batch_ent_acc = torch.tensor(0)

            ## masked LM loss calculation
            # ---only calculating the masked response and context part---- #
            masked_token_loss = loss_fct(class_logits, masked_ids)

            if masked_res_seq_output is not None:
                masked_res_token_loss = loss_fct(masked_res_lm_logits, masked_res)
                batch_masked_res_score = self.compute_score_with_logits(masked_res_lm_logits.view(-1, masked_res_lm_logits.size(-1)), masked_res.view(-1))
                batch_res_acc = torch.sum(batch_masked_res_score.float()) / torch.sum(masked_res_pos)
            else:
                masked_res_token_loss = torch.tensor(0)
                batch_res_acc = torch.tensor(0)

            batch_masked_score = self.compute_score_with_logits(class_logits.view(-1, class_logits.size(-1)), masked_ids.view(-1))
            batch_acc = torch.sum(batch_masked_score.float()) / torch.sum(masked_pos)

            outputs = (masked_token_loss, masked_res_token_loss, ent_matching_loss, 
                       class_logits, batch_acc, batch_res_acc,
                       batch_ent_acc,) + outputs[2:]

        else:
            class_logits = self.decoder(self.transform(sequence_output))
            outputs = (class_logits,) + outputs[2:]
        return outputs

    def _expand_for_beams(self, x, num_beams, num_fsm_states):
        num_expand = num_beams * num_fsm_states
        if x is None or num_expand == 1:
            return x

        input_shape = list(x.shape)
        expanded_shape = input_shape[:1] + [num_expand] + input_shape[1:]
        x = x.unsqueeze(1).expand(expanded_shape)
        # (batch_size * num_beams, ...)
        x = x.contiguous().view([input_shape[0] * num_expand] + input_shape[1:])
        return x

    def prepare_inputs_for_generation(self, curr_ids, past=None):
        # NOTE: if attention is on, it should be the token used to mask words in training
        mask_token_id = self.mask_token_id
        batch_size = curr_ids.shape[0]
        mask_ids = torch.full(
            (batch_size, 1), mask_token_id, dtype=torch.long, device=curr_ids.device
        )

        def _slice(t, start, end):
            if t is None:
                return t
            return t[:, start: end]

        def _remove_elements(t, start, end):
            if t is None:
                return t
            # assert t.shape == (batch_size, self.max_seq_len + self.cap_len + self.ent_labels_len)
            return torch.cat([t[:, :start], t[:, end:]], dim=1)

        if past is None:
            ## generation process: mask the next word, then predict the masked word for one step generation
            input_ids = torch.cat([curr_ids, mask_ids], dim=1)
            input_ids = torch.cat([input_ids, self.context_ids], dim=1)

            curr_len = input_ids.shape[1]
            if self.img_add_pos == "concat":
                full_len = self.max_seq_len + self.context_len
                if not self.no_turn_vis:
                    full_len += self.turn_img_len
                if not self.no_ent_vis:
                    full_len += self.ent_img_len
                if self.add_cap_labels:
                    full_len += self.cap_len
                if self.add_ent_labels:
                    full_len += self.ent_labels_len
            elif self.img_add_pos == "cross_attn":
                full_len = self.max_seq_len + self.context_len
                if self.add_cap_labels:
                    full_len += self.cap_len
                if self.add_ent_labels:
                    full_len += self.ent_labels_len
            elif self.img_add_pos is None:
                full_len = self.max_seq_len + self.context_len
            else:
                raise NotImplementedError

            assert self.full_attention_mask.shape == (batch_size, full_len, full_len), \
                f"expect size of {(batch_size, full_len.item(), full_len.item())}, get {self.full_attention_mask.shape}."

            def _remove_rows_cols(t, row_start, row_end, col_start, col_end):
                t00 = t[:, :row_start, :col_start]
                t01 = t[:, :row_start, col_end:]
                t10 = t[:, row_end:, :col_start]
                t11 = t[:, row_end:, col_end:]
                res = torch.cat([torch.cat([t00, t01], dim=2), torch.cat([t10, t11],
                                                                         dim=2)], dim=1)
                assert res.shape == (t.shape[0], t.shape[1] - row_end + row_start,
                                     t.shape[2] - col_end + col_start)
                return res

            seq_start = curr_len
            seq_end = self.max_seq_len + self.context_len

            ## reshape current attention mask
            attention_mask = _remove_rows_cols(self.full_attention_mask, seq_start, seq_end, seq_start, seq_end)

            masked_pos = _remove_elements(self.full_masked_pos, seq_start, seq_end)
            token_type_ids = _remove_elements(self.full_token_type_ids, seq_start, seq_end)
            position_ids = _remove_elements(self.full_position_ids, seq_start, seq_end)
            turn_img_feats = self.turn_img_feats
            ent_img_feats = self.ent_img_feats

            if self.add_cap_labels:
                assert self.cap_ids.shape[1] == self.cap_len
                input_ids = torch.cat([input_ids, self.cap_ids], dim=1)

            if self.add_ent_labels:
                assert self.ent_label_ids.shape[1] == self.ent_labels_len
                input_ids = torch.cat([input_ids, self.ent_label_ids], dim=1)
        else:
            last_token = curr_ids[:, -1:]
            # The representation of last token should be re-computed, because
            # it depends on both self-attention context and input tensor
            input_ids = torch.cat([last_token, mask_ids], dim=1)
            start_pos = curr_ids.shape[1] - 1
            end_pos = start_pos + input_ids.shape[1]
            masked_pos = _slice(self.full_masked_pos, start_pos, end_pos)
            token_type_ids = _slice(self.full_token_type_ids, start_pos, end_pos)
            position_ids = _slice(self.full_position_ids, start_pos, end_pos)

            turn_img_feats = None
            ent_img_feats = None
            assert past[0].shape[0] == batch_size
            # print('past[0].shape', past[0].shape)
            if self.prev_encoded_layers is None:
                assert start_pos == 1  # the first token after BOS
                # assert past[0].shape[1] == 2 + self.ent_labels_len + self.img_seq_len
                # reorder to [od_labels, img_feats, sentence]
                self.prev_encoded_layers = [
                    torch.cat([x[:, 2:, :], x[:, :start_pos, :]], dim=1)
                    for x in past]
                s2s = self.full_attention_mask[:, :self.max_seq_len,
                      :self.max_seq_len]
                s2i = self.full_attention_mask[:, :self.max_seq_len,
                      self.max_seq_len:]
                i2s = self.full_attention_mask[:, self.max_seq_len:,
                      :self.max_seq_len]
                i2i = self.full_attention_mask[:, self.max_seq_len:,
                      self.max_seq_len:]
                self.full_attention_mask = torch.cat(
                    [torch.cat([i2i, i2s], dim=2),
                     torch.cat([s2i, s2s], dim=2)],
                    dim=1)
            else:
                assert start_pos > 1
                assert past[0].shape[1] == 2
                self.prev_encoded_layers = [torch.cat([x, p[:, :-1, :]], dim=1)
                                            for x, p in zip(self.prev_encoded_layers, past)]

            curr_full_len = self.context_len
            if self.img_add_pos == "concat":
                curr_full_len = self.context_len
                if not self.no_turn_vis:
                    curr_full_len += self.turn_img_len
                if not self.no_ent_vis:
                    curr_full_len += self.ent_img_len
                if self.add_cap_labels:
                    curr_full_len += self.cap_len
                if self.add_ent_labels:
                    curr_full_len += self.ent_labels_len
            elif self.img_add_pos == "cross_attn":
                ## let's not support cross attention for now
                pass
            elif self.img_add_pos is None:
                pass
            else:
                raise NotImplementedError
            attention_mask = self.full_attention_mask[:,
                             curr_full_len + start_pos: curr_full_len + end_pos,
                             :curr_full_len + end_pos]

        return {'input_ids': input_ids, 'turn_img_feats': turn_img_feats,
                'ent_img_feats': ent_img_feats, 'masked_pos': masked_pos,
                'attention_mask': attention_mask, 'token_type_ids': token_type_ids,
                'position_ids': position_ids, 'is_training': False,
                'past_key_values': self.prev_encoded_layers}

    def generate(self,
                 turn_img_feats,
                 ent_img_feats,
                 attention_mask,
                 masked_pos,
                 token_type_ids=None,
                 position_ids=None,
                 head_mask=None,
                 input_ids=None,
                 max_length=None,
                 img_attention_mask=None,
                 do_sample=None,
                 num_beams=None,
                 temperature=None,
                 top_k=None,
                 top_p=None,
                 repetition_penalty=None,
                 bos_token_id=None,
                 pad_token_id=None,
                 eos_token_ids=None,
                 mask_token_id=None,
                 length_penalty=None,
                 num_return_sequences=None,
                 num_keep_best=1,
                 add_cap_labels=False,
                 add_ent_labels=False,
                 context_start_posid=None,
                 context_end_posid=None,
                 cap_start_posid=None,
                 ent_labels_start_posid=None,
                 use_cbs=False,
                 fsm=None,
                 num_constraints=None,
                 is_decode=None,
                 min_constraints_to_satisfy=None,
                 use_hypo=False,
                 gth_ids=None):

        batch_size = turn_img_feats.shape[0]
        self.turn_img_len = turn_img_feats.shape[1]
        self.ent_img_len = ent_img_feats.shape[1]

        ## max response length
        self.max_seq_len = max_length
        self.mask_token_id = mask_token_id
        self.img_attention_mask=img_attention_mask
        self.prev_encoded_layers = None
        # NOTE: num_keep_best is not equavilant to num_return_sequences
        # num_keep_best is the number of hypotheses to keep in beam search
        # num_return_sequences is the repeating times of input, coupled with
        # do_sample=True can generate more than one samples per image
        self.num_keep_best = num_keep_best

        vocab_size = self.config.vocab_size
        if not use_cbs:
            num_fsm_states = 1
        else:
            b, num_fsm_states, f1, v = fsm.shape
            assert b == batch_size and v == vocab_size and f1 == num_fsm_states

        self.add_ent_labels = add_ent_labels
        self.add_cap_labels = add_cap_labels
        # avoid position_ids collision of caption and od labels

        self.context_start_posid = context_start_posid
        self.context_end_posid = context_end_posid
        self.cap_start_posid = cap_start_posid
        self.ent_labels_start_posid = ent_labels_start_posid

        context_ids = input_ids[:, self.context_start_posid: self.context_end_posid]
        self.context_ids = self._expand_for_beams(context_ids, num_beams, num_fsm_states)
        self.context_len = self.context_end_posid - self.context_start_posid

        if self.add_cap_labels:
            assert input_ids.shape[0] == batch_size
            if ent_labels_start_posid is not None:
                cap_ids = input_ids[:, self.cap_start_posid: self.ent_labels_start_posid]
                self.cap_len = self.ent_labels_start_posid - self.cap_start_posid
            else:
                cap_ids = input_ids[:, self.cap_start_posid: ]
                self.cap_len = input_ids.shape[1] - self.cap_start_posid

            self.cap_ids = self._expand_for_beams(cap_ids, num_beams, num_fsm_states)
        else:
            self.cap_len = 0
            self.cap_ids = None


        if self.add_ent_labels:
            # get entity labels part from input_ids
            ent_label_ids = input_ids[:, self.ent_labels_start_posid:]
            # add for calculate the PPL by Jokie
            self.ent_labels_len = input_ids.shape[1] - self.ent_labels_start_posid ## remove eos token
            self.ent_label_ids = self._expand_for_beams(ent_label_ids, num_beams, num_fsm_states)
        else:
            self.ent_labels_len = 0
            self.ent_label_ids = None

        self.additional_len = self.context_len + self.cap_len + self.ent_labels_len
        input_ids = None

        if input_ids is None:
            input_ids = torch.full(
                (batch_size, 1), bos_token_id, dtype=torch.long, device=next(self.parameters()).device
            )
        else:
            assert input_ids.dim() == 2, "Input prompt should be of shape (batch_size, sequence length)."
            assert input_ids.shape[0] == batch_size, "Input batch size must match image features"

        if position_ids is None:
            position_ids = torch.arange(self.max_seq_len + self.context_len, dtype=torch.long, device=input_ids.device)
            posids_len = self.max_seq_len + self.context_len

            if self.add_cap_labels:
                cap_posids = torch.arange(
                    self.cap_start_posid,
                    self.cap_start_posid + self.cap_len, dtype=torch.long, device=input_ids.device)
                position_ids = torch.cat([position_ids, cap_posids])
                posids_len += self.cap_len
            if self.add_ent_labels:
                ent_labels_posids = torch.arange(
                    self.ent_labels_start_posid,
                    self.ent_labels_start_posid + self.ent_labels_len, dtype=torch.long, device=input_ids.device)
                position_ids = torch.cat([position_ids, ent_labels_posids])
                posids_len += self.ent_labels_len

            position_ids = position_ids.unsqueeze(0).expand([batch_size, posids_len])

        cur_len = input_ids.shape[1]
        assert num_return_sequences == 1, 'not supported num_return_sequences != 1'
        effective_batch_size = batch_size

        self.turn_img_feats = self._expand_for_beams(turn_img_feats, num_beams, num_fsm_states)
        self.ent_img_feats = self._expand_for_beams(ent_img_feats, num_beams, num_fsm_states)
        self.full_attention_mask = self._expand_for_beams(attention_mask, num_beams, num_fsm_states)
        self.full_masked_pos = self._expand_for_beams(masked_pos, num_beams, num_fsm_states)
        self.full_token_type_ids = self._expand_for_beams(token_type_ids, num_beams, num_fsm_states)
        self.full_position_ids = self._expand_for_beams(position_ids, num_beams, num_fsm_states)
        self.full_head_mask = self._expand_for_beams(head_mask, num_beams, num_fsm_states)

        if not use_cbs:
            if num_beams > 1:
                # print('generate beam search.')
                output = self._generate_beam_search(
                    input_ids,
                    cur_len,
                    max_length,
                    do_sample,
                    temperature,
                    top_k,
                    top_p,
                    repetition_penalty,
                    pad_token_id,
                    eos_token_ids,
                    effective_batch_size,
                    length_penalty,
                    num_beams,
                    vocab_size,
                )
            else:
                # print('generate greedy search.')
                output = self._generate_no_beam_search(
                    input_ids,
                    cur_len,
                    max_length,
                    do_sample,
                    temperature,
                    top_k,
                    top_p,
                    repetition_penalty,
                    pad_token_id,
                    eos_token_ids,
                    effective_batch_size,
                    gth_ids,
                )
        else:
            assert self.num_keep_best == 1, 'not supported n_best > 1 for CBS'
            searcher = ConstrainedBeamSearch(eos_token_ids, max_length,
                                             num_beams, use_hypo=use_hypo)
            curr_ids, sum_logprobs = searcher.search(
                input_ids,
                None,
                self._decode_step,
                fsm,
            )
            curr_ids, sum_logprobs = select_best_beam_with_constraints(
                curr_ids,
                sum_logprobs,
                num_constraints,
                min_constraints_to_satisfy,
            )
            # (batch_size, n_best, max_len), (batch_size, n_best)
            output = (curr_ids.unsqueeze(1), sum_logprobs.unsqueeze(1))

        return output

    def MLM_loss_with_Hungarian_Matching(self, src_logits, targets, indices):

        # src_logits shape: [b_s* tag_leng, vocab_size]
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t[J] for t, (_, J) in zip(targets, indices)])

        target_classes = torch.full(targets.shape[:2], self.vocab_size - 1,
                                    dtype=torch.int64, device=targets.device)

        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits, target_classes.view(-1))
        return loss_ce