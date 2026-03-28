# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np

import code
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from torch.nn.modules.activation import ReLU
import math
import torch.nn.functional as F

class StateHistoryEncoder(nn.Module):
    def __init__(self, activation_fn, input_size, tsteps, output_size, tanh_encoder_output=False):
        # self.device = device
        super(StateHistoryEncoder, self).__init__()
        self.activation_fn = activation_fn
        self.tsteps = tsteps

        channel_size = 10
        # last_activation = nn.ELU()

        self.encoder = nn.Sequential(
                nn.Linear(input_size, 3 * channel_size), self.activation_fn,
                )

        if tsteps == 50:
            self.conv_layers = nn.Sequential(
                    nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 8, stride = 4), self.activation_fn,
                    nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn,
                    nn.Conv1d(in_channels = channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn, nn.Flatten())
        elif tsteps == 10:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 4, stride = 2), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 2, stride = 1), self.activation_fn,
                nn.Flatten())
        elif tsteps == 20:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 6, stride = 2), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 4, stride = 2), self.activation_fn,
                nn.Flatten())
        else:
            raise(ValueError("tsteps must be 10, 20 or 50"))

        self.linear_output = nn.Sequential(
                nn.Linear(channel_size * 3, output_size), self.activation_fn
                )

    def forward(self, obs):
        # nd * T * n_proprio
        nd = obs.shape[0]
        T = self.tsteps
        # print("obs device", obs.device)
        # print("encoder device", next(self.encoder.parameters()).device)
        projection = self.encoder(obs.reshape([nd * T, -1])) # do projection for n_proprio -> 32
        output = self.conv_layers(projection.reshape([nd, T, -1]).permute((0, 2, 1)))
        output = self.linear_output(output)
        return output

class Actor(nn.Module):
    def __init__(self, num_prop, 
                 num_scan, 
                 num_actions, 
                 scan_encoder_dims,
                 actor_hidden_dims, 
                 priv_encoder_dims, 
                 num_priv_latent, 
                 num_priv_explicit, 
                 num_hist, activation,
                 use_transformer=True,
                 tanh_encoder_output=False) -> None:
        super().__init__()
        # prop -> scan -> priv_explicit -> priv_latent -> hist
        # actor input: prop -> scan -> priv_explicit -> latent
        self.num_prop = num_prop
        self.num_scan = num_scan
        self.num_hist = num_hist
        self.num_actions = num_actions
        self.num_priv_latent = num_priv_latent
        self.num_priv_explicit = num_priv_explicit
        self.if_scan_encode = scan_encoder_dims is not None and num_scan > 0

        if len(priv_encoder_dims) > 0:
                    priv_encoder_layers = []
                    priv_encoder_layers.append(nn.Linear(num_priv_latent, priv_encoder_dims[0]))
                    priv_encoder_layers.append(activation)
                    for l in range(len(priv_encoder_dims) - 1):
                        priv_encoder_layers.append(nn.Linear(priv_encoder_dims[l], priv_encoder_dims[l + 1]))
                        priv_encoder_layers.append(activation)
                    self.priv_encoder = nn.Sequential(*priv_encoder_layers)
                    priv_encoder_output_dim = priv_encoder_dims[-1]
        else:
            self.priv_encoder = nn.Identity()
            priv_encoder_output_dim = num_priv_latent

        self.history_encoder = StateHistoryEncoder(activation, num_prop-24, num_hist, priv_encoder_output_dim)

        if self.if_scan_encode:
            scan_encoder = []
            scan_encoder.append(nn.Linear(num_scan, scan_encoder_dims[0]))
            scan_encoder.append(activation)
            for l in range(len(scan_encoder_dims) - 1):
                if l == len(scan_encoder_dims) - 2:
                    scan_encoder.append(nn.Linear(scan_encoder_dims[l], scan_encoder_dims[l+1]))
                    scan_encoder.append(nn.Tanh())
                else:
                    scan_encoder.append(nn.Linear(scan_encoder_dims[l], scan_encoder_dims[l + 1]))
                    scan_encoder.append(activation)
            self.scan_encoder = nn.Sequential(*scan_encoder)
            self.scan_encoder_output_dim = scan_encoder_dims[-1]
        else:
            self.scan_encoder = nn.Identity()
            self.scan_encoder_output_dim = num_scan
        
        actor_layers = []
        actor_layers.append(nn.Linear(num_prop+
                                      self.scan_encoder_output_dim+
                                      num_priv_explicit+
                                      priv_encoder_output_dim, 
                                      actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        if tanh_encoder_output:
            actor_layers.append(nn.Tanh())
        if use_transformer:
            self.actor_backbone = Transformer(num_prop=self.num_prop,
                                              num_priv=num_priv_explicit + priv_encoder_output_dim)
        else:
            self.actor_backbone = nn.Sequential(*actor_layers)
        self.use_transformer = use_transformer

    def forward(self, obs, hist_encoding: bool, eval=False, scandots_latent=None):
        if not eval:
            if self.if_scan_encode:
                obs_scan = obs[:, self.num_prop:self.num_prop + self.num_scan]
                if scandots_latent is None:
                    scan_latent = self.scan_encoder(obs_scan)   
                else:
                    scan_latent = scandots_latent
                obs_prop_scan = torch.cat([obs[:, :self.num_prop], scan_latent], dim=1)
            else:
                obs_prop_scan = obs[:, :self.num_prop + self.num_scan]
            obs_priv_explicit = obs[:, self.num_prop + self.num_scan:self.num_prop + self.num_scan + self.num_priv_explicit]
            if hist_encoding:
                latent = self.infer_hist_latent(obs)
            else:
                latent = self.infer_priv_latent(obs)
            if self.use_transformer:
                backbone_output = self.actor_backbone.get_action_and_value(obs_prop_scan,
                                                                           torch.cat((obs_priv_explicit, latent),
                                                                                     dim=-1))
            else:
                backbone_input = torch.cat([obs_prop_scan, obs_priv_explicit, latent], dim=1)
                backbone_output = self.actor_backbone(backbone_input)
            return backbone_output
        else:
            if self.if_scan_encode:
                obs_scan = obs[:, self.num_prop:self.num_prop + self.num_scan]
                if scandots_latent is None:
                    scan_latent = self.scan_encoder(obs_scan)   
                else:
                    scan_latent = scandots_latent
                obs_prop_scan = torch.cat([obs[:, :self.num_prop], scan_latent], dim=1)
            else:
                obs_prop_scan = obs[:, :self.num_prop + self.num_scan]
            obs_priv_explicit = obs[:, self.num_prop + self.num_scan:self.num_prop + self.num_scan + self.num_priv_explicit]
            if hist_encoding:
                latent = self.infer_hist_latent(obs)
            else:
                latent = self.infer_priv_latent(obs)
            if self.use_transformer:
                backbone_output = self.actor_backbone.get_action_and_value(obs_prop_scan,
                                                                           torch.cat((obs_priv_explicit, latent),
                                                                                     dim=-1))
            else:
                backbone_input = torch.cat([obs_prop_scan, obs_priv_explicit, latent], dim=1)
                backbone_output = self.actor_backbone(backbone_input)
            return backbone_output
    
    def infer_priv_latent(self, obs):
        priv = obs[:, self.num_prop + self.num_scan + self.num_priv_explicit: self.num_prop + self.num_scan + self.num_priv_explicit + self.num_priv_latent]
        return self.priv_encoder(priv)
    
    def infer_hist_latent(self, obs):
        hist = obs[:, -self.num_hist*self.num_prop:]
        return self.history_encoder(hist.view(-1, self.num_hist, self.num_prop)[:,:,:-24])
    
    def infer_scandots_latent(self, obs):
        scan = obs[:, self.num_prop:self.num_prop + self.num_scan]
        return self.scan_encoder(scan)

class ActorCriticRMA(nn.Module):
    is_recurrent = False
    def __init__(self,  num_prop,
                        num_scan,
                        num_critic_obs,
                        num_priv_latent, 
                        num_priv_explicit,
                        num_hist,
                        num_actions,
                        scan_encoder_dims=[256, 256, 256],
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCriticRMA, self).__init__()

        self.kwargs = kwargs
        priv_encoder_dims= kwargs['priv_encoder_dims']
        activation = get_activation(activation)

        self.actor = Actor(num_prop, num_scan, num_actions, scan_encoder_dims, actor_hidden_dims, priv_encoder_dims, num_priv_latent, num_priv_explicit, num_hist, activation, tanh_encoder_output=kwargs['tanh_encoder_output'])
        

        # Value function
        critic_layers = []
        if num_critic_obs > 962:
            num_critic_obs=962
        critic_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)
    
    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations, hist_encoding):
        mean = self.actor(observations, hist_encoding)
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, observations, hist_encoding=False, **kwargs):
        self.update_distribution(observations, hist_encoding)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, hist_encoding=False, eval=False, scandots_latent=None, **kwargs):
        if not eval:
            actions_mean = self.actor(observations, hist_encoding, eval, scandots_latent)
            return actions_mean
        else:
            actions_mean, latent_hist, latent_priv = self.actor(observations, hist_encoding, eval=True)
            return actions_mean, latent_hist, latent_priv

    def evaluate(self, critic_observations, **kwargs):
        if self.actor.use_transformer:
            value = self.actor.actor_backbone.value
        else:
            value = self.critic(critic_observations)
        return value
    
    def get_value_head_intpu(self):
        return self.actor.actor_backbone.value_head_input

    def reset_std(self, std, num_actions, device):
        new_std = std * torch.ones(num_actions, device=device)
        self.std.data = new_std.data


class SelfAttention(nn.Module):

    def __init__(self, n_embd, n_head, n_tokens, dropout=0., masked=False):
        super(SelfAttention, self).__init__()

        assert n_embd % n_head == 0
        self.masked = masked
        self.n_head = n_head
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        # if self.masked:
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(n_tokens + 1, n_tokens + 1))
                             .view(1, 1, n_tokens + 1, n_tokens + 1))

        self.att_bp = None

    def forward(self, key, value, query):
        B, L, D = query.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(key).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        q = self.query(query).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        v = self.value(value).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)

        # causal attention: (B, nh, L, hs) x (B, nh, hs, L) -> (B, nh, L, L)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if self.masked:
            att = att.masked_fill(self.mask[:, :, :L, :L] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, nh, L, L) x (B, nh, L, hs) -> (B, nh, L, hs)
        y = y.transpose(1, 2).contiguous().view(B, L, D)  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)
        return y


class EncodeBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, n_tokens=0, dropout=0.):
        super(EncodeBlock, self).__init__()

        # self.attn = SelfAttention(n_embd, n_head, n_tokens, masked=True)
        self.attn = SelfAttention(n_embd, n_head, n_tokens, dropout, masked=False)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 1 * n_embd),
            nn.GELU()
        )

    def forward(self, x):
        x = x + self.attn(x, x, x)
        x = x + self.mlp(x)
        return x


class Transformer(nn.Module):

    def __init__(self, num_prop, num_priv,leg_size=18, n_embd=160, n_head=2, n_blocks=2, use_skill=True):
        super(Transformer, self).__init__()
        self.leg_embd = nn.Linear(leg_size, n_embd)
        self.obs_encoder = nn.Sequential()
        # self.pos_embed = nn.Parameter(torch.normal(mean=0, std=1., size=((1, 4, n_embd))))
        self.pos_embed = nn.Sequential(nn.Linear(6, n_embd))
        # for name, param in self.pos_embed.named_parameters():
        #         param.requires_grad = False
        self.blocks1 = EncodeBlock(n_embd, n_head)
        self.policy_head = nn.Sequential(nn.Linear(n_embd+num_priv , n_embd), nn.ELU(), nn.Linear(n_embd, 3))
        self.value_head = nn.Sequential(nn.Linear(n_embd +num_priv, n_embd), nn.ELU(), nn.Linear(n_embd, 1))
        self.num_prop = num_prop
        self.num_priv = num_priv
        self.use_skill = use_skill
        self.target_gait_phase = None
        # self.skillpolicy = SkillPolicy()

    def forward(self):
        pass

    def get_action_and_value(self, obs_prop_depth, hist_token):
        token_embd, vision_token, gait_obs = self.tokenize_obs(obs_prop_depth)
        x = token_embd
        x = self.obs_encoder(x)
        x = torch.cat((token_embd,vision_token.unsqueeze(1)*0.2),dim=1)
        x = self.blocks1(x)
        x = x[:,:4]
        hist_token = hist_token.unsqueeze(1)
        hist_token = torch.repeat_interleave(hist_token, 4, 1)
        x = torch.cat((x, hist_token*0.2), dim=-1)
        self.value_head_input = x.detach()
        value = self.value_head(x)
        value = torch.mean(value, dim=1)
        action = self.policy_head(x)
        action = action.reshape(-1, 12)
        action[:,[7.10]]*= 1.25
        action[:,[3,9]] *= -1
        self.value = value
        return action

    def tokenize_obs(self, obs_prop_scan):
        obs_prop = obs_prop_scan[:,:self.num_prop]
        obs_scan = obs_prop_scan[:,self.num_prop:]
        # skill_flag = obs_prop[:, -25].unsqueeze(-1)
        gait_obs = obs_prop[:, -24:].reshape(obs_prop.shape[0], 4, 6)
        # self.target_gait_phase = gait_obs.max(dim=-1).values
        leg_tokens_ = [torch.cat(
            (obs_prop[:, :8], obs_prop[:, 8 + 3 * i:11 + 3 * i], obs_prop[:, 20 + 3 * i:23 + 3 * i],
             obs_prop[:, 32 + 3 * i:35 + 3 * i],obs_prop[:,44+i].unsqueeze(-1)),
            dim=-1) for i in range(4)]
        leg_tokens = torch.stack((leg_tokens_[0], leg_tokens_[1], leg_tokens_[2], leg_tokens_[3],), dim=1)
        leg_tokens = self.leg_embd(leg_tokens) + self.pos_embed(gait_obs)
        return leg_tokens, obs_scan, gait_obs

class SkillPolicy(nn.Module):

    def __init__(self, input_size=189, downsample_size=192, hist_size=72*9,n_embd=256):
        super(SkillPolicy, self).__init__()
        # self.vision_embd = nn.Sequential(nn.Linear(downsample_size,n_embd),nn.ELU(),nn.Linear(n_embd,n_embd//2))
        # self.hist_embd = nn.Sequential(nn.Linear(hist_size, n_embd),nn.ELU(),nn.Linear(n_embd,n_embd//2))
        self.policy = nn.Sequential(nn.Linear(input_size, n_embd),nn.ELU(),nn.Linear(n_embd,n_embd//2),nn.ELU(),nn.Linear(n_embd//2, 3))

    def forward(self):
        raise NotImplementedError

    def get_skill(self,vision_embd, hist_embd):
        # downsample
        # height_obs = height_obs.reshape(-1, 24, 32)
        # height_obs = height_obs[:,::2,::2]
        # vision_embd = self.vision_embd(height_obs)
        # hist_embd = self.hist_embd(hist_obs)
        latent = torch.cat((vision_embd, hist_embd),dim=-1)
        skill = self.policy(latent)
        skill = F.softmax(skill, dim=-1)
        logskill = F.log_softmax(skill, dim=-1)
        self.entropy = -skill*logskill
        # skill = skill.flip(-1)
        skill = torch.argmax(skill, dim=-1) + 1
        # print(skill)
        skill = skill.unsqueeze(-1)
        return skill

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
