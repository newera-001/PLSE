from turtle import forward
import numpy as np
from rsl_rl.modules.actor_critic import get_activation

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from torch.nn.modules.activation import ReLU
from torch.nn.utils.parametrizations import spectral_norm

class Estimator(nn.Module):
    def __init__(self,  input_dim,
                        output_dim,
                        hidden_dims=[256, 128, 64],
                        activation="elu",
                        **kwargs):
        super(Estimator, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        activation = get_activation(activation)
        estimator_layers = []
        estimator_layers.append(nn.Linear(self.input_dim, hidden_dims[0]))
        estimator_layers.append(activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                estimator_layers.append(nn.Linear(hidden_dims[l], output_dim))
            else:
                estimator_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                estimator_layers.append(activation)
        # estimator_layers.append(nn.Tanh())
        self.estimator = nn.Sequential(*estimator_layers)
    
    def forward(self, input):
        return self.estimator(input)
    
    def inference(self, input):
        with torch.no_grad():
            return self.estimator(input)

class Discriminator(nn.Module):
    def __init__(self, n_states, 
                 n_skills, 
                 hidden_dims=[256, 128, 64], 
                 activation="elu"):
        super(Discriminator, self).__init__()
        self.n_states = n_states
        self.n_skills = n_skills

        activation = get_activation(activation)
        discriminator_layers = []
        discriminator_layers.append(nn.Linear(n_states, hidden_dims[0]))
        discriminator_layers.append(activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                discriminator_layers.append(nn.Linear(hidden_dims[l], n_skills))
            else:
                discriminator_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                discriminator_layers.append(activation)
        self.discriminator = nn.Sequential(*discriminator_layers)
        # self.hidden1 = nn.Linear(in_features=self.n_states, out_features=self.n_hidden_filters)
        # init_weight(self.hidden1)
        # self.hidden1.bias.data.zero_()
        # self.hidden2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        # init_weight(self.hidden2)
        # self.hidden2.bias.data.zero_()
        # self.q = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_skills)
        # init_weight(self.q, initializer="xavier uniform")
        # self.q.bias.data.zero_()

    def forward(self, states):
        return self.discriminator(states)

    def inference(self, states):
        with torch.no_grad():
            return self.discriminator(states)

class DiscriminatorLSD(nn.Module):
    def __init__(self, n_states, 
                 n_skills, 
                 hidden_dims=[256, 128, 64], 
                 activation="elu"):
        super(DiscriminatorLSD, self).__init__()
        self.n_states = n_states
        self.n_skills = n_skills

        activation = get_activation(activation)
        discriminator_layers = []
        discriminator_layers.append(spectral_norm(nn.Linear(n_states, hidden_dims[0])))
        discriminator_layers.append(activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                discriminator_layers.append(spectral_norm(nn.Linear(hidden_dims[l], n_skills)))
            else:
                discriminator_layers.append(spectral_norm(nn.Linear(hidden_dims[l], hidden_dims[l + 1])))
                discriminator_layers.append(activation)
        self.discriminator = nn.Sequential(*discriminator_layers)
        

    def forward(self, states):
        return self.discriminator(states)

    def inference(self, states):
        with torch.no_grad():
            return self.discriminator(states)
        
class DiscriminatorContDIAYN(nn.Module):
    def __init__(self, n_states, 
                 latent_z,
                 num_skills,
                 hidden_dims=[256, 128, 64], 
                 activation="elu"):
        super(DiscriminatorContDIAYN, self).__init__()
        self.n_states = n_states
        self.latent_z = latent_z
        self.control_c = num_skills

        activation = get_activation(activation)
        discriminator_layers = []
        discriminator_layers.append(nn.Linear(self.n_states, hidden_dims[0]))
        discriminator_layers.append(activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                discriminator_layers.append(nn.Linear(hidden_dims[l], self.latent_z))
            else:
                discriminator_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                discriminator_layers.append(activation)
        self.latent = nn.Sequential(*discriminator_layers)

        self.pre_condition = nn.Sequential(nn.ReLU(),nn.Linear(self.latent_z, 128),nn.ReLU(),nn.Linear(128, self.control_c))
        self.self_denial = nn.Sequential(nn.ReLU(),nn.Linear(self.latent_z, 2))

    def forward(self, states):
        z = self.latent(states)
        return self.pre_condition(z), self.self_denial(z)

    def condition_inference(self, states):
        with torch.no_grad():
            z = self.latent(states)
            return self.pre_condition(z)

    def self_denial_inference(self, states):
        with torch.no_grad():
            z = self.latent(states)
            return self.self_denial(z)

    def get_condition_latent(self, states):
        z = self.latent(states)
        return self.pre_condition(z), z

class DiscriminatorConv(nn.Module):
    def __init__(self, input_size, tsteps=10, latent_z=128, num_skills=1,tanh_encoder_output=False):
        # self.device = device
        super(DiscriminatorConv, self).__init__()
        self.activation_fn = nn.ReLU()
        self.tsteps = tsteps
        self.latent_z = latent_z
        self.num_skills = num_skills

        channel_size = 10
        # last_activation = nn.ELU()

        self.encoder = nn.Sequential(
                nn.Linear(input_size, 3 * channel_size), self.activation_fn,
                )

        if self.tsteps == 50:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels=3 * channel_size, out_channels=2 * channel_size, kernel_size=8, stride=4),
                self.activation_fn,
                nn.Conv1d(in_channels=2 * channel_size, out_channels=channel_size, kernel_size=5, stride=1),
                self.activation_fn,
                nn.Conv1d(in_channels=channel_size, out_channels=channel_size, kernel_size=5, stride=1),
                self.activation_fn, nn.Flatten())
        elif self.tsteps == 10:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels=3 * channel_size, out_channels=2 * channel_size, kernel_size=4, stride=2),
                self.activation_fn,
                nn.Conv1d(in_channels=2 * channel_size, out_channels=channel_size, kernel_size=2, stride=1),
                self.activation_fn,
                nn.Flatten())
        elif self.tsteps == 20:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels=3 * channel_size, out_channels=2 * channel_size, kernel_size=6, stride=2),
                self.activation_fn,
                nn.Conv1d(in_channels=2 * channel_size, out_channels=channel_size, kernel_size=4, stride=2),
                self.activation_fn,
                nn.Flatten())
        else:
            raise (ValueError("tsteps must be 10, 20 or 50"))

        self.linear_output = nn.Sequential(
                nn.Linear(channel_size * 3, self.latent_z), self.activation_fn
                )
        

        self.pre_condition = nn.Sequential(nn.Linear(self.latent_z, self.num_skills))


    def forward(self, obs):
        # nd * T * n_proprio
        nd = obs.shape[0]
        T = self.tsteps
        # print("obs device", obs.device)
        # print("encoder device", next(self.encoder.parameters()).device)
        projection = self.encoder(obs.reshape([nd * T, -1])) # do projection for n_proprio -> 32
        output = self.conv_layers(projection.reshape([nd, T, -1]).permute((0, 2, 1)))
        output = self.linear_output(output)
        return self.pre_condition(output)

    def get_state_latent_and_prediction(self, obs):
        nd = obs.shape[0]
        T = self.tsteps
        # print("obs device", obs.device)
        # print("encoder device", next(self.encoder.parameters()).device)
        projection = self.encoder(obs.reshape([nd * T, -1])) # do projection for n_proprio -> 32
        output = self.conv_layers(projection.reshape([nd, T, -1]).permute((0, 2, 1)))
        output = self.linear_output(output)
        return output, self.pre_condition(output)


    def set_new_prediction_head(self, all_num=1):
        #TODO May we can initialize new prediction head using old prediction in some ways
        self.pre_condition = nn.Sequential(nn.Linear(self.latent_z, all_num)).to('cuda')



class ExplorationValueEstimator(nn.Module):
    def __init__(self, states_dim, action_dim=12, latent_dim=16,
                 hidden_dims=[256, 128, 64],
                 activation="relu",
                 **kwargs):
        super(ExplorationValueEstimator, self).__init__()

        activation = get_activation(activation)
        estimator_layers = []
        estimator_layers.append(spectral_norm(nn.Linear(states_dim, hidden_dims[0])))
        estimator_layers.append(activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                estimator_layers.append(spectral_norm(nn.Linear(hidden_dims[l], 1)))
            else:
                estimator_layers.append(spectral_norm(nn.Linear(hidden_dims[l], hidden_dims[l + 1])))
                estimator_layers.append(activation)
        self.value_estimator = nn.Sequential(*estimator_layers)

        self.shared_feature = nn.Sequential(nn.Linear(states_dim, 256),
                                           activation,
                                           nn.Linear(256, 64),
                                           activation,
                                           nn.Linear(64, latent_dim))
        self.forward_model = nn.Sequential(nn.Linear(latent_dim+action_dim, 128),
                                          activation,
                                          nn.Linear(128, 64),
                                          activation,
                                          nn.Linear(64, latent_dim))
        self.inverse_model = nn.Sequential(nn.Linear(latent_dim*2, 128),
                                          activation,
                                          nn.Linear(128, 32),
                                          activation,
                                          nn.Linear(32, action_dim))

    def forward(self, latent_state):
        return self.value_estimator(latent_state)

    def reward_forward(self, s_c, s_n, a_c):
        s_c_latent = self.shared_feature(s_c)
        s_n_latent = self.shared_feature(s_n)
        s_n_pre = self.forward_model(torch.cat([s_c_latent, a_c],dim=-1))
        a_c_pre = self.inverse_model(torch.cat([s_c_latent, s_n_latent], dim=-1))
        return (s_n_pre-s_n_latent).norm(p=2, dim=-1), (a_c_pre-a_c).norm(p=2, dim=-1)

    def get_curiosity_reward(self, s_c, s_n, a_c):
        s_c_latent = self.shared_feature(s_c)
        s_n_latent = self.shared_feature(s_n)
        s_n_pre = self.forward_model(torch.cat([s_c_latent, a_c],dim=-1))
        # return torch.exp(-1(s_n_pre-s_n_latent).norm(p=2, dim=-1))
        return (s_n_pre-s_n_latent).norm(p=2, dim=-1) * 0.5

    def latent_dynamics(self, s_c):
        return self.shared_feature(s_c)
        

class InternalMotivationEstimator(nn.Module):
    def __init__(self, obs_dim=132, latent_dim=16, gait_dim=4, device="cuda:0"):
        super(InternalMotivationEstimator, self).__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.gait_dim= gait_dim
        self.noise = torch.randn
        self.device = device

        self.motivation = nn.Sequential(
            nn.Linear(self.obs_dim, 512),
            nn.ELU(),
            nn.Linear(512, 64),
            nn.ELU(),
            nn.Linear(64, self.gait_dim)
        )

    def forward(self, scan_latent):
        return self.motivation(scan_latent) + self.noise(self.gait_dim).to(self.device)

