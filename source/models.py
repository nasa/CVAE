import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
cuda = torch.cuda.is_available()


class Stochastic(nn.Module):
    """
    performs the reparameterizaton trick
    """
    
    def reparametrize(self, mu, log_var):
        
        epsilon = Variable(torch.randn(mu.size()), requires_grad=False)

        if mu.is_cuda:
            epsilon = epsilon.cuda()

        std = log_var.mul(0.5).exp_()
        z = mu.addcmul(std, epsilon)

        return z


class GaussianSample(Stochastic):
    """
    performs the sampling from the Gaussian latent space
    """
    
    def __init__(self, input_dim, output_dim):
        
        super(GaussianSample, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.mu = nn.Linear(input_dim, output_dim)
        self.log_var = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        mu = self.mu(x)
        log_var = F.softplus(self.log_var(x))
        z_sample = self.reparametrize(mu, log_var)
        return z_sample, mu, log_var    


class Encoder(nn.Module):
    """
    Encoder model with CNN architecture
    """
    
    def __init__(self, latent_dim, num_param, window_size, sample_layer=GaussianSample,
                 filter_1=8, filter_2=16, filter_3=32, filter_4=64):
        
        super(Encoder, self).__init__()
        self.num_branch = 3
        self.conv1_1 = nn.Conv1d(num_param, filter_1, 3, padding=1)
        self.conv1_2 = nn.Conv1d(num_param, filter_1, 5, padding=2)
        self.conv1_3 = nn.Conv1d(num_param, filter_1, 7, padding=3)
        self.bn1 = nn.BatchNorm1d(filter_1, track_running_stats=False)
        self.pool = nn.MaxPool1d(2)
        self.conv2_1 = nn.Conv1d(filter_1, filter_2, 3, padding=1)
        self.conv2_2 = nn.Conv1d(filter_1, filter_2, 5, padding=2)
        self.conv2_3 = nn.Conv1d(filter_1, filter_2, 7, padding=3)
        self.bn2 = nn.BatchNorm1d(filter_2, track_running_stats=False)
        self.conv3_1 = nn.Conv1d(filter_2, filter_3, 3, padding=1)
        self.conv3_2 = nn.Conv1d(filter_2, filter_3, 5, padding=2)
        self.conv3_3 = nn.Conv1d(filter_2, filter_3, 7, padding=3)
        self.bn3 = nn.BatchNorm1d(filter_3, track_running_stats=False)
        self.conv4_1 = nn.Conv1d(filter_3, filter_4, 3, padding=1)
        self.conv4_2 = nn.Conv1d(filter_3, filter_4, 5, padding=2)
        self.conv4_3 = nn.Conv1d(filter_3, filter_4, 7, padding=3)
        self.bn4 = nn.BatchNorm1d(filter_4, track_running_stats=False)
        self.conv5 = nn.Conv1d(filter_4, filter_4, int(np.floor(window_size/16)))
        self.bn5 = nn.BatchNorm1d(filter_4, track_running_stats=False)
        self.flat = nn.Flatten()
        self.sample = sample_layer(int(filter_4*self.num_branch), latent_dim)
        
    def forward(self, x):
        
        x1 = self.pool(F.relu(self.bn1(self.conv1_1(x))))
        x1 = self.pool(F.relu(self.bn2(self.conv2_1(x1))))
        x1 = self.pool(F.relu(self.bn3(self.conv3_1(x1))))
        x1 = self.pool(F.relu(self.bn4(self.conv4_1(x1))))
        x1 = F.relu(self.bn5(self.conv5(x1)))
        x1 = self.flat(x1)
        
        x2 = self.pool(F.relu(self.bn1(self.conv1_2(x))))
        x2 = self.pool(F.relu(self.bn2(self.conv2_2(x2))))
        x2 = self.pool(F.relu(self.bn3(self.conv3_2(x2))))
        x2 = self.pool(F.relu(self.bn4(self.conv4_2(x2))))
        x2 = F.relu(self.bn5(self.conv5(x2)))
        x2 = self.flat(x2)
        
        x3 = self.pool(F.relu(self.bn1(self.conv1_3(x))))
        x3 = self.pool(F.relu(self.bn2(self.conv2_3(x3))))
        x3 = self.pool(F.relu(self.bn3(self.conv3_3(x3))))
        x3 = self.pool(F.relu(self.bn4(self.conv4_3(x3))))
        x3 = F.relu(self.bn5(self.conv5(x3)))
        x3 = self.flat(x3)
        
        x_out = torch.cat((x1, x2, x3), dim=1)
        
        x_out = self.sample(x_out)
        
        return x_out


class Decoder(nn.Module):
    """
    Decoder model with CNN architecture
    """
    def __init__(self, latent_dim, num_param, window_size, scale_flag,
                 filter_1=8, filter_2=16, filter_3=32, filter_4=64):
        
        super(Decoder, self).__init__()
        
        self.num_param = num_param
        self.window_size = window_size
        self.filter_4 = filter_4
        self.fc = nn.Linear(latent_dim, filter_4)
        self.deconv1_1 = nn.ConvTranspose2d(filter_4, filter_3, (num_param, int(np.floor(window_size/16))))
        self.bn1 = nn.BatchNorm2d(filter_3, track_running_stats=False)
        self.up1 = nn.Upsample(size=(num_param, int(np.floor(window_size/8))),
                               mode='bilinear')
        self.deconv2_1 = nn.ConvTranspose2d(filter_3, filter_2, (1, 3), padding=(0, 1))
        self.deconv2_2 = nn.ConvTranspose2d(filter_3, filter_2, (1, 5), padding=(0, 2))
        self.deconv2_3 = nn.ConvTranspose2d(filter_3, filter_2, (1, 7), padding=(0, 3))
        self.bn2 = nn.BatchNorm2d(filter_2, track_running_stats=False)
        self.up2 = nn.Upsample(size=(num_param, int(np.floor(window_size/4))),
                               mode='bilinear')
        self.deconv3_1 = nn.ConvTranspose2d(filter_2, filter_1, (1, 3), padding=(0, 1))
        self.deconv3_2 = nn.ConvTranspose2d(filter_2, filter_1, (1, 5), padding=(0, 2))
        self.deconv3_3 = nn.ConvTranspose2d(filter_2, filter_1, (1, 7), padding=(0, 3))
        self.bn3 = nn.BatchNorm2d(filter_1, track_running_stats=False)
        self.up3 = nn.Upsample(size=(num_param, int(np.floor(window_size/2))),
                               mode='bilinear')
        self.deconv4_1 = nn.ConvTranspose2d(filter_1, 1, (1, 3), padding=(0, 1))
        self.deconv4_2 = nn.ConvTranspose2d(filter_1, 1, (1, 5), padding=(0, 2))
        self.deconv4_3 = nn.ConvTranspose2d(filter_1, 1, (1, 7), padding=(0, 3))
        self.bn4 = nn.BatchNorm2d(1, track_running_stats=False)
        self.up4 = nn.Upsample(size=(num_param, window_size),
                               mode='bilinear')
        self.convlast = nn.Conv2d(3, 1, (1, 1))
        if scale_flag == 1:
            self.output_activation = nn.Identity()
        elif scale_flag == 0:
            self.output_activation = nn.Sigmoid()
        
    def forward(self, x):
        
        x1 = F.relu(self.fc(x))
        x1 = torch.reshape(x1, (-1, self.filter_4, 1, 1))
        x1 = self.up1(F.relu(self.bn1(self.deconv1_1(x1))))
        x1 = self.up2(F.relu(self.bn2(self.deconv2_1(x1))))
        x1 = self.up3(F.relu(self.bn3(self.deconv3_1(x1))))
        x1 = self.up4(F.relu(self.bn4(self.deconv4_1(x1))))
        
        x2 = F.relu(self.fc(x))
        x2 = torch.reshape(x2, (-1, self.filter_4, 1, 1))
        x2 = self.up1(F.relu(self.bn1(self.deconv1_1(x2))))
        x2 = self.up2(F.relu(self.bn2(self.deconv2_2(x2))))
        x2 = self.up3(F.relu(self.bn3(self.deconv3_2(x2))))
        x2 = self.up4(F.relu(self.bn4(self.deconv4_2(x2))))
        
        x3 = F.relu(self.fc(x))
        x3 = torch.reshape(x3, (-1, self.filter_4, 1, 1))
        x3 = self.up1(F.relu(self.bn1(self.deconv1_1(x3))))
        x3 = self.up2(F.relu(self.bn2(self.deconv2_3(x3))))
        x3 = self.up3(F.relu(self.bn3(self.deconv3_3(x3))))
        x3 = self.up4(F.relu(self.bn4(self.deconv4_3(x3))))
        
        x_out = torch.cat((x1, x2, x3), dim=1)
        
        x_out = torch.reshape(self.output_activation(self.convlast(x_out)), (-1, self.num_param, self.window_size))

        return x_out


class VAE(nn.Module):
    """
    VAE model with CNN architecture
    """
    def __init__(self, latent_dim, num_param, window_size, scale_flag):
        
        super(VAE, self).__init__()
        
        self.z_dim = latent_dim
        self.p_dim = num_param
        self.t_dim = window_size
        self.scale_flag = scale_flag
        self.encoder = Encoder(self.z_dim, self.p_dim, self.t_dim)
        self.decoder = Decoder(self.z_dim, self.p_dim, self.t_dim, self.scale_flag)
        self.kl_div = 0
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def kld_(self, q_param): 
        (mu, log_var) = q_param
        kl = 0.5 * torch.sum(-1 - log_var + torch.pow(mu, 2) + torch.exp(log_var), dim=-1)
        return kl
    
    
    def forward(self, x):
        z, z_mu, z_log_var = self.encoder(x)
        self.kl_div = self.kld_((z_mu, z_log_var))
        x_rec = self.decoder(z) 
        return x_rec
    
    def sample(self, z):
        x_rec = self.decoder(z)   
        return x_rec