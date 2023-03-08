import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

####################################Simple Score####################################

class Score(nn.Module):

    def __init__(self,n):
        nodes = [64,64,64]
        super(Score, self).__init__()
        self.first_layer = nn.Linear(n+1, nodes[0])
        self.second_layer = nn.Linear(nodes[0], nodes[1])
        self.third_layer = nn.Linear(nodes[1], nodes[2])
        self.final_score = nn.Linear(nodes[2], n)

    def forward(self, x,t):
        x = torch.cat((x,t),dim=-1)
        x = x.float()
        
        x = F.logsigmoid(self.first_layer(x))
        x = F.logsigmoid(self.second_layer(x))
        x = F.logsigmoid(self.third_layer(x))
        x = self.final_score(x)
        return x


class SimpleScore(nn.Module):

    def __init__(self,n):
        nodes = [128,256,128]
        super(SimpleScore, self).__init__()
        self.first_layer = nn.Linear(n+1, nodes[0])
        self.second_layer = nn.Linear(nodes[0], nodes[1])
        self.third_layer = nn.Linear(nodes[1], nodes[2])
        self.final_score = nn.Linear(nodes[2], n)

    def forward(self, x,t):
        t = t.unsqueeze(-1)
        x = torch.cat((x,t),dim=-1)
        x = x.float()
        
        x = F.logsigmoid(self.first_layer(x))
        x = F.logsigmoid(self.second_layer(x))
        x = F.logsigmoid(self.third_layer(x))
        x = self.final_score(x)
        return x

####################################Auto-Encoder####################################

class Encoder(nn.Module):
    def __init__(self, n, num_layers=2):
        super().__init__()
        self.layers = []
        for i in range(num_layers):
            if i == 0:
                self.layers.append(nn.Linear(n, n-1))
            else:
                self.layers.append(nn.Linear(n-1, n-1))

    def forward(self, x):
        x = x.float()
        for layer in layers:
            x = layer(F.silu(x))
        return x


class Decoder(nn.Module):
    def __init__(self, n, num_layers=2):
        super().__init__()
        self.layers = []
        for i in range(num_layers):
            if i == num_layers-1:
                self.layers.append(nn.Linear(n-1, n))
            else:
                self.layers.append(nn.Linear(n-1, n-1))

    def forward(self, x):
        x = x.float()
        for layer in layers:
            x = layer(F.silu(x))
        return x


class AutoEncoder(nn.Module):

    def __init__(self, n, num_layers=2):
        super().__init__()
        self.E = Encoder(n, num_layers=num_layers)
        self.D = Decoder(n, num_layers=num_layers)

    def forward(self, x):
        x_hidden = self.E(x)
        x = self.D(x_hidden)
        return x


@dataclass
class LDM:
    score: Score
    autoencoder: AutoEncoder


####################################FNO Score####################################


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes, verbose=False):
        super(SpectralConv1d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.verbose = verbose

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, 2))
        #self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2))
        
    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bixt,ioxt->boxt", input, weights)

    # Downsampling by truncating fourier modes?
    def forward(self, x, out_dim):
        print("Spec conv input, weights: ", x.shape, self.weights1.shape) if self.verbose else None
        batchsize, in_channel, res = x.shape
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)
        print("fft output shape: ", x_ft.shape) if self.verbose else None
        print("out dim: ", out_dim) if self.verbose else None
        x_ft = torch.view_as_real(x_ft)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros((batchsize, self.out_channels, out_dim // 2 + 1, 2), device=x.device)
        # I guess we are implicitly taking the lowest and highest modes?
        out_ft[:, :, :self.modes1] = \
            self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        out_ft = torch.view_as_complex(out_ft)
        x = torch.fft.irfft(out_ft)
        print("ifft output shape: ", x.shape) if self.verbose else None
        return x


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, up=False, down=False, verbose=True):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.down = down
        self.up = up

        self.verbose = verbose

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2))
        
    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixyt,ioxyt->boxyt", input, weights)

    # Downsampling by truncating fourier modes?
    def forward(self, x):
        print("Spec conv input, weights: ", x.shape, self.weights1.shape) if self.verbose else None
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        # TODO: Are we actually discarding low frequency? Check this carefully
        x_ft = torch.fft.rfft2(x) # (B, C, H, W // 2 + 1)
        x_ft = torch.view_as_real(x_ft)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros((batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, 2), device=x.device)
        # I guess we are implicitly taking the lowest and highest modes?
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        out_ft = torch.view_as_complex(out_ft)
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class MLP(nn.Module):
    def __init__(self, n_layers, input_dim, hidden_dim, output_dim, verbose=False):
        super(MLP, self).__init__()
        layers = []
        for i in range(n_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            elif i == n_layers - 1:
                layers.append(nn.Linear(hidden_dim, output_dim))
            else:
                layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.ModuleList(layers)
        self.verbose = verbose
    
    def forward(self, x):
        print("MLP input: ", x.shape) if self.verbose else None
        for i, layer in enumerate(self.layers):
            if i == 0:
                x = layer(x)
            else:
                x = layer(F.silu(x))
        return x


class FNOScore(nn.Module):
    def __init__(self, n_layers, hidden_channels, hidden_dim, modes, time_embed_type, res_layer_type, verbose=False):
        super(FNOScore, self).__init__()

        self.n_layers = n_layers
        self.hidden_channels = hidden_channels
        self.hidden_dim = hidden_dim
        self.modes = modes
        self.verbose = verbose
        self.time_embed_type = time_embed_type
        self.res_layer_type = res_layer_type

        layers = []
        for i in range(n_layers):
            if i == 0:
                layers.append(SpectralConv1d(1, hidden_channels, modes, verbose=verbose))
            elif i == n_layers - 1:
                layers.append(SpectralConv1d(hidden_dim, 1, modes, verbose=verbose))
            else:
                layers.append(SpectralConv1d(hidden_dim, hidden_dim, modes, verbose=verbose))
        self.layers = nn.ModuleList(layers)

        if self.res_layer_type == "linear":
            res_layers = []
            for _ in range(n_layers-1):
                res_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.res_layers = nn.ModuleList(res_layers)

        if self.time_embed_type == "mlp":
            self.time_embd = MLP(3, 1, hidden_dim, hidden_dim, verbose=verbose)
        

    def forward(self, x, t):
        print("input shape: ", x.shape) if self.verbose else None
        print("t shape: ", t.shape) if self.verbose else None
        if self.time_embed_type == "mlp":
            embds = self.time_embd(t).view(-1, 1, self.hidden_dim)
            print("embds shape: ", embds.shape) if self.verbose else None
        else:
            x = torch.cat((x, t.unsqueeze(-1)), dim=-1)
            embds = 0
        output_dim = x.shape[-1]
        for i, layer in enumerate(self.layers):
            print("Layer: ", i) if self.verbose else None
            if i == 0:
                x = layer(x, self.hidden_dim) + embds
            elif i < self.n_layers - 1:
                x_act = F.silu(x)
                x = layer(x_act, self.hidden_dim) + embds
                if self.res_layer_type is not None:
                    x += self.res_layers[i-1](x_act)
            else:
                x = layer(F.silu(x), output_dim)
        return x