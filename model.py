import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from gdn_layer import GDN
from constants import device, epsilon
import time
from tqdm import tqdm
from piq import multi_scale_ssim
from itertools import cycle


class DensityEstimator(nn.Module):

    def __init__(self, layer_sizes, channels):
        '''
        Estimates a probability density according to Appendix 6.1 of https://arxiv.org/pdf/1802.01436.
        An "underlying cumulative distribution function" (CDF) is learned through a MLP, and its PDF is convolved with
        a uniform distribution U(-1/2, 1/2) to generate a probability density for encoder model output. The
        PDF does not need to be computed explicity for this since the convolution can be written in closed
        form using the CDF.

        Args:
            layer_sizes (List): List of integers representing the sizes of hidden layers. Note that the first and
                last layers have size one to create a univariate density function, which is handled implicitly.

            channels (int): Integer representing the number of input channels to compute densities for.

        Example:
            DensityEstimator([3, 4, 2]) creates a 5-layer model with shapes 1 -> 3 -> 4 -> 2 -> 1.
        '''
        
        super(DensityEstimator, self).__init__()

        self.num_layers = len(layer_sizes) + 1 #does not count input layer

        full_sizes = [1] + layer_sizes + [1]
        self.a = nn.ParameterList([torch.randn((channels, sz)) for sz in layer_sizes])
        self.b = nn.ParameterList([torch.randn((channels, full_sizes[i])) for i in range(1, self.num_layers + 1)])
        self.H = nn.ParameterList([torch.randn((channels, full_sizes[i], full_sizes[i+1])) for i in range(self.num_layers)])

        self.sigmoid = nn.Sigmoid()
        self.loss = nn.NLLLoss()

        self.softplus = nn.functional.softplus

        self.optimizer = SGD(self.parameters(), lr=1e-3)


    def channel_matrix_product(self, x, k):
        """
        Performs a "semi-element-wise" multiplication by taking the matrix product of
        the (b, n)th vector of length M in `x` and the nth (M x P) matrix in self.H[k].
        
        Args:
            x (Tensor): Input tensor of shape (B, N, M).
            k (int): Layer index.
        
        Returns:
            torch.Tensor: Output tensor of shape (B, N, P).
        """
        # 'bni,npi->bnp' translates to:
        # - For each batch `b`, for each channel `n`, perform a matrix multiplication between `x[b, n, :]` and `w[n, :, :]`
        y = torch.einsum('bni,nip->bnp', x, self.softplus(self.H[k]))
        return y


    def g(self, x, k):
        '''
        Activation function for hidden layers.
        '''
        return x + torch.tanh(self.a[k]) * torch.tanh(x)


    def c(self, x):
        '''
        CDF (cumulative distribution function) for "underlying model".

        Args:
            x (Tensor): Must have shape (batch_size, channels, 1).

        Returns:
            c (Tensor): PDF corresponding to underlying model at x, has shape (batch_size, channels, 1).
        '''
        for k in range(self.num_layers - 1):
            x = self.channel_matrix_product(x, k) + self.b[k]
            x = self.g(x, k)
        x = self.channel_matrix_product(x, -1) + self.b[-1]
        c = self.sigmoid(x)
        return c
    

    def forward(self, x):
        '''
        PDF (probability density function) for y_tilde.

        Args:
            x (Tensor): Must have shape (batch_size, channels, 1).

        Returns:
            p (Tensor): PDF corresponding to y_tilde at x, has shape (batch_size, channels, 1).
        '''
        p = self.c(x + 1/2) - self.c(x - 1/2)
        return p



class VAE(nn.Module):

    def __init__(self, channel_sizes, kernel_sizes, strides, pdf_layer_sizes, lam, height, width, color=True):
        '''
        Args:
            channel_sizes (list of ints): List of channel sizes from input to output

            kernel_sizes (list of ints): List of kernel sizes from input to output

            strides (list of ints): List of kernel strides from input to output

            pdf_layer_sizes (List): List of integers representing the sizes of hidden layers
                in the DensityEstimator neural networks. Note that the first and last layers have 
                size one to create a univariate density function, which is handled implicitly.

            lam (float): Rate-distortion tradeoff parameter, should be between 0 and 1.

            height (int): Height of training input images. Not used in eval mode.

            width (int): Width of training input images. Not used in eval mode.

            color (boolean): Whether training images are 3 channel (color) or 1 channel (grayscale). Not currently used.
            
        Example:
            `VAE([1, 10, 20, 5], [9, 5, 5, 9], [4, 2, 2, 4], [3, 4, 2], 0.5)` specifies a variational autoencoder
            that creates an encoder like `Encoder([1, 10, 20], [9, 5], [4, 2])` and a decoder like
            `Decoder([20, 5, 1], [5, 9], [2, 4])`, a set of DensityEstamator estimators with shapes
            1 -> 3 -> 4 -> 2 -> 1, and a lam rate-distortion tradeoff parameter of 0.5.
        '''
        if not isinstance(channel_sizes, list):
            raise Exception("channel_sizes must be a list.")
        if not isinstance(kernel_sizes, list):
            raise Exception("kernel_sizes must be a list.")
        if not isinstance(strides, list):
            raise Exception("strides must be a list.")
        
        if len(channel_sizes) != len(kernel_sizes) or len(channel_sizes) != len(strides):
            raise Exception("All inputs must have the same length. See specifications.")

        # this is not a condition Balle et al. uses because they use Loss = R + lam * D instead of Loss = (1 - lam) * R + lam * D.
        # if (lam < 0) or (lam > 1):
        #     raise Exception(f"Rate-distortion tradeoff parameter `lam` must be between 0 and 1. Got: {lam}.")


        super(VAE, self).__init__()

        self.channel_sizes = channel_sizes
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.pdf_layer_sizes = pdf_layer_sizes
        self.lam = lam
        self.height = height
        self.width = width
        self.color = color


        mid_idx = len(channel_sizes) // 2

        # create encoder and decoder from inputs
        channel_sizes_encoder = channel_sizes[:mid_idx + 1]
        kernel_sizes_encoder = kernel_sizes[:mid_idx]
        strides_encoder = strides[:mid_idx]

        channel_sizes_decoder = channel_sizes[mid_idx:] + [channel_sizes[0]]
        kernel_sizes_decoder = kernel_sizes[mid_idx:]
        strides_decoder = strides[mid_idx:]

        self.encoder_layers = []

        for i in range(mid_idx):
            sz_in = channel_sizes_encoder[i]
            sz_out = channel_sizes_encoder[i+1]
            sz_kernel = kernel_sizes_encoder[i]
            sz_downsample = strides_encoder[i]

            self.encoder_layers.append(nn.Conv2d(sz_in, sz_out, sz_kernel, sz_downsample, padding=(sz_kernel - sz_downsample)//2 + 1))
            self.encoder_layers.append(GDN(sz_out, device, inverse=False))


        self.decoder_layers = []

        for i in range(mid_idx):
            sz_in = channel_sizes_decoder[i]
            sz_out = channel_sizes_decoder[i+1]
            sz_kernel = kernel_sizes_decoder[i]
            sz_downsample = strides_decoder[i]

            self.decoder_layers.append(GDN(sz_in, device, inverse=True))
            self.decoder_layers.append(nn.ConvTranspose2d(sz_in, sz_out, sz_kernel, sz_downsample, padding=(sz_kernel - sz_downsample)//2 + 1, output_padding=1))

        # encoder and decoder as layer sequences for easy calling. also adds layer parameters to module.
        self.encoder = nn.Sequential(*self.encoder_layers)
        self.decoder = nn.Sequential(*self.decoder_layers)

        # optimization and training
        self.lr = 1e-3
        self.optimizer = Adam(self.parameters(), lr=1e-3)

        # to compute entropy (rate part of loss function), we need to create a DensityEstimator network
        # this will be trained alongside the VAE in the VAE's training loop. It learns a PDF for each
        # latent space pixel.
        self.req_divisor = np.prod(strides[:mid_idx])
        mid_channel_depth = int(channel_sizes_encoder[-1] * width * height / self.req_divisor**2)
        self.p = DensityEstimator(pdf_layer_sizes, mid_channel_depth)

        # prevent gradient explosion that kills training. 
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.p.parameters(), max_norm=1.0)
    
    
    def encode(self, x):
        return self.encoder(x)


    def quantize(self, x):
        if self.training:
            #returns noise distributed as Unif(x_{ijk} - 0.5, x_{ijk} + 0.5) for all i, j, k.
            return x - 0.5 + torch.rand_like(x)
        else:
            #quantizes continuous-valued input into integers.
            return torch.round(x)


    def decode(self, x):
        return self.decoder(x)


    def forward(self, x):
        y = self.encode(x)
        q = self.quantize(y)
        x_hat = self.decode(q)
        return x_hat, q #return both the reconstruction and the quantized feature representation.


    def loss_function(self, x, x_hat, y, distortion_only=True):
        batch_size = x.shape[0]
        # distortion is sample mean of image error across entire batch
        distortion = torch.sum((x - x_hat)**2) / batch_size
        
        if distortion_only:
            return distortion

        # rate is a minimization of the entropy of the image probabilities across entire batch
        ps = self.p(y.view(y.shape[0], -1, 1))
        ps = torch.clamp(ps, min=epsilon)
        entropy = -torch.sum(torch.log2(ps)) / batch_size

        # loss is quantified by tradeoff between rate and distortion
        # I think it is nicer to represent this tradeoff as (1 - self.lam) * entropy + self.lam * distortion
        return entropy + self.lam * distortion
    
        
    def train_model(self, epochs:int, data_loader:DataLoader, test_loader:DataLoader|None = None, tol:float=1e-4, distortion_only=True):
        '''
        Train a VAE (variational autoencoder) model and its underlying probability DensityEstimator.

        Args:
            epochs (int): Number of epochs to train over.

            data_loader (DataLoader): Image data loader.

            test_loader (DataLoader|None): Test image data loader. Only used for MS-SSIM scores. Default: None.

            tol (float): Tolerance specifying the maximum relative difference between consecutive average
                epoch losses to qualify as convergence. This will end training early.

            distortion_only (boolean): Whether to use only the distortion MSE loss term, or include the rate entropy term.
        
        Returns:
            all_losses (Tensor): (N x 2) Tensor of MSE and MS-SSIM scores, averaged over batches.

            epoch_losses (Tensor): (N,) Tensor of MSE scores, averaged over batches and epoch steps.
        '''
        model = self.to(device)
        model.train()
        overall_loss = 0
        all_losses = []
        epoch_losses = []
        test_iter = cycle(test_loader)
        for epoch in tqdm(range(epochs)):
            print()
            t1 = time.perf_counter()
            overall_loss = 0
            all_epoch_losses = []
            for batch_idx, x in enumerate(data_loader):
                x = x.to(device)

                self.optimizer.zero_grad()
                self.p.optimizer.zero_grad()

                x_hat, y = model(x)
                loss = self.loss_function(x, x_hat, y, distortion_only)
                
                overall_loss += loss.item()
                
                loss.backward()
                self.optimizer.step()
                self.p.optimizer.step()

                with torch.no_grad(): 
                    test_x = next(test_iter)
                    # need to align test_x with stride sizes for proper encoding and decoding convolution sizes.
                    new_h = test_x.shape[2] - (test_x.shape[2] % self.req_divisor)
                    new_w = test_x.shape[3] - (test_x.shape[3] % self.req_divisor)
                    test_x = test_x[:, :, :new_h, :new_w]
                    test_x = test_x.to(device)

                    test_x_hat, _ = model(test_x)
                    if test_x.shape[2] > 160 and test_x.shape[3] > 160:
                        ms_ssim = multi_scale_ssim(test_x, torch.clip(test_x_hat, 0, 1))
                        # print(f"Average Batch Performance: \tMSE Loss: {loss.item():.0f} \tMS-SSIM: {ms_ssim.item():.4f}")
                        all_losses.append([loss.item(), ms_ssim.item()])
                        all_epoch_losses.append([loss.item(), ms_ssim.item()])
                    else:
                        # print(f"Average Batch Performance: \tMSE Loss: {loss.item():.0f} \tMS-SSIM unavailable due to small image dimensions.")
                        all_losses.append([loss.item(), 0])
                        all_epoch_losses.append([loss.item(), 0])
            
            epoch_losses.append(np.mean(all_epoch_losses, axis=0))
            t2 = time.perf_counter()

            
            
            if epoch > 0:
                rel_epoch_improvement = (epoch_losses[-2][0] - epoch_losses[-1][0]) / epoch_losses[-1][0]
                print(
                    "\tEpoch", epoch + 1,
                    "\tAverage Loss: ", overall_loss / batch_idx,
                    "\tRelative Epoch Improvement: ", rel_epoch_improvement,
                    "\tMS-SSIM: ", epoch_losses[-1][1],
                    "\tTime: ", t2 - t1, " seconds",
                )

                #if stagnated, decrease the learning rate
                if rel_epoch_improvement < 0.05 and self.lr > 1e-7:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] /= 10
                    for param_group in self.p.optimizer.param_groups:
                        param_group['lr'] /= 10
            else:
                print(
                    "\tEpoch", epoch + 1,
                    "\tAverage Loss: ", overall_loss / batch_idx,
                    "\tRelative Epoch Improvement: N/A",
                    "\tMS-SSIM: ", epoch_losses[-1][1],
                    "\tTime: ", t2 - t1, " seconds",
                )

            if epoch > 0 and np.abs(epoch_losses[-1][0] - epoch_losses[-2][0]) / epoch_losses[-1][0] < tol:
                #converged early
                print(f"Converged in {epoch} epochs.")
                break
        
        return torch.tensor(all_losses), torch.tensor(epoch_losses)