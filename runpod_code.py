'''
This file documents the exact code on Runpod.io that was used for GPU access. This should be identical to the 
code in other files on submission.
'''

import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
from piq import multi_scale_ssim
from torch.autograd import Function
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from itertools import cycle
import heapq
from scipy.integrate import quad_vec

# Define paths to project dependencies
project_path = ""
code_path = "code"
coco_folder_path = "coco_dataset"
trainable_dataset_path = "trainable_dataset"
figure_path = "figures"
save_path = "saved_data"

# Get working device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}.")

#small value
epsilon = 1e-8



class CocoDataset(Dataset):
    def __init__(self, width=None, height=None, sample_width=256, sample_height=256, pre_transform=None, post_transform=None, color=True, num=-1):
        """
        Args:
            width (int, optional): Optional reshaped width of image.

            height (int, optional): Optional reshaped height of image.

            sample_width (int, optional): Optional width of image samples. Choosing -1 causes the full 
                image to be returned. Default: 256.

            sample_height (int, optional): Optional height of image samples. Choosing -1 causes the full
                image to be returned. Default: 256.

            pre_transform (callable, optional): Optional transform to be applied
                on an image before it is converted to a tensor.

            post_transform (callable, optional): Optional transform to be applied
                on an image after it is converted to a tensor.

            color (boolean, optional): Optional boolean to return 3-channel data or 1-channel data.
                Default: True.

            num (int, optional): Optional integer representing the number of images to load from the database.
                Default: -1 indicates that all images are loaded. 
        """

        if (sample_height == -1 and sample_width != -1) or (sample_height != -1 and sample_width == -1):
            raise Exception("Both sample_height and sample_width must be -1 if either is -1.")

        self.root_dir = trainable_dataset_path
        self.color = color
        self.sample_width = sample_width
        self.sample_height = sample_height

        if (sample_width != -1):
            self.upsample_transform = transforms.Resize((sample_height, sample_width))
        else:
            self.upsample_transform = None

        if height and width:
            if pre_transform:
                pre_transform = transforms.Compose([pre_transform, transforms.Resize((height, width))])
            else:
                pre_transform = transforms.Resize((height, width))
        
        #set up the appropriate transformation. will always convert images to tensors.
        self.transform = None
        if pre_transform:
            self.transform = transforms.Compose([pre_transform, transforms.ToTensor()])
        if post_transform:
            if self.transform:
                self.transform = transforms.Compose([self.transform, post_transform])
            else:
                self.transform = transforms.Compose([transforms.ToTensor(), post_transform])
        if not self.transform:
            self.transform = transforms.ToTensor()

        self.image_files = [f for f in os.listdir(self.root_dir) if os.path.isfile(os.path.join(self.root_dir, f))]
        if num != -1:
            self.image_files = self.image_files[:num]
    

    def __len__(self):
        return len(self.image_files)
    

    def __getitem__(self, idx):

        if isinstance(idx, torch.Tensor):
            # transform already applied 
            idx = idx.item()
        
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert("RGB")
        
        image = self.transform(image)

        if not self.color:
            image = torch.mean(image, dim=0, keepdim=True)

        if self.sample_height == -1:
            return image

        max_h_idx = image.shape[1] - self.sample_height
        max_w_idx = image.shape[2] - self.sample_width

        if max_h_idx <= 0 or max_w_idx <= 0:
            image = self.upsample_transform(image)
            return image

        h_idx = torch.randint(0, max_h_idx, (1,))[0]
        w_idx = torch.randint(0, max_w_idx, (1,))[0]

        image = image[:, h_idx : h_idx + self.sample_height, w_idx : w_idx + self.sample_width]
        #add uniform noise:
        image += (torch.rand_like(image) - 0.5)/256
        image = torch.clip(image, 0, 1)
        
        return image


def get_COCO_train_and_test_datasets(coco_dataset: CocoDataset, train_proportion: float = 0.8):
    '''
    Returns a training dataset and testing dataset from the total COCO dataset, where
    `train_proportion` controls the proportion of the total dataset used for training.
    Does not edit the original dataset.

    Args:
        coco_dataset (CocoDataset): total COCO dataset object.

        train_proportion (float, optional): Proportion of the total COCO dataset used for training.
            Must be between 0 and 1. Default 0.8.

    Returns:
        train_data (CocoDataset): Training data.
        
        test_data (CocoDataset): Testing data.
    '''
    if train_proportion <= 0 or train_proportion >= 1:
        raise Exception(f"Argument `train_proportion` must be between 0 and 1. Got: {train_proportion}.")

    train_data = copy.deepcopy(coco_dataset)
    test_data = copy.deepcopy(coco_dataset)

    cutoff = int(len(coco_dataset) * train_proportion)

    train_data.image_files = train_data.image_files[:cutoff]
    test_data.image_files = test_data.image_files[cutoff:]

    return train_data, test_data



def plot_image_from_tensor(t:torch.Tensor, ax=None, save_path=None, show=True):
    '''
    Shows and/or saves an image represented by a tensor of shape (3, height, width).

    Args:
        t (torch.Tensor): Tensor to show as image.

        ax (matplotlib.pyplot.Axes, optional): Axes to plot and show

        save_path (str, optional): Path to save image to

        show (boolean, optional): Whether image should be shown. Default True.
    '''
    if ((t.shape[0] != 3 and t.shape[0] != 1) or len(t.shape) != 3):
        raise Exception(f"Tensor argument `t` has wrong shape or dimension. Got shape: {t.shape}, expected: (1 or 3, height, width).")
    
    if t.shape[0] == 1:
        grayscale = True
    else:
        grayscale = False

    if ax:
        if grayscale:
            ax.imshow(t.numpy().transpose((1, 2, 0)), cmap="gray")
        else:
            ax.imshow(t.numpy().transpose((1, 2, 0)))
        ax.axis('off')
    else:
        if grayscale:
            plt.imshow(t.numpy().transpose((1, 2, 0)), cmap="gray")
        else:
            plt.imshow(t.numpy().transpose((1, 2, 0)))
        plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    
    if show:
        plt.show()
    
    if not ax:
        plt.close()


class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones(inputs.size(), device=inputs.device)*bound
        b = b.to(inputs.device)
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)
  
    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors

        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None


class GDN(nn.Module):
    """Generalized divisive normalization layer.
    y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]^2))
    """
  
    def __init__(self,
                 ch,
                 device,
                 inverse=False,
                 beta_min=1e-6,
                 gamma_init=.1,
                 reparam_offset=2**-18):
        super(GDN, self).__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.gamma_init = gamma_init
        self.reparam_offset = torch.tensor([reparam_offset], device=device)

        self.build(ch, torch.device(device))
  
    def build(self, ch, device):
        self.pedestal = self.reparam_offset**2
        self.beta_bound = (self.beta_min + self.reparam_offset**2)**.5
        self.gamma_bound = self.reparam_offset

        # Create beta param
        beta = torch.sqrt(torch.ones(ch, device=device)+self.pedestal)
        self.beta = nn.Parameter(beta)

        # Create gamma param
        eye = torch.eye(ch, device=device)
        g = self.gamma_init*eye
        g = g + self.pedestal
        gamma = torch.sqrt(g)
        self.gamma = nn.Parameter(gamma)

    def forward(self, inputs):
        unfold = False
        if inputs.dim() == 5:
            unfold = True
            bs, ch, d, w, h = inputs.size() 
            inputs = inputs.view(bs, ch, d*w, h)

        _, ch, _, _ = inputs.size()

        # Beta bound and reparam
        beta = LowerBound.apply(self.beta, self.beta_bound)
        beta = beta**2 - self.pedestal 

        # Gamma bound and reparam
        gamma = LowerBound.apply(self.gamma, self.gamma_bound)
        gamma = gamma**2 - self.pedestal
        gamma  = gamma.view(ch, ch, 1, 1)

        # Norm pool calc
        norm_ = nn.functional.conv2d(inputs**2, gamma, beta)
        norm_ = torch.sqrt(norm_)
  
        # Apply norm
        if self.inverse:
            outputs = inputs * norm_
        else:
            outputs = inputs / norm_

        if unfold:
            outputs = outputs.view(bs, ch, d, w, h)
        return outputs
    

class DensityEstimator(nn.Module):

    def __init__(self, layer_sizes, channels, lr=0.001):
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
        self.lr = lr

        self.optimizer = SGD(self.parameters(), lr=lr)


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
            # self.encoder_layers.append(nn.ReLU())
            # self.encoder_layers.append(nn.BatchNorm2d(sz_out))


        self.decoder_layers = []

        for i in range(mid_idx):
            sz_in = channel_sizes_decoder[i]
            sz_out = channel_sizes_decoder[i+1]
            sz_kernel = kernel_sizes_decoder[i]
            sz_downsample = strides_decoder[i]

            self.decoder_layers.append(GDN(sz_in, device, inverse=True))
            # self.decoder_layers.append(nn.BatchNorm2d(sz_in))            
            # self.decoder_layers.append(nn.ReLU())
            self.decoder_layers.append(nn.ConvTranspose2d(sz_in, sz_out, sz_kernel, sz_downsample, padding=(sz_kernel - sz_downsample)//2 + 1, output_padding=1))
        
        # encoder and decoder as layer sequences for easy calling. also adds layer parameters to module.
        self.encoder = nn.Sequential(*self.encoder_layers)
        self.decoder = nn.Sequential(*self.decoder_layers)

        # optimization and training
        self.lr = 1e-4
        self.optimizer = Adam(self.parameters(), self.lr, eps=1e-3, amsgrad=True)
        self.lam = lam

        # to compute entropy (rate part of loss function), we need to create a DensityEstimator network
        # this will be trained alongside the VAE in the VAE's training loop. It learns a PDF for each
        # latent space pixel.
        self.req_divisor = np.prod(strides[:mid_idx])
        mid_channel_depth = int(channel_sizes_encoder[-1] * width * height / self.req_divisor**2)
        self.p = DensityEstimator(pdf_layer_sizes, mid_channel_depth, lr=self.lr)

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

        '''
        I think there should be a "compression step" here...
        We should have: x -> encoder -> quantizer -> compressor -> decompressor -> decoder -> x_hat
        '''

        x_hat = self.decode(q)
        return x_hat, q


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
        return (1 - self.lam) * entropy + self.lam * distortion
    
        

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
                if rel_epoch_improvement < 0.001:
                    if self.lr > 1e-7:
                        print("Detected learning stagnation. Attempting to continue learning by decreasing learning rate.")
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] /= 10
                        for param_group in self.p.optimizer.param_groups:
                            param_group['lr'] /= 10
                    else:
                        break
            else:
                print(
                    "\tEpoch", epoch + 1,
                    "\tAverage Loss: ", overall_loss / batch_idx,
                    "\tRelative Epoch Improvement: N/A",
                    "\tMS-SSIM: ", epoch_losses[-1][1],
                    "\tTime: ", t2 - t1, " seconds",
                )
        
        return torch.tensor(all_losses), torch.tensor(epoch_losses)