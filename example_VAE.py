import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from mpl_toolkits.axes_grid1 import ImageGrid
from torchvision.utils import save_image, make_grid


# create a transofrm to apply to each datapoint
transform = transforms.Compose([transforms.ToTensor()])

# download the MNIST datasets
path = '~/datasets'
train_dataset = MNIST(path, transform=transform, download=True)
test_dataset  = MNIST(path, transform=transform, download=True) #lmao this never gets used. bad guide

# create train and test dataloaders
batch_size = 100
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     

# get 25 sample training images for visualization
dataiter = iter(train_loader)
image = next(dataiter)

num_samples = 25
sample_images = [image[0][i,0] for i in range(num_samples)] 

fig = plt.figure(figsize=(5, 5))
grid = ImageGrid(fig, 111, nrows_ncols=(5, 5), axes_pad=0.1)

for ax, im in zip(grid, sample_images):
    ax.imshow(im, cmap='gray')
    ax.axis('off')

plt.show()
     


class Encoder(nn.Module):
    
    def __init__(self, input_dim=784, hidden_dim=512, latent_dim=256):
        super(Encoder, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.var = nn.Linear (hidden_dim, latent_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.training = True
        
    def forward(self, x):
        x = self.LeakyReLU(self.linear1(x))
        x = self.LeakyReLU(self.linear2(x))

        mean = self.mean(x)
        log_var = self.var(x)                     
        return mean, log_var
     

class Decoder(nn.Module):
    
    def __init__(self, output_dim=784, hidden_dim=512, latent_dim=256):
        super(Decoder, self).__init__()

        self.linear2 = nn.Linear(latent_dim, hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        x = self.LeakyReLU(self.linear2(x))
        x = self.LeakyReLU(self.linear1(x))
        
        x_hat = torch.sigmoid(self.output(x))
        return x_hat
     

class VAE(nn.Module):

    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=200):
        super(VAE, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2)
            )
        
        # latent mean and variance 
        self.mean_layer = nn.Linear(latent_dim, 2)
        self.logvar_layer = nn.Linear(latent_dim, 2)
        
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
            )
     
    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(device)      
        z = mean + var*epsilon
        return z

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar
        
    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) 
        x_hat = self.decode(z)  
        return x_hat, mean, log_var
     

model = VAE().to(device)
optimizer = Adam(model.parameters(), lr=1e-3)


def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD
     

def train(model, optimizer, epochs, device, x_dim=784):
    model.train()
    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(batch_size, x_dim).to(device)

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)
            
            overall_loss += loss.item()
            
            loss.backward()
            optimizer.step()

        print("\tEpoch", epoch + 1, "\tAverage Loss: ", overall_loss/(batch_idx*batch_size))
    return overall_loss
     

train(model, optimizer, epochs=50, device=device)
     
	# Epoch 1 	Average Loss:  180.83752029750104
	# Epoch 2 	Average Loss:  163.2696611703099
	# Epoch 3 	Average Loss:  158.45957443721306
	# Epoch 4 	Average Loss:  155.22525566699707
	# Epoch 5 	Average Loss:  153.2932642294449
	# Epoch 6 	Average Loss:  151.85221148202734
	# Epoch 7 	Average Loss:  150.7171567847454
	# Epoch 8 	Average Loss:  149.69312638577316
	# Epoch 9 	Average Loss:  148.78454667284015
	# Epoch 10 	Average Loss:  148.15143693264815
	# Epoch 11 	Average Loss:  147.4398520809422
	# Epoch 12 	Average Loss:  147.11726756508244
	# Epoch 13 	Average Loss:  146.85235046692404
	# Epoch 14 	Average Loss:  146.26805739057284
	# Epoch 15 	Average Loss:  145.90820591284955
	# Epoch 16 	Average Loss:  145.59176821395033
	# Epoch 17 	Average Loss:  145.32233085415797
	# Epoch 18 	Average Loss:  145.02244184643678
	# Epoch 19 	Average Loss:  144.71482943577837
	# Epoch 20 	Average Loss:  144.56753637246973
	# Epoch 21 	Average Loss:  144.13816487766067
	# Epoch 22 	Average Loss:  144.29326325125209
	# Epoch 23 	Average Loss:  143.72742146741965
	# Epoch 24 	Average Loss:  143.7537937089159
	# Epoch 25 	Average Loss:  143.51257628273686
	# Epoch 26 	Average Loss:  143.23082544801233
	# Epoch 27 	Average Loss:  142.9511411434422
	# Epoch 28 	Average Loss:  142.95305561352254
	# Epoch 29 	Average Loss:  142.90418625769513
	# Epoch 30 	Average Loss:  142.62190700320846
	# Epoch 31 	Average Loss:  142.55285526332952
	# Epoch 32 	Average Loss:  142.27065741078883
	# Epoch 33 	Average Loss:  142.47379079064066
	# Epoch 34 	Average Loss:  142.14543451325125
	# Epoch 35 	Average Loss:  142.18472630164857
	# Epoch 36 	Average Loss:  141.92655483748956
	# Epoch 37 	Average Loss:  141.8069266811874
	# Epoch 38 	Average Loss:  141.78985615674563
	# Epoch 39 	Average Loss:  141.5138033604184
	# Epoch 40 	Average Loss:  141.42040278719742
	# Epoch 41 	Average Loss:  141.42817399115714
	# Epoch 42 	Average Loss:  141.27378532906405
	# Epoch 43 	Average Loss:  141.03225473445326
	# Epoch 44 	Average Loss:  141.09916806330864
	# Epoch 45 	Average Loss:  141.24284993217864
	# Epoch 46 	Average Loss:  140.94183424196578
	# Epoch 47 	Average Loss:  140.7548155128339
	# Epoch 48 	Average Loss:  140.76203237492174
	# Epoch 49 	Average Loss:  140.66125756469114
	# Epoch 50 	Average Loss:  140.6244228499322


def generate_digit(mean, var):
    z_sample = torch.tensor([[mean, var]], dtype=torch.float).to(device)
    x_decoded = model.decode(z_sample)
    digit = x_decoded.detach().cpu().reshape(28, 28) # reshape vector to 2d array
    plt.title(f'[{mean},{var}]')
    plt.imshow(digit, cmap='gray')
    plt.axis('off')
    plt.show()
     

#img1: mean0, var1 / img2: mean1, var0
generate_digit(0.0, 1.0), generate_digit(1.0, 0.0)


def plot_latent_space(model, scale=5.0, n=25, digit_size=28, figsize=15):
    # display a n*n 2D manifold of digits
    figure = np.zeros((digit_size * n, digit_size * n))

    # construct a grid 
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = torch.tensor([[xi, yi]], dtype=torch.float).to(device)
            x_decoded = model.decode(z_sample)
            digit = x_decoded[0].detach().cpu().reshape(digit_size, digit_size)
            figure[i * digit_size : (i + 1) * digit_size, j * digit_size : (j + 1) * digit_size,] = digit

    plt.figure(figsize=(figsize, figsize))
    plt.title('VAE Latent Space Visualization')
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("mean, z [0]")
    plt.ylabel("var, z [1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()
     

plot_latent_space(model, scale=1.0)
plot_latent_space(model, scale=5.0)