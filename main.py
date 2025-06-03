import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from mpl_toolkits.axes_grid1 import ImageGrid
from data_handling import CocoDataset, plot_image_from_tensor, get_COCO_train_and_test_datasets, load_image_as_tensor
from constants import figure_path, save_path, device
from model import VAE, DensityEstimator
from gdn_layer import GDN


def main():
    # Create train and test datasets from COCO database
    width = 64
    height = 64
    train_dataset = CocoDataset(sample_width=width, sample_height=height, color=False)
    test_dataset = CocoDataset(sample_width=-1, sample_height=-1, color=False)

    # Create train and test dataloaders
    batch_size = 32
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)


    # get 25 sample training images for visualization
    dataiter = iter(train_loader)
    image = next(dataiter)

    num_samples = 25
    sample_images = [image[i] for i in range(num_samples)] 

    fig = plt.figure(figsize=(5, 5))
    grid = ImageGrid(fig, 111, nrows_ncols=(5, 5), axes_pad=0.1)

    for ax, im in zip(grid, sample_images):
        plot_image_from_tensor(im, ax, None, False)

    plt.savefig(figure_path + "\\coco_sample.png", bbox_inches='tight')
    plt.show()
    
    lam = 10000
    channel_sizes = [1, 128, 128, 128, 128, 128]
    kernel_sizes = [9, 5, 5, 5, 5, 9]
    strides = [4, 2, 2, 2, 2, 4]
    pdf_layer_sizes = [4, 4, 4]

    epochs = 50

    # model = VAE(channel_sizes, kernel_sizes, strides, pdf_layer_sizes, lam, height, width)
    model = torch.load(save_path + f"big_model_model_50_epochs_10_lam", map_location=device)
    # losses, epoch_avg_losses = model.train_model(epochs=epochs, data_loader=train_loader, test_loader=test_loader)

    # torch.save(model, save_path + f"gray_model_{epochs}_epochs_{lam}_lam")
    # np.save(save_path + f"gray_losses_{epochs}_epochs_{lam}_lam.npy", losses)
    # np.save(save_path + f"gray_epoch_avg_losses_{epochs}_epochs_{lam}_lam.npy", epoch_avg_losses)
    
    # model = torch.load(save_path + f"gray_model_{epochs}_epochs_{lam}_lam")

    test_x = load_image_as_tensor("C:\\Users\\dwdjr\\Documents\\ENGSCI 250 Project\\figures\\harvard_logo.png", False)
    new_h = model.req_divisor - (test_x.shape[2] % model.req_divisor)
    new_w = model.req_divisor - (test_x.shape[3] % model.req_divisor)
    test_x = torch.block_diag(torch.zeros((new_h//2, new_w//2)), test_x[0, 0, :, :], torch.zeros((new_h//2 + 1, new_w//2)))
    test_x = test_x.view((1, 1, *test_x.shape))

    plot_image_from_tensor(test_x[0], save_path="C:\\Users\\dwdjr\\Documents\\ENGSCI 250 Project\\figures\\harvard_logo_gray.png")

    test_x = test_x.to(device)
    plot_image_from_tensor(model(test_x)[0][0].detach().cpu(), save_path="C:\\Users\\dwdjr\\Documents\\ENGSCI 250 Project\\figures\\harvard_logo_compressed.png")

    return

    losses = np.load(save_path + f"gray_losses_{epochs}_epochs_{lam}_lam.npy")
    # epoch_avg_losses = np.load(save_path + f"gray_epoch_avg_losses_{epochs}_epochs_{lam}_lam.npy")
    # losses_per_epoch = np.ceil(len(train_dataset) / batch_size)

    dataiter = iter(test_loader)
    image = next(dataiter)
    plot_image_from_tensor(image[0])
    plot_image_from_tensor(model(image)[0][0].detach())

    plt.plot(losses[:, 0])
    # plt.plot(losses_per_epoch * np.arange(len(epoch_avg_losses)) + losses_per_epoch/2, (epoch_avg_losses), 'o', label='Average Losses Over Epochs')
    plt.title(f'Training Convergence, $\\lambda = {lam}$')
    plt.xlabel('Training Steps')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    plt.show()

    plt.plot(losses[:, 1], label='Average Batch MS-SSIM')
    plt.title(f'MS-SSIM Performance, $\\lambda = {lam}$')
    plt.xlabel('Training Steps')
    plt.ylabel('MS-SSIM')
    plt.show()


if __name__ == "__main__":
    main()