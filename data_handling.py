import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import copy
from constants import coco_folder_path, figure_path

def main():
    # Example usage:

    # Load images with shape (height, width)
    height = 256
    width = 256
    dataset = CocoDataset(width=width, height=height)

    # Example: Load a batch of batch_size images
    batch_size = 4
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    images = next(iter(data_loader))
    print(images.shape)  # Example shape: (batch_size, 3, height, width)

    # Show and save the first image in the batch
    fig_save_path = figure_path + "example_fig.png"
    plot_image_from_tensor(images[0], None, fig_save_path, True)




class CocoDataset(Dataset):
    def __init__(self, width=None, height=None, sample_width=256, sample_height=256, pre_transform=None, post_transform=None, color=True, num=-1, rand_sample=True):
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

        self.root_dir = coco_folder_path
        self.color = color
        self.sample_width = sample_width
        self.rand_sample = rand_sample
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
        
        if self.rand_sample:
            h_idx = torch.randint(0, max_h_idx, (1,))[0]
            w_idx = torch.randint(0, max_w_idx, (1,))[0]
        else:
            h_idx = 0
            w_idx = 0

        image = image[:, h_idx : h_idx + self.sample_height, w_idx : w_idx + self.sample_width]
        
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


def load_image_as_tensor(path, color=True):
    '''
    Loads image as tensor using PIL.
    '''
    image = Image.open(path).convert("RGB")
    transform = transforms.ToTensor()

    # Convert the image to a tensor
    image = transform(image)
    if not color:
        image = torch.mean(image, dim=0, keepdim=True)
    image = image.view((1, image.shape[0], image.shape[1], image.shape[2]))
    return image


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




if __name__ == "__main__":
    main()