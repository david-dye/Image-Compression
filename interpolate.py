import numpy as np
import torch
import copy
import matplotlib.pyplot as plt
from constants import save_path, device, project_path
from model import DensityEstimator, VAE     #required to load model
from gdn_layer import GDN                   #required to load model
import os
from tqdm import tqdm
from piq import multi_scale_ssim
import zlib
from encoder import get_density_estimator_codes, get_binary_representation
from data_handling import CocoDataset, plot_image_from_tensor
from torch.utils.data import DataLoader
import pickle


SMALL_SIZE = 16
MEDIUM_SIZE = 16

plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rcParams['legend.fontsize'] = MEDIUM_SIZE  # Example: 14-point font
plt.rcParams['figure.figsize'] = (8, 6)  # Example: 8 inches wide, 6 inches tall



def main():

    # Create train and test datasets from COCO database
    width = 256
    height = 256
    test_dataset = CocoDataset(sample_width=width, sample_height=height, color=False, rand_sample=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    idx = 9

    model = torch.load(os.path.join(project_path, "iter_models",  f"iter_{idx}_model"), map_location=device)
    print(model.lam)
    return
    # losses = np.load(os.path.join(save_path, f"iter_0_losses.npy"))
    # epoch_avg_losses = np.load(os.path.join(save_path, f"iter_0_epoch_avg_losses.npy"))
    # losses_per_epoch = np.ceil(len(train_dataset) / batch_size)

    test_iter = iter(test_loader)
    test_x = next(test_iter)

    # need to align test_x with stride sizes for proper encoding and decoding convolution sizes.
    new_h = test_x.shape[2] - (test_x.shape[2] % model.req_divisor)
    new_w = test_x.shape[3] - (test_x.shape[3] % model.req_divisor)
    test_x = test_x[0:1, :, :new_h, :new_w]

    # plot_image_from_tensor(test_x[0], save_path=os.path.join(project_path, "figures", "original_surf.png"))

    # test_x = test_x.to(device)
    
    # plot_image_from_tensor(model(test_x)[0][0].detach().cpu(), save_path=os.path.join(project_path, "figures", f"iter_{idx}_surf.png"))

    
    # with open(os.path.join(project_path, "iter_models", "codes", f"iter_{idx}_model"),"rb") as f:
    #     huffman_codes = pickle.load(f)

    # rate, distortion = get_rate_distortion_performance(model, test_loader, huffman_codes, 1)
    # rate = rate[0]
    # distortion = distortion[0]
    # print(model.lam, rate, distortion)

    return

    lams, paths = get_all_lambdas(os.path.join(project_path, "iter_models"))

    #first good iter is iter_9, or lams[:35]

    lams = lams[:35]
    paths = paths[:35]

    # model = get_interpolated_model(0.9995, lams, paths, True)

    # huffman_codes = get_density_estimator_codes(model.p, -100, 100)
    # with open(save_path + f"lam_09995_codes","wb") as f:
    #     pickle.dump(huffman_codes, f)

    # with open(save_path + f"lam_09995_codes","rb") as f:
    #     huffman_codes = pickle.load(f)

    # test_dataset = CocoDataset(sample_width=256, sample_height=256, color=False)
    # test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    # bpps, ms_ssims = get_rate_distortion_performance(model, test_loader, huffman_codes, 100)

    # print(
    #     f"\nSummary statistics for lambda = {model.lam} model:",
    #     "\nAverage bpp: ", np.mean(bpps), 
    #     "\tStandard Deviation: ", np.std(bpps, ddof=1),
    #     "\nAverage MS-SSIM: ", np.mean(ms_ssims), 
    #     "\tStandard Deviation: ", np.std(ms_ssims, ddof=1),
    #     "\n",
    # )
    # return

    bpp_means = []
    bpp_stds = []
    ssim_means = []
    ssim_stds = []

    interp_lams = []
    interp_bpp_means = []
    interp_bpp_stds = []
    interp_ssim_means = []
    interp_ssim_stds = []

    for i, lam in enumerate(lams):

        # model = get_interpolated_model((lam + lams[i+1]) / 2, lams, paths, linear=True)
        # huffman_codes = get_density_estimator_codes(model.p, -50, 50)
        # with open(os.path.join(project_path, "iter_models", "codes", f"sub_iter_{i}"),"wb") as f:
        #     pickle.dump(huffman_codes, f)

        # with open(os.path.join(project_path, "iter_models", "codes", f"sub_iter_{i}"),"rb") as f:
        #     huffman_codes = pickle.load(f)
        
        # test_dataset = CocoDataset(sample_width=256, sample_height=256, color=False)
        # test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
        # bpps, ms_ssims = get_rate_distortion_performance(model, test_loader, huffman_codes, 100)

        # np.save(os.path.join(project_path, "iter_models", "performances", f"sub_iter_{i}" + "_bpp"), bpps)
        # np.save(os.path.join(project_path, "iter_models", "performances", f"sub_iter_{i}" + "_ssim"), ms_ssims)

        bpps = np.load(os.path.join(project_path, "iter_models", "performances", f"iter_{43-i}_model" + "_bpp.npy"))
        ms_ssims = np.load(os.path.join(project_path, "iter_models", "performances", f"iter_{43-i}_model" + "_ssim.npy"))

        bpp_means.append(np.mean(bpps))
        bpp_stds.append(np.std(bpps, ddof=1))
        ssim_means.append(np.mean(ms_ssims))
        ssim_stds.append(np.std(ms_ssims, ddof=1))

        if i == len(lams) - 1:
            break
        
        interp_bpps = np.load(os.path.join(project_path, "iter_models", "performances", f"sub_iter_{i}" + "_bpp.npy"))
        interp_ms_ssims = np.load(os.path.join(project_path, "iter_models", "performances", f"sub_iter_{i}" + "_ssim.npy"))

        interp_lams.append((lam + lams[i+1]) / 2)
        interp_bpp_means.append(np.mean(interp_bpps))
        interp_bpp_stds.append(np.std(interp_bpps, ddof=1))
        interp_ssim_means.append(np.mean(interp_ms_ssims))
        interp_ssim_stds.append(np.std(interp_ms_ssims, ddof=1))

        # print(
        #     f"\nSummary statistics for lambda = {model.lam} model:",
        #     "\nAverage bpp: ", np.mean(bpps), 
        #     "\tStandard Deviation: ", np.std(bpps, ddof=1),
        #     "\nAverage MS-SSIM: ", np.mean(ms_ssims), 
        #     "\tStandard Deviation: ", np.std(ms_ssims, ddof=1),
        #     "\n",
        # )

    # plt.errorbar(1-np.array(lams), bpp_means, yerr=bpp_stds, ecolor='blue', capsize=3, linestyle='None', marker='o', markersize=5)
    # plt.xlabel('$1 - \\lambda$')
    # plt.ylabel('Compression Rate (bpp)')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.xscale('log')
    # plt.savefig(os.path.join(project_path, "figures", "true_only_bpps.png"), dpi=300)
    # plt.show()

    plt.errorbar(1-np.array(lams), bpp_means, yerr=bpp_stds, ecolor='blue', capsize=3, linestyle='None', markerfacecolor='blue', markeredgecolor='blue', marker='o', markersize=5, label="True")
    plt.errorbar(1-np.array(interp_lams), interp_bpp_means, yerr=interp_bpp_stds, ecolor='red', capsize=3, linestyle='None', markerfacecolor='red', markeredgecolor='red', marker='o', markersize=5, label="Interpolated")
    plt.xlabel('$1 - \\lambda$')
    plt.ylabel('Compression Rate (bpp)')
    # plt.grid(True)
    plt.tight_layout()
    plt.xscale('log')
    plt.legend()
    plt.savefig(os.path.join(project_path, "figures", "bpps.png"), dpi=300)
    plt.show()

    plt.errorbar(1-np.array(lams), ssim_means, yerr=ssim_stds, ecolor='blue', capsize=3, linestyle='None', markerfacecolor='blue', markeredgecolor='blue', marker='o', markersize=5, label="True")
    plt.errorbar(1-np.array(interp_lams), interp_ssim_means, yerr=interp_ssim_stds, ecolor='red', capsize=3, linestyle='None', markerfacecolor='red', markeredgecolor='red', marker='o', markersize=5, label="Interpolated")
    plt.xlabel('$1 - \\lambda$')
    plt.ylabel('Distortion (MS-SSIM)')
    # plt.grid(True)
    plt.tight_layout()
    plt.xscale('log')
    plt.legend()
    plt.savefig(os.path.join(project_path, "figures", "ms_ssims.png"), dpi=300)
    plt.show()

    plt.plot(bpp_means, 1 - np.array(ssim_means), color='blue', linestyle='None', marker='o', markersize=5, label="True")
    plt.plot(interp_bpp_means, 1 - np.array(interp_ssim_means), color='red', linestyle='None', marker='o', markersize=5, label="Interpolated")
    plt.xlabel('Compression Rate (bpp)')
    plt.ylabel('Distortion (1 - MS-SSIM)')
    # plt.grid(True)
    plt.tight_layout()
    plt.yscale('log')
    plt.legend()
    plt.savefig(os.path.join(project_path, "figures", "rate_distortion_curve.png"), dpi=300)
    plt.show()

    #now without the interpolated parts

    plt.errorbar(1-np.array(lams), bpp_means, yerr=bpp_stds, ecolor='blue', capsize=3, linestyle='None', markerfacecolor='blue', markeredgecolor='blue', marker='o', markersize=5)
    plt.xlabel('$1 - \\lambda$')
    plt.ylabel('Compression Rate (bpp)')
    # plt.grid(True)
    plt.tight_layout()
    plt.xscale('log')
    # plt.legend()
    plt.savefig(os.path.join(project_path, "figures", "bpps_nointerp.png"), dpi=300)
    plt.show()

    plt.errorbar(1-np.array(lams), ssim_means, yerr=ssim_stds, ecolor='blue', capsize=3, linestyle='None', markerfacecolor='blue', markeredgecolor='blue', marker='o', markersize=5)
    plt.xlabel('$1 - \\lambda$')
    plt.ylabel('Distortion (MS-SSIM)')
    # plt.grid(True)
    plt.tight_layout()
    plt.xscale('log')
    # plt.legend()
    plt.savefig(os.path.join(project_path, "figures", "ms_ssims_nointerp.png"), dpi=300)
    plt.show()

    plt.errorbar(bpp_means, 1 - np.array(ssim_means), xerr=bpp_stds, yerr=ssim_stds, ecolor='blue', capsize=3, linestyle='None', markerfacecolor='blue', markeredgecolor='blue', marker='o', markersize=5)
    plt.xlabel('Compression Rate (bpp)')
    plt.ylabel('Distortion (1 - MS-SSIM)')
    # plt.grid(True)
    plt.tight_layout()
    plt.yscale('log')
    # plt.legend()
    plt.savefig(os.path.join(project_path, "figures", "rate_distortion_curve_nointerp.png"), dpi=300)
    plt.show()

    # lams = get_lambdas(np.linspace(3, 5, 11))[::-1]
    # print(lams)
    # lams = np.linspace(0.99, 0.9999, 100)
    # weights_high = [get_interpolant_weights(lam, 0.99, 0.9999, False)[1] for lam in lams]
    # plt.plot(lams, weights_high)
    # plt.show()









def get_lambdas(x):
    return (1 - 10**(-x))


def get_interpolated_model(lam, known_lambdas, paths, linear=True):
    '''
    Interpolate new models given model weights. Specifically, given a query in the form of a 
    lambda value between min_lambda and max_lambda, return a model that mixes weights of known models.

    Args:
        lam: Parameter lying within range [min_lambda, max_lambda], inclusively, where
            these bounds are given by a known lambda array and paths. The returned model
            will be an approximation to a model trained with this lambda value.

        known_lambdas: array of known lambdas

        paths: list of paths to models corresponding to the known lambdas.

        linear (bool, optional): Whether to use linear interpolation or logarithmic interpolation
            between model weights. Consider allowing a cubic spline interpolation.

    Returns:
        model_interp (VAE): a variational autoencoder model with default weights approximating the
            optimal training weights with lambda parameter given by lam.
    '''

    if (lam < known_lambdas[0] or lam > known_lambdas[-1]):
        #could consider extrapolating curve outside of known interpolant points.
        #for now, raise exception
        raise ValueError(f'Parameter `lam` must be greater than {known_lambdas[0]} and less than {known_lambdas[-1]}. Got lam = {lam}.')
    
    #find the known model indices/lambda values
    idx_low = -1
    idx_high = -1
    known_low = -1
    known_high = -1
    for i, known_lam in enumerate(known_lambdas):
        if known_lam > lam:
            idx_low = i-1
            idx_high = i
            known_low = known_lambdas[idx_low]
            known_high = known_lambdas[idx_high]
            break
    
    print(f"Given lam = {lam}, interpolating between models lam_low = {known_low} and lam_high = {known_high}")
    
    model_low = torch.load(paths[idx_low], map_location=device)
    model_high = torch.load(paths[idx_high], map_location=device)
    model_low.training = False
    model_high.training = False
    
    # Create interpolant model
    model_interp = VAE(
        model_low.channel_sizes, 
        model_low.kernel_sizes, 
        model_low.strides, 
        model_low.pdf_layer_sizes, 
        lam,
        model_low.height,
        model_low.width,
        model_low.color
    )

    state_dict_low = model_low.state_dict()
    state_dict_high = model_high.state_dict()
    p_low = model_low.p.state_dict()
    p_high = model_high.p.state_dict()

    averaged_state_dict = dict()
    weight_low, weight_high = get_interpolant_weights(lam, known_low, known_high, linear)
    for key in state_dict_low.keys():
        averaged_state_dict[key] = weight_low * state_dict_low[key] + weight_high * state_dict_high[key]

    p_averaged_state_dict = dict()
    for key in p_low.keys():
        p_averaged_state_dict[key] = weight_low * p_low[key] + weight_high * p_high[key]

    model_interp.load_state_dict(averaged_state_dict)
    model_interp.p.load_state_dict(p_averaged_state_dict)

    return model_interp



def get_interpolant_weights(lam, low, high, linear=True):
    '''
    Gets two interpolant weights according to the chosen weighing scheme (either linear or logarithmic).

    NOTE: weight_low + weight_high = 1, weight_low = 0 if lam == high, and weight_low = 1 if lam == low.

    Returns:
        weight_low: the weight placed on the lower lambda model

        weight_high: the weight placed on the higher lambda model
    '''
    if (low >= high or lam < low or lam > high):
        raise ValueError('Invalid arguments. Must have low < high, lam >= low, and lam <= high.')

    if linear:
        #linearly interpolate
        def f(lam):
            return lam
    else:
        #logarithmically interpolate; i.e. bias towards low lambda
        #This is the inverse to the get_lambdas function.
        def f(lam):
            return np.log10(1/(1 - lam))
    
    dist_to_low = f(lam) - f(low)
    dist_to_high = f(high) - f(lam)
    total_dist = dist_to_low + dist_to_high

    weight_low = 1 - (dist_to_low/total_dist)
    weight_high = 1 - (dist_to_high/total_dist)

    return weight_low, weight_high


def get_all_lambdas(path):
    lams = []
    paths = []
    for model_file in tqdm(os.listdir(path)):
        model_path = os.path.join(path, model_file)
        if os.path.isfile(model_path):
            model = torch.load(model_path, map_location=device)
            lam = model.lam
            lams.append(lam)
            paths.append(model_path)
            del model

    lams = np.array(lams)
    sorted_indices = np.argsort(lams)

    ret_lams = []
    ret_paths = []
    for idx in sorted_indices:
        ret_lams.append(lams[idx])
        ret_paths.append(paths[idx])

    return ret_lams, ret_paths




def get_rate_distortion_performance(model, data_loader, huffman_codes, count):
    '''
    Gets `count` rate (bits per pixel) and distortion (MS-SSIM) values given a model,
    dataset, and Huffman coding scheme.
    '''

    ms_ssims = np.zeros(count)
    bpps = np.zeros(count)

    with torch.no_grad():
        # print(f"\nComputing rate and distortion values for model with lambda = {model.lam}. Progress:\n")
        # for i in tqdm(range(count)):
        for i in range(count):
            test_iter = iter(data_loader)
            x = next(test_iter)

            # need to align x with stride sizes for proper encoding and decoding convolution sizes.
            new_h = x.shape[2] - (x.shape[2] % model.req_divisor)
            new_w = x.shape[3] - (x.shape[3] % model.req_divisor)
            x = x[0:1, :, :new_h, :new_w]

            model.training = False
            x = x.to(device)

            x_hat, q = model(x)
            ms_ssim = multi_scale_ssim(x, torch.clip(x_hat, 0, 1))
            ms_ssims[i] = ms_ssim

            encoded_im = q[0]
            # encoded_shape = encoded_im.shape #only used for decoding
            a_enc = encoded_im.view((-1,)) #flatten for encoding

            #now, a_enc contains the values that we used to pass into a DensityEstimator.
            binary_rep = get_binary_representation(a_enc.detach(), huffman_codes)
            lempziv = zlib.compress(binary_rep.encode())
            bpp_lempziv = 8 * len(lempziv) / (new_h * new_w)
            bpps[i] = bpp_lempziv

    return bpps, ms_ssims





if __name__ == "__main__":
    main()