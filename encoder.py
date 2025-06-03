# Huffman coding + Golomb-Exponential coding schemes.
import numpy as np
import torch
import heapq
import matplotlib.pyplot as plt
from model import DensityEstimator, VAE
from gdn_layer import GDN
from constants import save_path, device, p1_init, shift_idx, PROB_BITS
from scipy.integrate import quad_vec
from tqdm import tqdm
from data_handling import CocoDataset, plot_image_from_tensor
from torch.utils.data import DataLoader
import pickle
import zlib



def main():
    
    # # Given example
    # chars = ['a', 'b', 'c', 'd', 'e', 'f']
    # freq = [4, 7, 2, 9, 9, 3]

    # # Build the Huffman tree
    # root = build_huffman_tree(chars, freq)

    # # Generate Huffman codes
    # huffman_codes = generate_huffman_codes(root)

    # # Print Huffman codes
    # for char, code in huffman_codes.items():
    #     print(f"Character: {char}, Code: {code}")

    model = torch.load(save_path + f"lam_0999_model_50_epochs_0.999_lam", map_location=device)
    # model = torch.load(save_path + f"lam_099999_big_model", map_location=device)
    model.training = False

    huffman_codes = get_density_estimator_codes(model.p)
    with open(save_path + f"lam_0999_codes","wb") as f:
        pickle.dump(huffman_codes, f)

    with open(save_path + f"lam_0999_codes","rb") as f:
        huffman_codes = pickle.load(f)

    width = 256
    height = 256
    train_dataset = CocoDataset(sample_width=width, sample_height=height, color=False)
    test_dataset = CocoDataset(sample_width=-1, sample_height=-1, color=False)

    # Create train and test dataloaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    test_iter = iter(train_loader)

    bpps = []

    num_test_images = 100

    for i in tqdm(range(num_test_images)):


        test_x = next(test_iter)

        # new_h = test_x.shape[2] - (test_x.shape[2] % model.req_divisor)
        # new_w = test_x.shape[3] - (test_x.shape[3] % model.req_divisor)
        # test_x = test_x[:, :, :new_h, :new_w]

        # plot_image_from_tensor(test_x[0])
        test_x = test_x.to(device)
        
        y = model.encode(test_x)
        q = model.quantize(y)
        encoded_im = q[0]
        # print(test_x.shape, encoded_im.shape)
        encoded_shape = encoded_im.shape
        a_enc = encoded_im.view((-1,)) #flatten for encoding

        #now, a_enc contains the values that we used to pass into a DensityEstimator.
        binary_rep = get_binary_representation(a_enc.detach(), huffman_codes)
        lempziv = zlib.compress(binary_rep.encode())
        bpp_lempziv = 8 * len(lempziv) / (height * width)

        bpps.append(bpp_lempziv)
        
        # print(
        #     "\nHuffman Code Binary String Length: ", len(binary_rep),
        #     "\nCompressed (Lempel-Ziv) Binary String Length: ", 8 * len(lempziv),
        #     "\nbit/pix: ", bpp_lempziv,
        #     "\n",
        # )

        #undo compression
        # de_lempziv = zlib.decompress(lempziv).decode()
        # a_dec = get_integer_representation(de_lempziv, huffman_codes)

        # a_dec = a_dec.view((1, *encoded_shape)) #unflatten for decoding
        # test_xhat = model.decode(a_dec)[0].detach().cpu()
        # plot_image_from_tensor(test_xhat)

    print(bpps)
    print(np.mean(bpps), np.std(bpps, ddof=1))





'''
Huffman Coding Algorithm, copied from https://www.geeksforgeeks.org/huffman-coding-in-python/
'''
class Node:
    def __init__(self, symbol=None, frequency=None):
        self.symbol = symbol
        self.frequency = frequency
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.frequency < other.frequency

def build_huffman_tree(chars, freq):
  
    # Create a priority queue of nodes
    priority_queue = [Node(char, f) for char, f in zip(chars, freq)]
    heapq.heapify(priority_queue)

    # Build the Huffman tree
    while len(priority_queue) > 1:
        left_child = heapq.heappop(priority_queue)
        right_child = heapq.heappop(priority_queue)
        merged_node = Node(frequency=left_child.frequency + right_child.frequency)
        merged_node.left = left_child
        merged_node.right = right_child
        heapq.heappush(priority_queue, merged_node)

    return priority_queue[0]

def generate_huffman_codes(node, code="", huffman_codes={}):
    if node is not None:
        if node.symbol is not None:
            huffman_codes[node.symbol] = code
        generate_huffman_codes(node.left, code + "0", huffman_codes)
        generate_huffman_codes(node.right, code + "1", huffman_codes)

    return huffman_codes

'''
End of copied code for Huffman coding
'''

def get_density_estimator_codes(p:DensityEstimator, min_x = -50, max_x = 50):
    '''
    Generates Huffman Codes for each PDF in a given DensityEstimator, p.

    Returns:
        huffman_codes: a list of dictionaries containing the Huffman codes for all 
            quanta in p between min_x and max_x
    '''
    huffman_codes = []

    with torch.no_grad():
        
        # This code plots PDFs 
        # num_x = 1001
        # num_channels = p.a[0].shape[0]
        # x = torch.linspace(min_x, max_x, num_x)
        # x_full = x[:, torch.newaxis] @ torch.ones((1, num_channels))
        # all_pdfs = p.forward(x_full.view((num_x, num_channels, 1)))[:, :, 0].T
        # for pdf in all_pdfs:
        #     plt.plot(x.detach(), pdf.detach())
        #     plt.show()



        num_channels = p.a[0].shape[0]
        xs = np.arange(min_x, max_x + 1)
        
        # num_x = num_channels
        # num_channels = p.a[0].shape[0]
        # x = -50 * torch.ones(num_x)
        # pdf = p.forward(x.view((1, num_channels, 1)))[0, :, 0].T
        # plt.plot(np.arange(num_channels), pdf.detach())
        # plt.yscale('log')
        # plt.show()

        # assert(False)

        Ps = []

        def f(x):
            x = torch.full((1, num_channels, 1), x)
            ps = p.forward(x).detach()
            ps = ps[0, :, 0]
            return ps
            
        for x in tqdm(xs):
            Ps.append(quad_vec(f, x-0.5, x+0.5, epsabs=1e-5, epsrel=1e-3)[0])
        
        #each row in Ps contains PMF values
        Ps = np.array(Ps).T

        print("\nGenerating Huffman codes. Progress:")

        for pmf in tqdm(Ps):
            root = build_huffman_tree(xs, pmf)
            huffman_codes.append(generate_huffman_codes(root))
        
    return huffman_codes
        

def get_binary_representation(x, huffman_codes, min_x = -50, max_x = 50):
    '''
    Returns the binary encoding for some flattened image x using given huffman codes.
    '''
    bstr = ''
    for i, val in enumerate(x):
        key = int(val)
        codes_dict = huffman_codes[i]
        if key >= max_x or key < min_x:
            val = codes_dict[max_x]
            bstr += val
            if key < min_x:
                key += min_x
            else:
                key -= max_x
            #now we Goloumb-Exponential code key
            b = bin(key)
            if key < 0:
                b = "0" * (len(b)-2) + "11" + b[3:]
            else:
                b = "0" * (len(b)-1) + "10" + b[2:]
            bstr += b
        else:
            val = codes_dict[key]
            bstr += val
    return bstr


def get_integer_representation(bstr, huffman_codes, min_x = -50, max_x = 50):
    '''
    Returns the integer decoding for some flattened image x using given huffman codes.
    Currently does not account for Goloumb-Exponential encoding part
    '''
    x = torch.zeros((len(huffman_codes),))
    idx = 0
    for k, codes in enumerate(huffman_codes):
        code_found = False
        i = 1
        while not code_found:
            bstr_part = bstr[idx:idx+i]
            for key, value in codes.items():
                if value == bstr_part:
                    code_found = True
                    idx += i
                    x[k] = key
                    i = 0
                    break
            i += 1

    return x



# def encode_cabac(bstr):
#     #copied from CABAC python demo file

#     # Create an encoder
#     enc = cabac.cabacTraceEncoder()
#     enc.initCtx([(p1_init, shift_idx)])
#     enc.start()

#     # Encode the bits
#     for i, bit in enumerate(bstr):
#         ctx = 0
#         enc.encodeBin(bit, ctx)
#     enc.encodeBinTrm(1)
#     enc.finish()
#     enc.writeByteAlignment()

#     # Get the bitstream
#     bs = enc.getBitstream()

#     return bs


# def decode_cabac(bstr):
#     #copied from CABAC python demo file

#     # Decode in order to ensure that we get a enc/dec match
#     decoded_bits = ''
#     dec = cabac.cabacDecoder(bstr)
#     dec.initCtx([(p1_init, shift_idx)])
#     dec.start()

#     for i in range(0, len(bstr)):
#         ctx = 0
#         decoded_bit = dec.decodeBin(ctx)
#         decoded_bits += decoded_bit

#     dec.decodeBinTrm()
#     dec.finish()

#     return decoded_bits




if __name__ == "__main__":
    main()

