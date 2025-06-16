import torch

# Define paths to project dependencies 
# User-defined
project_path = ""
code_path = ""
coco_folder_path = ""
figure_path = ""
save_path = ""

# Get working device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#small value
epsilon = 1e-8


# Setup CABAC
p1_init = 0.5  # Initial value for p1
shift_idx = 13   # change to e.g. 0 to see the effect of faster adaptation

# Number of bits used for representing the probabilities in VTM-CABAC
PROB_BITS = 15  # Don't change this
