import os
# Define the constant parameters
DATASET_DIR = os.path.join(os.getcwd(), 'data')
IMAGE_FILE = 't10k-images.idx3-ubyte'
LABEL_FILE = 't10k-labels.idx1-ubyte'
IMAGE_PATH = os.path.join(DATASET_DIR, IMAGE_FILE)
LABEL_PATH = os.path.join(DATASET_DIR, LABEL_FILE)
IMAGE_SAVE_DIR = os.path.join(os.getcwd(), 'aug_gen_data')
SPACING_RANGE = {'min': 0, 'max': 10}
GEN_DATASET_NAME = 'augmented_data'

# spacing = 10
# all_images = x
# image_height = 28
# input_sequence = [3, 8, 9, 1, 5]
#
#
# num_samples = 7
# seq_len = 5
# dataset_images = x
# label_idex_mapping = label_mapping(y)
# spacing = 10


