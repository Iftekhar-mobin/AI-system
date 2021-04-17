from main import SeqGen

# initialization for data+library loading and mapping
driver = SeqGen()

# To generate only one sequence
driver.horizontal_seq_gen(
    input_sequence=[2, 4, 1, 7],
    space_range={'min': 0, 'max': 5},
    image_width=200
)

# To generate bulk without augmenting source dataset
driver.generate_random_sequence(
    num_samples=10,
    seq_len=10,
    space_range={'min': 0, 'max': 10},
    image_width=200
)

# Data augmentation and saving
driver.augment_dataset(
    zca_whitening=False,
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
)
