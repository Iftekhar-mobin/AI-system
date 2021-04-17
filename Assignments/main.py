from constants import IMAGE_PATH, \
    LABEL_PATH, \
    IMAGE_SAVE_DIR, \
    SPACING_RANGE, \
    GEN_DATASET_NAME
from methods import load_data, \
    label_mapping, \
    generate_image, \
    save_image, \
    random_seq_generator, \
    augmented_data_generator


class SeqGen:
    def __init__(self):
        # Load data API
        self.dataset_images, self.labels = load_data(IMAGE_PATH, LABEL_PATH)
        # Generate mapping Image => class Labels
        self.label_maps = label_mapping(self.labels)

    def horizontal_seq_gen(self, input_sequence, space_range=SPACING_RANGE, image_width=100):
        # Generate Horizontal image with optimal spacing
        horizontal_image_array = generate_image(self.dataset_images, self.label_maps,
                                                input_sequence, space_range, image_width)
        save_image(horizontal_image_array, input_sequence, IMAGE_SAVE_DIR)

    def generate_random_sequence(self, num_samples, seq_len, space_range=SPACING_RANGE, image_width=100):
        random_seq_generator(num_samples, seq_len, self.dataset_images, self.label_maps,
                             space_range, image_width, IMAGE_SAVE_DIR, GEN_DATASET_NAME)

    def augment_dataset(self,
                        zca_whitening=False,
                        rotation_range=10,
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        shear_range=0.2,
                        ):
        augmented_data_generator(
            self.dataset_images,
            self.labels,
            IMAGE_SAVE_DIR,
            f_center=False,
            s_center=False,
            f_std_normalization=False,
            s_std_normalization=False,
            zca_whitening=zca_whitening,
            zca_epsilon=1e-6,
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            brightness_range=None,
            shear_range=shear_range,
            zoom_range=0,
            channel_shift_range=0.,
            fill_mode='nearest',
            c_val=0.,
            horizontal_flip=False,
            vertical_flip=False,
            rescale=0.0001,
            preprocessing_function=None,
            validation_split=0.0,
        )
