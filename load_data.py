"""
Loads data from file 'data/nnrccar.features' and saves output in variables: images, labels
"""

import numpy as np

def load_data(features_file):
    """
    features_file contains image pixel data for 3862 176x144 images and corresponding labels.
    Lines in the file alternate between a label and its corresponding data.

    Set of unique labels: {(0, 0, 1, 0), (0, 1, 1, 0), (1, 0, 1, 0)}
    Label mapping:
      (0, 0, 0, 1) --> 0 (reverse)
      (0, 0, 1, 0) --> 1 (forward)
      (0, 1, 1, 0) --> 2 (left)
      (1, 0, 1, 0) --> 3 (right)
    Label format: (right, left, forward, reverse)
    """

    imgs = np.zeros((3862, 176, 144))
    lbls = np.zeros(3862)
    forward_label = np.array([0., 0., 1., 0.])
    left_label = np.array([0., 1., 1., 0.])
    right_label = np.array([1., 0., 1., 0.])

    with open(features_file, 'r') as features:
        for idx, line in enumerate(features):

            line_arr = np.fromstring(line, sep=' ')

            if (idx % 2) == 0: # label
                if np.array_equal(line_arr[1:], forward_label):
                    lbls[idx//2] = 1
                elif np.array_equal(line_arr[1:], left_label):
                    lbls[idx//2] = 2
                elif np.array_equal(line_arr[1:], right_label):
                    lbls[idx//2] = 3
                else:
                    print("Label '{}' not recognized".format(line_arr[1:]))

            else: # image pixel data
                image = line_arr[0:25344] # 176 x 144 = 25344
                # correct image for easier viewing
                image = (np.reshape(image, (144, 176)).transpose() * (-1))
                imgs[idx//2] = image

    return imgs, lbls

images, labels = load_data('data/nnrccar.features')
