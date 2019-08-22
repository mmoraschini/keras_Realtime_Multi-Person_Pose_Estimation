import argparse
import time

import cv2

from config_reader import read_config
from processing import extract_parts
from output import draw, init_out_file, append_to_out_file
from model.cmu_model import get_testing_model

output_file = './out.txt'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='input image')
    parser.add_argument('--output', type=str, default='result.png', help='output image')
    parser.add_argument('--model', type=str, default='model/keras/model.h5', help='path to the weights file')

    args = parser.parse_args()
    image_path = args.image
    output = args.output
    keras_weights_file = args.model

    tic = time.time()
    print('start processing...')

    # load model

    # authors of original model don't use
    # vgg normalization (subtracting mean) on input images
    model = get_testing_model()
    model.load_weights(keras_weights_file)

    # load config
    params, model_params = read_config()
    
    input_image = cv2.imread(image_path)  # B,G,R order
    
    subsets, candidates = extract_parts(input_image, params, model, model_params)
    canvas = draw(input_image, subsets, candidates)

    init_out_file(output_file)
    append_to_out_file(output_file, subsets, candidates)
    
    toc = time.time()
    print('processing time is %.5f' % (toc - tic))

    cv2.imwrite(output, canvas)

    cv2.destroyAllWindows()
