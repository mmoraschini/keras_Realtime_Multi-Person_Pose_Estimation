import argparse
import time

import cv2
import requests
from config_reader import read_config
from processing import extract_parts
from output import draw
from model.cmu_model import get_testing_model


def _crop(image, w, f):
    return image[:, int(w * f): int(w * (1 - f))]


def _prepare_object(subsets, candidates):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    obj = {'timestamp': ts}
    for i, line in enumerate(subsets):
        id_str = 'id{}'.format(i)
        obj[id_str] = {'parts': []}
        for candidate_id in line[:-2]:
            candidate_id = int(candidate_id)

            if candidate_id != -1:
                part_info = candidates[candidate_id, :]
            else:
                part_info = [-1, -1, -1]

            obj[id_str]['parts'].append({'x': part_info[0],
                                         'y': part_info[1],
                                         'score': part_info[2]})
        obj[id_str]['overall_score'] = line[-2]
        obj[id_str]['n_valid_parts'] = line[-1]

    return obj


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0, help='ID of the device to open')
    parser.add_argument('--model', type=str, default='model/keras/model.h5', help='path to the weights file')
    parser.add_argument('--frame_rate', type=int, default=None,
                        help='desired frame rate (depends on processing speed and camera)')
    # --process_speed changes how many scales the model analyzes each frame at
    parser.add_argument('--process_speed', type=int, default=1,
                        help='Int 1 (fastest, lowest quality) to 4 (slowest, highest quality)')
    parser.add_argument('--service_host', type=str, default='http://localhost',
                        help='name of the host where the service is running')
    parser.add_argument('--service_port', type=int, default=9090, help='port on the host where the service is running')
    parser.add_argument('--post_path', type=str, default=None, required=True,
                        help='path to the html POST endpoint')
    parser.add_argument('--save_mp4_path', type=str, default=None,
                        help='if defined saves the output to the specified path')
    parser.add_argument('--mirror', type=bool, default=True, help='whether to mirror the camera')

    args = parser.parse_args()
    device = args.device
    keras_weights_file = args.model
    frame_rate = args.frame_rate
    process_speed = args.process_speed
    host = args.service_host
    port = args.service_port
    post_path = args.post_path
    out_mp4_path = args.save_mp4_path
    mirror = args.mirror

    # load model
    # The authors of the original model don't use vgg normalization (subtracting mean) on input images
    model = get_testing_model()
    model.load_weights(keras_weights_file)

    # load config
    params, model_params = read_config()

    # Video reader
    cam = cv2.VideoCapture(device)
    # CV_CAP_PROP_FPS
    cam.set(cv2.CAP_PROP_FPS, frame_rate)
    input_fps = cam.get(cv2.CAP_PROP_FPS)
    print("Running at {} fps.".format(input_fps))

    ret_val, orig_image = cam.read()

    width = orig_image.shape[1]
    height = orig_image.shape[0]
    factor = 0.3

    out = None
    # Output mp4
    if out_mp4_path is not None and ret_val is not None:
        # Video writer
        output_fps = input_fps

        tmp = _crop(orig_image, width, factor)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_mp4_path, fourcc, output_fps, (tmp.shape[1], tmp.shape[0]))

        del tmp

    scale_search = [0.22, 0.25, .5, 1, 1.5, 2]  # [.5, 1, 1.5, 2]
    scale_search = scale_search[0:process_speed]

    params['scale_search'] = scale_search

    service_url = host + ':' + str(port) + '/' + post_path

    i = 0  # default is 0
    resize_fac = 8
    # while(cam.isOpened()) and ret_val is True:
    while True:

        cv2.waitKey(10)

        if cam.isOpened() is False or ret_val is False:
            break

        if mirror:
            orig_image = cv2.flip(orig_image, 1)

        cropped = _crop(orig_image, width, factor)

        input_image = cv2.resize(cropped, (0, 0), fx=1 / resize_fac, fy=1 / resize_fac, interpolation=cv2.INTER_CUBIC)

        input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)

        # generate image with body parts
        subsets, candidates = extract_parts(input_image, params, model, model_params)
        obj = _prepare_object(subsets, candidates)
        print(obj)

        canvas = draw(cropped, subsets, candidates, resize_fac=resize_fac)

        headers = {'content-type': 'application/json'}
        r = requests.post(url=service_url, headers=headers, json=obj)

        print('Service POST response: {}'.format(r))

        if out is not None:
            out.write(canvas)

        cv2.imshow('frame', canvas)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        ret_val, orig_image = cam.read()

        i += 1
