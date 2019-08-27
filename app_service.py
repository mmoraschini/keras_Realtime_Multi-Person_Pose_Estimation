import argparse
import time

import connexion
from connexion.resolver import RestyResolver
from connexion.decorators import validation


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=9090, help='Port to open')

    args = parser.parse_args()
    port = args.port

    print('Service started at {}'.format(time.strftime('%Y-%m-%d %H:%M:%S')))

    app = connexion.App(__name__, specification_dir='swagger/')
    app.add_api('camera_service.yaml', resolver=RestyResolver('api'), strict_validation=True,
                validator_map={'body': validation.RequestBodyValidator})
    app.run(port=port)
