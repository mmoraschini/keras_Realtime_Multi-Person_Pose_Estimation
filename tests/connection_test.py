import threading

import requests
import connexion
from connexion.resolver import RestyResolver
from connexion.decorators import validation
# from injector import Binder, CallableProvider
# from flask_injector import FlaskInjector
#
# from injector import inject, provider, Module


class MockProvider(object):
    def __init__(self, items: list=[]):
        self._items = items

    def get(self) -> list:
        return self._items

    def post(self, item):
        self._items.append(item)


# class MockProvider(Module):
#     @inject
#     def __init__(self, items: list=[]):
#         self._items = items
#
#     @provider
#     def get(self) -> list:
#         return self._items
#
#     def post(self, item):
#         self._items.append(item)
#
#
# def _configure(binder: Binder) -> Binder:
#     binder.bind(
#         interface=MockProvider,
#         to=MockProvider(items=[{'str': 'a', 'num': 0}, {'str': 'b', 'num': 1}, {'str': 'c', 'num': 2}])
#     )
#
#     return binder


def connection_test():

    app = connexion.FlaskApp(__name__, specification_dir='./')
    app.add_api('mock_service.yaml', resolver=RestyResolver('api'), strict_validation=True,
                validator_map={'body': validation.RequestBodyValidator})
    # FlaskInjector(app=app.app, modules=[_configure])
    app.run(port=9090)


if __name__ == '__main__':
    service_thread = threading.Thread(target=connection_test)
    service_thread.daemon = True
    service_thread.start()

    quit_program = False
    while not quit_program:
        print('Let\'s add values to the app!')
        print('Insert the string:')
        s = input()
        print('Insert the integer:')
        i = int(input())

        if s == 'exit':
            exit(0)

        # api-endpoint
        URL = "http://127.0.0.1:9090/v1.0/data/add"

        # defining a params dict for the parameters to be sent to the API
        PARAMS = {"str": s, "num": i}

        headers = {'content-type': 'application/json'}
        # sending get request and saving the response as response object
        r = requests.post(url=URL, headers=headers, json=PARAMS)
