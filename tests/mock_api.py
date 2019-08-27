# from injector import inject
from connection_test import MockProvider


provider = MockProvider([{'str': 'a', 'num': 0}, {'str': 'b', 'num': 1}, {'str': 'c', 'num': 2}])


def get() -> list:
    return provider.get()


def post(payload=None) -> [None, int]:
    return provider.post(payload), 201

# @inject
# def search(data_provider: MockProvider) -> list:
#     return data_provider.get()
#
#
# @inject
# def post(data_provider: MockProvider, payload=None):
#     data_provider.post(payload)
