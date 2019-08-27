from services.providers import SubsetsProvider


provider = SubsetsProvider()


def get() -> list:
    return provider.get()


def post(payload) -> [None, int]:
    return provider.post(payload), 201
