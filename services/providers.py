class SubsetsProvider(object):
    def __init__(self, items: list = []):
        self._items = items

    def post(self, item):
        self._items.append(item)

    def get(self) -> list:
        """
        Returns all subsets it has
        :return: a list of subsets
        """
        if not self._items:
            return []

        return self._items
