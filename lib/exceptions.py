
class NoSorter(Exception):
    def __init__(self, message="Unable to find lego sorter. Is it connected?"):
        self.message = message
        super().__init__(self.message)


class NotConnectedToSorter(Exception):
    def __init__(self, message="Sorter disconnected. Connect it first"):
        self.message = message
        super().__init__(self.message)


class NotImplemented(Exception):
    def __init__(self, message="This method is not implemented. Please review your code"):
        self.message = message
        super().__init__(self.message)

    @staticmethod
    def raise_this():
        raise NotImplemented()
