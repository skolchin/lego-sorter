
class NoSorter(Exception):
    def __init__(self, message="Unable to find lego sorter. Is it connected?"):
        self.message = message
        super().__init__(self.message)


class NotConnectedToSorter(Exception):
    def __init__(self, message="Sorter disconnected. Connect it first"):
        self.message = message
        super().__init__(self.message)
