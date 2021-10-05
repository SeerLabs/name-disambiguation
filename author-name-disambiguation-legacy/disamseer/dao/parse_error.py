class NameParseError(Exception):
    """
    Simple error class to distinguish name parse error
    """
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)