class AuthorPoint(object):
    """
    Data Structure contains author, citation pair
    """
    def __init__(self, id, author, doc):
        self.id = id
        self.author = author
        self.doc = doc

    def get_author(self):
        return self.author

    def get_doc(self):
        return self.doc

    def get_id(self):
        return self.id
