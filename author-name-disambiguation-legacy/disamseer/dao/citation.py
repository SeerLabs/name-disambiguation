class Citation(object):
    """
    Contains basic metadata for each document
    """
    def __init__(self):
        self.id = ""
        self.title = ""
        self.authors = []
        self.keywords = set()
        self.year = 0
        self.venue = ""

    def get_id(self):
        return self.id

    def get_title(self):
        return self.title

    def get_year(self):
        return self.year

    def get_venue(self):
        return self.venue

    def set_id(self, id):
        self.id = id

    def set_title(self, title):
        self.title = title

    def set_year(self, year):
        self.year = year

    def set_venue(self, venue):
        self.venue = venue

    def add_author(self, author):
        self.authors.append(author)

    def add_keyword(self, keyword):
        self.keywords.add(keyword)

    def get_authors(self):
        return self.authors

    def get_num_authors(self):
        return len(self.authors)

    def get_keywords(self):
        return self.keywords

    def get_author_by_id(self, id):
        for author in self.authors:
            if author.get_id() == id:
                return author
        return None

    def get_shared_authors(self, citation, curname):
        names = set()
        shared_names = set()
        for author in self.authors:
            if author.get_last_name().lower().strip() == curname:
                continue
            names.add(author.get_last_name().lower().strip())

        for author in citation.authors:
            if author.get_last_name().lower().strip() == curname:
                continue
            this_name = author.get_last_name().lower().strip()
            if this_name in names:
                shared_names.add(this_name)
        return shared_names

    def get_shared_keywords(self, citation):
        target_keywords = citation.get_keywords()
        shared_keywords = set()
        for keyword in target_keywords:
            if keyword in self.keywords:
                shared_keywords.add(keyword)
        return shared_keywords

    def is_year_compatible(self, citation):
        if self.get_year() > 500 and citation.get_year() > 500 and abs(self.get_year() - citation.get_year()) > 10:
            return False
        else:
            return True

