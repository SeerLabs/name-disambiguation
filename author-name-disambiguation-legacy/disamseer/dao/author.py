import re
from parse_error import NameParseError

class Author(object):
    """
    contains metadata for each Author
    """

    def __init__(self):
        # do nothing for now
        self.id = 0
        self.lastName = ""
        self.firstName = ""
        self.middleName =""
        self.affil = ""
        #self.addr = ""
        #self.email = ""
        self.order = 0

    def get_name(self):
        name = self.firstName
        if self.middleName:
            name += " " + self.middleName
        name += " " + self.lastName
        return name

    def get_last_name(self):
        return self.lastName

    def get_first_name(self):
        return self.firstName

    def get_middle_name(self):
        return self.middleName

    def get_last_and_first_init(self):
        if not self.firstName:
            return self.lastName
        else:
            return self.lastName + ", " + self.firstName[:1]

    def get_last_and_first(self):
        if not self.firstName:
            return self.lastName
        else:
            return self.lastName + ", " + self.firstName

    def get_id(self):
        return self.id

    def get_affil(self):
        return self.affil

    def get_order(self):
        return self.order

    def set_id(self, id):
        self.id = id

    def set_affil(self, affil):
        self.affil = affil

    def set_order(self, order):
        self.order = order

    def set_name(self, name):
        match = re.search("^([^ ]+)( ([^ ]+))?( ([^ ]+))$", name)
        if match:
            self.firstName = match.group(1).strip()
            self.middleName = match.group(3).strip()
            self.lastName = match.group(5).strip()
        else:
            raise NameParseError("Could not parse name " + name)

    def is_middle_name_compatible(self, middle_name):
        m = self.get_middle_name().lower()
        m2 = middle_name.lower()

        if len(m) == 0 or len(m2) == 0:
            return True
        # at least one of them has middle name, check if they all have full
        elif len(m) > 1 and len(m2) > 1:
            if m == m2:
                return True
            else:
                return False
        # at least one of them has middle name, one of them is not full,
        # so we have to check only the initial
        else:
            if m[0] == m2[0]:
                return True
            else:
                return False

    def is_first_name_compatible(self, first_name):
        f = self.get_first_name().lower()
        f2 = first_name.lower()

        # if it has only initial, they should be same because of blocking
        if len(f) > 1 and len(f2) > 1 and f != f2:
            return False
        else:
            return True

    def is_compatible(self, auth):
        if self.is_middle_name_compatible(auth.get_middle_name()) and \
                self.is_first_name_compatible(auth.get_first_name()):
            return True
        else:
            return False
