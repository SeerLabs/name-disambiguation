import math
import re
import unicodedata

from author import Author
from util import config, utils
from parse_error import NameParseError


class QueryAuthor(Author):
    """
    Contains metadata for each Pubmed Author
    inherited Author
    """

    def __init__(self, input):
        if input is not None:
            name = input.get("name")
            self.set_name(name.strip())

            if input.get("affil") is not None:
                self.affil = input.get('affil').lower().strip()
                if self.affil is None:
                    self.affil = ""
            else:
                self.affil = ""

            self.order = input.get('order')
            self.affil_bow = dict()

            if config.CALC_BOW:
                self.calc_bow_vectors()

    def calc_bow_vectors(self):
        if len(self.affil) > 0:
            aff = utils.remove_punc(self.affil)
            aff_list = [token for token in aff.split() if token not in utils.STOP_WORDS]
            for token in aff_list:
                weight = utils.AFFIL_IDF_DICT.get(token)
                if weight is None:
                    weight = 0.0

                if self.affil_bow.get(token) is None:
                    self.affil_bow[token] = weight
                else:
                    self.affil_bow[token] += weight

            if len(self.affil_bow) > 0:
                # normalize vector
                total = math.sqrt(sum([self.affil_bow[term] ** 2 for term in self.affil_bow]))
                if total > 0:
                    self.affil_bow = {k: v / total for k, v in self.affil_bow.iteritems()}

    def set_name(self, name):
        curname = re.sub("Ph\\.? D\\.?", "", name)
        curname = unicodedata.normalize('NFKD', curname).encode('ASCII', 'ignore')
        curname = re.sub(" -[iI]ng", "", curname)
        curname = re.sub(" \\(.*$", "", curname)
        curname = re.sub("[\\.\\/\\(\\)]", "", curname)
        curname = curname.replace(" [a-zA-Z]$", "")

        match = re.search("^ *(((Dr|dr|Prof|prof|Investigators|Supervisor|" +
                          "Supervisor Dr) )*)?([^ ]+)( (.+))?( ([^ ]+))$", curname)

        if match:
            # print match.groups()

            self.firstName = match.group(4).strip()
            if self.firstName is None:
                self.firstName = ""
            if match.group(6) is not None:
                self.middleName = match.group(6).strip()
            else:
                if self.firstName is not None and len(self.firstName) == 2 and self.firstName[1] >= 'A' and \
                        self.firstName[1] <= 'Z':
                    self.middleName = self.firstName[1:2]
                    self.firstName = self.firstName[0:1]
                else:
                    self.middleName = ""
            self.lastName = match.group(8).strip()
            if self.lastName is None:
                self.lastName = ""
            if self.middleName is None:
                self.middleName = ""
        else:
            raise NameParseError("Could not parse name " + name)

    def get_name(self):
        name = self.firstName
        if len(self.middleName) > 0:
            name += " " + self.middleName
        name += " " + self.lastName
        return name

    def get_first_middle(self):
        return self.firstName + "_" + self.middleName


