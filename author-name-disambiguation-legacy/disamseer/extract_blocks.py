import os
import sys

import MySQLdb
import MySQLdb.cursors
import src.disamseer.config

from disamseer.dao import NameParseError
from disamseer.dao import PubmedAuthor


def run(dirpath):
    # connect db
    db = MySQLdb.connect(src.disamseer.config.DB_HOST, src.disamseer.config.DB_USER, src.disamseer.config.DB_PWD,
                         src.disamseer.config.DB_NAME, charset='utf8', use_unicode=True,
                         port=src.disamseer.config.DB_PORT)
    cursor = MySQLdb.cursors.SSCursor(db)

    query = "SELECT id, firstname, lastname, affiliation, ord FROM authors"

    cursor.execute(query)
    result = cursor.fetchone()

    blocks = dict()

    count = 0
    while result is not None:
        count += 1
        if count % 10000 == 0:
            print count
            #break
        try:
            author = PubmedAuthor(result)
            last_name = author.get_last_name().lower()
            first_name = author.get_first_name().lower()

            if len(last_name) and len(first_name):
                last_init = last_name[0].upper()
                first_init = first_name[0].upper()

                if 'A'<=last_init<='Z' and 'A'<=first_init<='Z':
                    key = last_init + first_init + "/" + last_name + "_" + \
                          first_init.lower() + ".txt"

                    if key in blocks:
                        blocks.get(key).append(author.get_id())
                    else:
                        blocks[key] = [author.get_id()]
        except NameParseError as e:
            result = cursor.fetchone()
            continue

        result = cursor.fetchone()

    #print blocks

    print "making directories.."
    # make directories
    for i in range(ord('A'), ord('Z')+1):
        for j in range(ord('A'), ord('Z')+1):
            if not os.path.exists(dirpath + '/' + chr(i) + chr(j)):
                os.makedirs(dirpath + '/' + chr(i) + chr(j))

    print "write block files..."
    # write block files
    for key in blocks:
        block = blocks.get(key)
        if len(block) > 1:
            with open(dirpath + "/" + key, 'w+') as fp:
                for aid in block:
                    fp.write(str(aid) + '\n')
    

    print "close db.."
    cursor.close()
    db.close()

if __name__ == '__main__':
    run(sys.argv[1])
