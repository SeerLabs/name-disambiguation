import os
import re
import sys
from multiprocessing import Pool

import MySQLdb
import config

from dao.pubmed_author import PubmedAuthor
from dao.pubmed_doc import PubmedDoc

NOISE_CLUSTER = "[noise]"
TYPE_NOISE = 1
TYPE_SINGLETON = 2
OUTPUT_FILE_NAME = 'output_import.txt'

fp = None


def helper_func(path):
    run(path)

def parse_disambiguate_file(path):
    clusters = {}
    with open(path) as fp:
        cid = ''
        for line in fp:
            cur_line = line.strip()
            if cur_line[0] == '[':
                cid = cur_line
                clusters[cid] = []
            else:
                clusters.get(cid).append(int(cur_line))
    return clusters

INSERT_CANNAME = "INSERT INTO cannames (canname, fname, mname, lname, affil, affil2, affil3) VALUES (%s,%s,%s,%s,%s,%s,%s);";

def create_canname(auth, cursor):
    name = auth.get_name()
    fname = auth.get_first_name()
    mname = auth.get_middle_name()
    lname = auth.get_last_name()
    
    if len(name) == 0:
        name = None
    if len(fname) == 0:
        fname = None
    if len(mname) == 0:
        mname = None
    if len(lname) == 0:
        lname = None

    query_string = (name, fname, mname, lname, None, None, None)
    #print INSERT_CANNAME
    #print query_string
    cursor.execute(INSERT_CANNAME, query_string)
    cur_id = cursor.lastrowid
    return cur_id

UPDATE_PREV_CID = "update authors set cluster = 0 where cluster = %s"

def update_prev_authors(cid, cursor):
    cursor.execute(UPDATE_PREV_CID % (cid))

UPDATE_AUTHOR_CID = "update authors set cluster = %s where id = %s"

def update_author_cluster(aid, cid, cursor):
    query_string = (cid, aid)
    cursor.execute(UPDATE_AUTHOR_CID, query_string)

SELECT_AUTHOR_CID = "SELECT a.id, a.firstname, a.lastname, a.affiliation, a.ord, year(p.date) as pyear FROM authors as a JOIN papers as p ON a.paperid=p.pmid WHERE a.cluster = %s ORDER BY pyear"

def cleanup_affil(affil):
    affil = affil.replace("Department", "Dept.")
    affil = affil.replace("Univ. ", "University ")
    affil = affil.replace(" & ", " and ")
    affil = re.sub("\\b([a-z0-9],)*[a-z0-9]\\b","", affil)
    affil = re.sub("( ;)*","", affil)
    affil = re.sub("\\B;\\B","",affil)
    affil = re.sub(" +", " ",affil).strip()
    return affil

UPDATE_CANNAME = "UPDATE cannames SET canname=%s, fname=%s, mname=%s, lname=%s, affil=%s, affil2=%s, "\
        "affil3=%s where id=%s"

def update_canname_info(cid, cursor):
    cursor.execute(SELECT_AUTHOR_CID % (cid))
    year_of = {}
    fname_count = {}
    mname_count = {}
    lname_count = {}
    canname_count = {}
    affil_count = {}

    authors = cursor.fetchall()
    for auth in authors:
        author = PubmedAuthor(auth)
        fname = author.get_first_name()
        if len(fname) > 0:
            if fname_count.get(fname) is None:
                fname_count[fname] = 0
            fname_count[fname] += 1

        mname = author.get_middle_name()
        if len(mname) > 0:
            if mname_count.get(mname) is None:
                mname_count[mname] = 0
            mname_count[mname] += 1

        lname = author.get_last_name()
        if len(lname) > 0:
            if lname_count.get(lname) is None:
                lname_count[lname] = 0
            lname_count[lname] += 1

        canname = author.get_name()
        if len(canname) > 0:
            if canname_count.get(canname) is None:
                canname_count[canname] = 0
            canname_count[canname] += 1


        affil_str = author.get_affil()
        if affil_str is not None and len(affil_str) > 0:
            affil_str = cleanup_affil(author.get_affil())
            affil_str = affil_str.replace(";",',')
            if len(affil_str) > 0:
                if affil_count.get(affil_str) is None:
                    affil_count[affil_str] = 0
                affil_count[affil_str] += 1

        if len(affil_str) > 0:
            if auth[5] is not None:
                year = int(auth[5])
                if year > 2020 or year < 1500:
                    year = 0
            else:
                year = 0
            
            year_of[affil_str] = year
    
    selected_fname = None
    selected_mname = None
    selected_lname = None
    selected_canname = None

    if len(fname_count) > 0:
        selected_fname = sorted(fname_count, key=fname_count.__getitem__, reverse=True)[0]
        if len(selected_fname)==0:
            selected_fname = None
    if len(mname_count) > 0:
        selected_mname = sorted(mname_count, key=mname_count.__getitem__, reverse=True)[0]
        if len(selected_mname)==0:
            selected_mname = None
    if len(lname_count) > 0:
        selected_lname = sorted(lname_count, key=lname_count.__getitem__, reverse=True)[0]
        if len(selected_lname)==0:
            selected_lname = None
    if len(canname_count) > 0:
        selected_canname = sorted(canname_count, key=canname_count.__getitem__, reverse=True)[0]
        if len(selected_canname)==0:
            selected_canname = None

    selected_affils = []
    if len(affil_count)> 0:
        sorted_affil = sorted(affil_count, key=affil_count.__getitem__, reverse=True)

        for affil in sorted_affil:
            dup = False
            s1 = affil.replace(",", "")
            for selected_affil in selected_affils:
                s2 = selected_affil.replace(",", "")
                if s1 in s2 or s2 in s1:
                    dup = True
                    break
            if not dup:
                selected_affils.append(affil)
                if len(selected_affils) >= 3:
                    break

    selected_affil1 = None
    selected_affil2 = None
    selected_affil3 = None

    if len(selected_affils) > 0:
        selected_affil1 = selected_affils[0]
        if len(selected_affils) > 1:
            selected_affil2 = selected_affils[1]
            if len(selected_affils) > 2:
                selected_affil3 = selected_affils[2]

    selected_values = [selected_canname, selected_fname, selected_mname, selected_lname, 
            selected_affil1, selected_affil2, selected_affil3, cid]
    cursor.execute(UPDATE_CANNAME, selected_values)

def run(path):
    # connect db
    #print path
    db = MySQLdb.connect(src.disamseer.config.DB_HOST, src.disamseer.config.DB_USER, src.disamseer.config.DB_PWD,
                         src.disamseer.config.DB_NAME, charset='utf8', use_unicode=True,
                         port=src.disamseer.config.DB_PORT)
    cursor = db.cursor()

    import_file_list = []

    for dirpath, dirs, files in os.walk(path):
        for importfile in files:
            import_file_list.append(os.path.join(path, importfile))

    cnt = 0
    #for import_file in tqdm(import_file_list):
    for import_file in import_file_list:
        cnt += 1
        #print "importing " + import_file
        clusters = parse_disambiguate_file(import_file)
        for key in clusters:
            #print "processing cluster " + key
            aids = clusters.get(key)
            cid = TYPE_NOISE
            if key != NOISE_CLUSTER:
                doc = PubmedDoc(cursor, aids[0], False)
                auth = doc.get_author_by_id(aids[0])

                if auth is None:
                    fp.write(import_file + " Error : Warning author is null. This should not happen on correct database, ignoring cluster. aid: " + str(aids.get(0)) + '\n')
                    continue
                
                cid = create_canname(auth, cursor)
                if cid == 0:
                    fp.write(import_file + " Error : create_canname returned cid = 0 for author aid " + str(aids[0]) + '\n')
                    db.rollback()
                else:
                    db.commit()
                    #update_prev_authors(cid, cursor)
                    #db.commit()
        
            for aid in aids:
                update_author_cluster(aid, cid, cursor)
                db.commit()

            if cid != TYPE_NOISE:
                update_canname_info(cid, cursor)
                db.commit()
    with open(OUTPUT_FILE_NAME, 'a') as fp:
        fp.write(path + " : " + str(cnt) + " / " + str(len(import_file_list)) + '\n')

def run_from_lists(infile, nthreads):
    pathes = list()

    with open(infile) as fp:
        for line in fp:
            path = line.strip()
            pathes.append(path)

    pool = Pool(nthreads)
    pool.map(helper_func, pathes)

if __name__ == '__main__':
    #run(sys.argv[1])
    #fp = open(OUTPUT_FILE_NAME, 'w')
    run_from_lists(sys.argv[1], int(sys.argv[2]))
    #fp.close()
    #pathes = list()
    #with open(sys.argv[1]) as lfp:
    #    for line in lfp:
    #        path = line.strip()
    #        pathes.append(path)

    #for path in pathes:
    #    run(path)
