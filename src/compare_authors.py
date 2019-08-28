import numpy as np
import shutil
from tqdm import tqdm
from scipy.special import comb
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity
from src.utility_functions import cleanName
from collections import Counter



def compareAuthors(args):
    key,tag,a, b, str_algorithm = args
    name_a = a["name"].split(" ")
    name_b = b["name"].split(" ")
    initials_a = [x[0] for x in name_a]
    initials_b = [x[0] for x in name_b]
    len_name_a = len(name_a)
    len_name_b = len(name_b)
    full_name_count = 2
    if len_name_a >= full_name_count and len_name_b >= full_name_count:
        first_name_score = str_algorithm(name_a[0], name_b[0])
        last_name_score = str_algorithm(name_a[-1], name_b[-1])
        if len_name_a > full_name_count and len_name_b > full_name_count:
            middle_name_score = str_algorithm(" ".join(name_a[1:-1]), " ".join(name_b[1:-1]))
        elif len_name_a == full_name_count and len_name_b == full_name_count:
            middle_name_score = 1
        else:
            middle_name_score = 0
    else:
        # They both only have 1 name
        if len_name_a == len_name_b:
            first_name_score = 1
            middle_name_score = 1
            last_name_score = str_algorithm(name_a[0], name_b[0])
        else:
            first_name_score = 0
            middle_name_score = 0
            last_name_score = str_algorithm(name_a[-1], name_b[-1])
    shared_initials = 0
    len_ia = len(initials_a)
    len_ib = len(initials_b)
    for i in range(min(len_ia,len_ib)):
        if initials_a[i] == initials_b[i]:
            shared_initials+=1
    initials_score = shared_initials * min(len_ia,len_ib) / float(max(len_ia,len_ib))
    address_a = a["address"]
    address_b = b["address"]
    address_keys = ["postCode","settlement","country"]
    address_scores = []
    for k in address_keys:
        if k in address_a and k in address_b:
            address_scores.append(str_algorithm(address_a[k],address_b[k]))
        else:
            address_scores.append(0)
    if not a["aff_name"] or not b["aff_name"]:
        org_name_score = 0
    else:
        org_name_score = str_algorithm(a["aff_name"],b["aff_name"])

    org_type_score = 1 if a["aff_type"] == b["aff_type"] else 0
    email_domain_score = 0
    if a["email_domain"] and b["email_domain"]:
        email_domain_score = str_algorithm(a["email_domain"],b["email_domain"])
    email_user_score = 0
    if a["email_user"] and b["email_user"]:
        email_user_score = str_algorithm(a["email_user"],b["email_user"])

    co_auth_score = sum([1 for x in a["co_authors_name"] if x in b["co_authors_name"]])
    department_score = 0
    if a["department"] and b["department"]:
        department_score = sum([1 for x in a["department"] if x in b["department"]])
    same_title_words = sum([1 for x in a["title_tokenized"] if x in b["title_tokenized"]])
    out = [first_name_score,middle_name_score,last_name_score,initials_score,org_name_score,org_type_score,email_domain_score,
           email_user_score,co_auth_score,department_score,same_title_words]
    out.extend(address_scores)
    return key,tag,np.asarray(out)