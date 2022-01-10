import math
from collections import defaultdict,Counter
from inverted_index_gcp import MultiFileReader, InvertedIndex
import json
import string
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords


###################---> create all globals members <---###################
# create stopwords set

nltk.download('stopwords')
ALL_STOPWORDS = frozenset(stopwords.words('english')).union(["category", "references", "also", "external", "links", 
                                                            "may", "first", "see", "history", "people", "one", "two", 
                                                            "part", "thumb", "including", "second", "following", 
                                                            "many", "however", "would", "became"])
# resources
PAGE_RANKS = None
DOCS_TITLES = None

PAGE_VIEW = None
MIN_TF_IDF_THRESH = 0.45
BUCKET_NAME = "ir-project-bucket-313191645-201013310"  
PAGE_VIEWS_ROUTE = "page_views/pageviews-202108-user.pkl"  
PAGE_RANKS_ROUTE = "json_files/page_rank.json"   
DOCS_TITLES_ROUTE = "json_files/doc_titles.json" 

# index types- details
BODY_INDEX = None
BODY_INDEX_ROUTE = "body_index/postings_tfidf_gcp/body_index.pkl"
BODY_INDEX_BINS = "body_index/postings_gcp_body_index"

TITLE_INDEX = None
TITLE_INDEX_ROUTE = "title_index/postings_tfidf_gcp/title_index.pkl"
TITLE_INDEX_BINS = "title_index/postings_gcp_title_index"

ANCHOR_INDEX = None
ANCHOR_INDEX_ROUTE = "anchor_index/postings_tfidf_gcp/anchor_index.pkl"
ANCHOR_INDEX_BINS = "anchor_index/postings_gcp_anchor_index"


###################---> all getters functions <---###################

# indexes
def get_body_index():
    global BODY_INDEX
    global BUCKET_NAME
    global BODY_INDEX_ROUTE
    if BODY_INDEX == None:
        BODY_INDEX = InvertedIndex.read_index(BUCKET_NAME, BODY_INDEX_ROUTE)
    return BODY_INDEX

def get_title_index():
    global TITLE_INDEX
    global BUCKET_NAME
    global TITLE_INDEX_ROUTE
    if TITLE_INDEX == None:
        TITLE_INDEX = InvertedIndex.read_index(BUCKET_NAME, TITLE_INDEX_ROUTE)
    return TITLE_INDEX

def get_anchor_index():
    global ANCHOR_INDEX
    global BUCKET_NAME
    global ANCHOR_INDEX_ROUTE
    if ANCHOR_INDEX == None:
        ANCHOR_INDEX = InvertedIndex.read_index(BUCKET_NAME, ANCHOR_INDEX_ROUTE)
    return ANCHOR_INDEX

# resources
def get_page_view():
    global PAGE_VIEW
    global BUCKET_NAME
    global PAGE_VIEW_ROUTE
    if PAGE_VIEW == None:
        PAGE_VIEW = InvertedIndex.read_page_views(BUCKET_NAME, PAGE_VIEWS_ROUTE)
    return PAGE_VIEW


def get_docs_page_ranks_dict():
    global PAGE_RANKS
    global PAGE_RANKS_ROUTE
    if PAGE_RANKS == None:
        PAGE_RANKS = get_json_file(PAGE_RANKS_ROUTE)
    return PAGE_RANKS


def get_docs_titles():
    global DOCS_TITLES
    global DOCS_TITLES_ROUTE
    if DOCS_TITLES == None:
        DOCS_TITLES = get_json_file(DOCS_TITLES_ROUTE)
    return DOCS_TITLES


def get_json_file(file_route):
    return InvertedIndex.read_json(file_route, BUCKET_NAME)



###################---> helper functions <---###################
def split_clean_up_query(query, index):
    global ALL_STOPWORDS
    return [term for term in query.translate(str.maketrans('', '', string.punctuation)).lower().split() if (term not in ALL_STOPWORDS) and term in index.df.keys()]


def our_sum(lst):
    summ = 0
    for i in lst:
        summ += i
    return summ


###################---> search functions <---###################

def helper_get_pagerank(wiki_ids):
    res = []
    # BEGIN SOLUTION
    docs_page_ranks = get_docs_page_ranks_dict()
    for doc_id in wiki_ids:
        rank = 0
        try:
            rank = docs_page_ranks[str(doc_id)]
        except:
            pass
        res.append(rank)
    # END SOLUTION
    return res


def helper_get_pageview(wiki_ids):
    res = []
    # BEGIN SOLUTION
    page_views = get_page_view()
    for doc_id in wiki_ids:
        doc_id = int(doc_id)
        views = 0
        try:
            views = page_views[doc_id]
        except:
            pass
        res.append(views)
    # END SOLUTION
    return res

def helper_search(query):
    res = []
    # BEGIN SOLUTION
    res_title = helper_search_title(query)
    res_body = helper_search_body(query)
    # add score to all documents in results
    w_body = 0.65
    w_title = 0.35
    size_res_body = len(res_body)
    res_body_with_score_dict = dict([(res_body[i][0], [(size_res_body-i)*w_body/size_res_body, res_body[i][1]]) for i in range(size_res_body)])
    res_title = res_title[:min(100, len(res_title))]
    size_res_title = len(res_title)
    res_title_with_score_dict = dict([(res_title[i][0], [(size_res_title-i)*w_title/size_res_title, res_title[i][1]]) for i in range(size_res_title)])
    # merge 2 dicts into title dict
    
    for doc_id, details in res_body_with_score_dict.items():
        body_score = details[0]
        try:
            title_score = res_title_with_score_dict[doc_id][0]
            res_title_with_score_dict[doc_id][0] = body_score + title_score
        except:
            res_title_with_score_dict[doc_id] = details
    
    w_pv = 0.1
    w_pr = 0.1
    pr_dict = get_docs_page_ranks_dict()
    pv_dict = get_page_view()
    for doc_id, details in res_title_with_score_dict.items():
        pr_score = 0
        pv_score = 0
        try:
            pr_score = pr_dict[str(doc_id)]
            pv_score = pv_dict[doc_id]
        except:
            pass
        res_title_with_score_dict[doc_id][0] += pr_score*w_pr + pv_score*w_pv + res_title_with_score_dict[doc_id][0]*(1-w_pv-w_pr)
    
    # END SOLUTION
    return [(doc_id, details[1]) for doc_id, details in sorted(res_title_with_score_dict.items(), key=lambda x: x[1][0], reverse=True)][:min(100, len(res_title_with_score_dict))]

def calculate_doc_id_title_tuples(sorted_lst_docs):
    doc_title = get_docs_titles()
    res = []
    for doc_id in sorted_lst_docs:
        try:
            res.append((doc_id, doc_title[str(doc_id)]))
        except:
            pass
    return res


def helper_search_body(query):
    index = get_body_index()
    return non_binary_search(query, index, BODY_INDEX_BINS)

def helper_search_title(query):
    # END SOLUTION
    index = get_title_index()
    return non_binary_search(query, index, TITLE_INDEX_BINS)


def helper_binary_search_title(query):
    res = []
    # BEGIN SOLUTION
    title_index = get_title_index()
    res = binary_search(title_index, TITLE_INDEX_BINS, query)
    # END SOLUTION
    return res


def helper_binary_search_anchor(query):
    res = []
    # BEGIN SOLUTION
    anchor_index = get_anchor_index()
    res = binary_search(anchor_index, ANCHOR_INDEX_BINS, query)
    # END SOLUTION
    return res


def binary_search(index, route_dir, query):
    terms = split_clean_up_query(query, index)
    # create query vector
    query_term = set(terms)
    queries_posting_lists = {}
    set_of_relevant_docs = set()
    for term in query_term:
        try:
            post_list = index.read_posting_list(term, route_dir)
        except:
            post_list = {}
        queries_posting_lists[term] = dict(post_list)
        for doc_id, tf_idf in post_list:
            set_of_relevant_docs.add(doc_id)
    doc_score = {}
    for doc_id in set_of_relevant_docs:
        score = 0
        for term in queries_posting_lists.keys():
            if str(doc_id) in queries_posting_lists[term].keys():
                score += 1
        doc_score[doc_id] = score
    doc_title = get_docs_titles()
    sorted_scores = sorted(doc_score.items(), key=lambda x: x[1], reverse=True)
    to_ret = []
    # transform the outcomes to tuples of doc id and doc title
    for doc_id, score in sorted_scores:
        try:
            to_ret.append((doc_id, doc_title[str(doc_id)]))
        except:
            pass
    return to_ret


def non_binary_search(query, index, bins):
    res = []
    # BEGIN SOLUTION
    # create query vector
    terms = split_clean_up_query(query, index)
    q_vec = Counter(terms)
    queries_posting_lists = {}  # key = tem, value = dict( k = doc_id, v = tf_idf)
    dict_of_relevant_docs = Counter()
    for term in q_vec.keys():
        # get posting list of the term
        try:
            post_list = index.read_posting_list(term, bins)
        except:
            post_list = {}
        queries_posting_lists[term] = dict(post_list)
        # ger all relevant docs
        for doc_id, tf_idf in post_list:
            if tf_idf >= MIN_TF_IDF_THRESH:
                dict_of_relevant_docs[doc_id] += 1
    # calculate query vector size
    q_size = our_sum([freq ** 2 for term, freq in q_vec.items()]) ** 0.5
    # calculate cos_sim for each doc from relevant docs
    doc_cos_sim_dict = {}
    for doc_id in dict_of_relevant_docs.keys():
        d_vec = dict()
        # get doc vectors 
        for term in q_vec.keys():
            score = 0
            try:
                score = queries_posting_lists[term][doc_id]
            except:
                pass
            d_vec[term] = score/(q_size*10000.0)
        cos_sim = our_sum([score*q_vec[term] for term, score in d_vec.items()])
        doc_cos_sim_dict[doc_id] = cos_sim + dict_of_relevant_docs[doc_id]/len(q_vec)
    number_of_relevant_doc = min(100, len(dict_of_relevant_docs.keys()))
    sorted_lst_of_doc_cos_sim = sorted(doc_cos_sim_dict.items(), key=lambda x: x[1], reverse=True)[:number_of_relevant_doc]
    doc_title_dict = get_docs_titles()
    res = [(doc_id, doc_title_dict[str(doc_id)]) for doc_id, cos_sim in sorted_lst_of_doc_cos_sim]
    # END SOLUTION
    return res

def wrap_helper_search_body(query, lst):
    lst = helper_search_body(query)

def wrap_helper_search_title(query, lst):
    lst = helper_search_title(query)
