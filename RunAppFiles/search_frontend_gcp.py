import math
from collections import defaultdict,Counter
from inverted_index_gcp import MultiFileReader, InvertedIndex
from flask import Flask, request, jsonify
import json
import string
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

# create stopwords set
nltk.download('stopwords')
ALL_STOPWORDS = frozenset(stopwords.words('english')).union(["category", "references", "also", "external", "links", 
                                                            "may", "first", "see", "history", "people", "one", "two", 
                                                            "part", "thumb", "including", "second", "following", 
                                                            "many", "however", "would", "became"])


#create all globals members

PAGE_RANKS = None
DOCS_TITLES = None

PAGE_VIEW = None
MIN_TF_IDF_THRESH = 0.45
BUCKET_NAME = "ir-project-bucket-313191645-201013310"  
PAGE_VIEWS_ROUTE = "page_views/pageviews-202108-user.pkl"  
PAGE_RANKS_ROUTE = "json_files/page_rank.json"   
DOCS_TITLES_ROUTE = "json_files/doc_titles.json" 

#index types- details
BODY_INDEX = None
BODY_INDEX_ROUTE = "body_index/postings_tfidf_gcp/body_index.pkl"
BODY_INDEX_BINS = "body_index/postings_gcp_body_index"

TITLE_INDEX = None
TITLE_INDEX_ROUTE = "title_index/postings_tfidf_gcp/title_index.pkl"
TITLE_INDEX_BINS = "title_index/postings_gcp_title_index"

ANCHOR_INDEX = None
ANCHOR_INDEX_ROUTE = "anchor_index/postings_tfidf_gcp/anchor_index.pkl"
ANCHOR_INDEX_BINS = "anchor_index/postings_gcp_anchor_index"

# getters functions
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
    global BODY_INDEX
    global BUCKET_NAME
    global ANCHOR_INDEX_ROUTE
    if ANCHOR_INDEX == None:
        ANCHOR_INDEX = InvertedIndex.read_index(BUCKET_NAME, ANCHOR_INDEX_ROUTE)
    return ANCHOR_INDEX


def get_page_view():
    global PAGE_VIEW
    global BUCKET_NAME
    global PAGE_VIEW_ROUTE
    if PAGE_VIEW == None:
        PAGE_VIEW = InvertedIndex.read_page_views(BUCKET_NAME, PAGE_VIEWS_ROUTE)
    return PAGE_VIEW


def get_docs_page_ranks():
    global PAGE_RANKS
    global PAGE_RANKS_ROUTE
    if PAGE_RANKS == None:
        return get_json_file(PAGE_RANKS_ROUTE)
    return PAGE_RANKS

def get_docs_titles():
    global DOCS_TITLES
    global DOCS_TITLES_ROUTE
    if DOCS_TITLES == None:
        return get_json_file(DOCS_TITLES_ROUTE)
    return DOCS_TITLES

def get_json_file(file_route):
    return InvertedIndex.read_json(file_route, BUCKET_NAME)

#helpers functions

def our_sum(lst):
    summ = 0
    for i in lst:
        summ += i
    return summ


def get_all_doc_titles_relevants(index, route_dir, query_term):
    queries_posting_lists = {}
    set_of_relevant_docs = set()
    for term in query_term:
        post_list = index.read_posting_list(term, route_dir)
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
    return [(doc_id, doc_title[str(doc_id)]) for doc_id, score in sorted(doc_score.items(), key=lambda x: x[1], reverse=True)]

def split_clean_up_query(query, index):
    global ALL_STOPWORDS
    return [term for term in query.translate(str.maketrans('', '', string.punctuation)).lower().split() if (term not in ALL_STOPWORDS) and term in index.df.keys()]

# all search functions

@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    query = query.lower()
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    
    # END SOLUTION
    return jsonify(res)

@app.route("/search_body")
def search_body():
    # TODO: change code, no tf idf calc needed, consider tfidf thresh
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the
        staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    # create query vector
    body_index = get_body_index()
    terms = split_clean_up_query(query, body_index)
    q_vec = Counter(terms)
    queries_posting_lists = {}  # key = tem, value = dict( k = doc_id, v = tf_idf)
    set_of_relevant_docs = set()
    for term in q_vec.keys():
        # get posting list of the term
        post_list = body_index.read_posting_list(term, BODY_INDEX_BINS)
        queries_posting_lists[term] = dict(post_list)
        # ger all relevant docs
        for doc_id, tf_idf in post_list:
            if tf_idf >= MIN_TF_IDF_THRESH:
                set_of_relevant_docs.add(doc_id)
    # calculate query vector size
    q_size = our_sum([freq ** 2 for term, freq in q_vec.items()]) ** 0.5
    # calculate cos_sim for each doc from relevant docs
    doc_cos_sim_dict = {}
    for doc_id in set_of_relevant_docs:
        d_vec = dict()
        # get doc vectors
        for term in q_vec.keys():
            score = 0
            if str(doc_id) in queries_posting_lists[term].keys():
                score = queries_posting_lists[term][str(doc_id)]
            d_vec[term] = score/(q_size*10000.0)
        cos_sim = our_sum([score*q_vec[term] for term, score in d_vec.items()])
        doc_cos_sim_dict[doc_id] = cos_sim
    number_of_relevant_doc = min(100, len(set_of_relevant_docs))
    sorted_lst_of_doc_cos_sim = sorted(doc_cos_sim_dict.items(), key=lambda x: x[1], reverse=True)[:number_of_relevant_doc]
    doc_title_dict = get_docs_titles()
    res = [(doc_id, doc_title_dict[str(doc_id)]) for doc_id, cos_sim in sorted_lst_of_doc_cos_sim]
    # END SOLUTION
    return jsonify(res)

@app.route("/search_title")
def search_title(query):
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF
        QUERY WORDS that appear in the title. For example, a document with a
        title that matches two of the query words will be ranked before a
        document with a title that matches only one query term.

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    title_index = get_title_index()
    terms = split_clean_up_query(query, titley_index)
    # create query vector
    query_term = set(terms)
    set_of_relevant_docs = set()
    res = get_all_doc_titles_relevants(title_index, TITLE_INDEX_BINS, query_term)
    # END SOLUTION
    print(res)
    return jsonify(res)

@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        For example, a document with a anchor text that matches two of the 
        query words will be ranked before a document with anchor text that 
        matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    anchor_index = get_anchor_index()
    terms = split_clean_up_query(query, anchor_index)
    # create query vector
    query_term = set(terms)
    set_of_relevant_docs = set()
    
    res = get_all_doc_titles_relevants(anchor_index, ANCHOR_INDEX_BINS, query_term)
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    docs_page_ranks = get_docs_page_ranks()
    for doc_id in wiki_ids:
        rank = -1
        if str(doc_id) in docs_page_ranks:
            rank = docs_page_ranks[str(doc_id)]
        res.append(rank)
    # END SOLUTION
    print(res)
    return jsonify(res)

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    page_views = get_page_view()
    for doc_id in wiki_ids:
        views = -1
        if str(doc_id) in page_views:
            views = page_views[str(doc_id)]
        res.append(views)
    # END SOLUTION
    print(res)
    return jsonify(res)

if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)