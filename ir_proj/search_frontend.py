import math
from collections import defaultdict,Counter
from flask import Flask, request, jsonify
from IndexBuilderGCP.inverted_index_gcp import MultiFileReader

class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


MIN_TF_IDF_THRESH = 0.45
BASE_DIR = ""  # TODO: ...
INDEX_NAME = ""  # TODO: ...

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
    terms = query.split()
    # create query vector
    q_vec = Counter(terms)
    body_index = MultiFileReader.read_index(BASE_DIR, INDEX_NAME)
    queries_posting_lists = {}
    doc_vectors_by_query = {}
    set_of_relevant_docs = set()
    for term in q_vec.keys():
        # get posting list of the term
        post_list = body_index.read_posting_list(term)
        queries_posting_lists[term] = dict(post_list)
        # ger all relevant docs
        for doc_id, freq in post_list:
            set_of_relevant_docs.add(doc_id)
    number_of_relevant_doc = min(100, len(set_of_relevant_docs))
    # create doc vector for all relevant docs
    for doc_id in set_of_relevant_docs:
        vec = {}
        for term, post_lst in queries_posting_lists.items():
            tf_idf = 0
            if doc_id in post_lst:
                tf = post_lst[doc_id] / body_index.doc_len[doc_id]
                idf = math.log(body_index.N/body_index.df[term], 2)
                tf_idf = tf*idf
            vec[term] = tf_idf
        doc_vectors_by_query[doc_id] = vec
    cos_sim_lst = []
    q_size = sum([i**2 for i in q_vec.elements()])**0.5
    for doc_id, d_vec in doc_vectors_by_query.items():
        to_add = ((doc_id, f"title of {doc_id}"), cosine_similarity(d_vec, q_vec, body_index.doc_len[doc_id], q_size)) # TODO: replace body_index.doc_len[doc_id] to body_index.doc_vector_size[doc_id]
        cos_sim_lst.append(to_add)
    res = [x[0] for x in sorted(cos_sim_lst, lambda x: x[1], reverse=True)][:number_of_relevant_doc]
    # END SOLUTION
    return jsonify(res)

def cosine_similarity(d_vec, q_vec, d_size, q_size):
    dot_mul = sum([d_vec[term]*tfidf for term, tfidf in q_vec.keys()])
    return dot_mul/(d_size*q_size)

@app.route("/search_title")
def search_title():
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

    # END SOLUTION
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

    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
