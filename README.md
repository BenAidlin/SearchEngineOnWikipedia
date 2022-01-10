# SearchEngineOnWikipedia
Search engine project for IR course 2022, Shahar Kramer and Ben Aidlin.

### Intro
During our 3rd year in the Information and Software System Engineering studies, we took a course at Information Retrieval.
In that course, we gained knowledge regarding different retrival, indexing, crawling and evaluation techiniques,
which resulted in creating the following project. The project ran during 2 days on the gcp (Google cloud) compute engine, and answered queries through 5 different url's - retriving pages from the entire english wikipedia corpus. 

### Content
- IndexBuilderGcp - Contains .ipynb files for creating 3 different indexes we used for retrival, including body, anchor and title index builders. Alongside with the 3 main indexes, there were also json files containing wiki page id's to titles mapping, and also a page rank calculation for each wiki page.
  - create_title_index.ipynb
  - create_body_index_page_rank_doc_title.ipynb
  - create_anchor_index.ipynb
- RunAppFiles - Contains all files neeeded to deploy the code into Google cloud compute engine, along with the RESTfull API supporting the requests, the virtual machine startup script. There are another 3 files provided to help examine the results with the "queris_train.json" file containing queries alongside their top wiki pages, and 2 .ipynb files that run the application and test the results.
  - run_frontend_in_gcp.sh 
  - startup_script_gcp.sh
  - search_frontend_gcp.py
  - helper_functions.py
  - testing startup and app run.ipynb
  - examinate search results.ipynb

### Capabillities
Through the engines end points, you can retrieve information using 5 different techniques:
- Search: retrive information with a query, use both body and title index (0.65-0.35 ratio of results favoring the body). Also include page rank and page view consideration in return order.
- Search body: retrive information only through the wiki page body. Use tf-idf measure for comparrison, with a tf-idf thresh of over 0.45 per term.
- Serach title: retrive information only through the wiki page title. Use a binary ranking of terms existing or not in the title. More terms in title get priorotize.
- Search anchor: retrive information only through the wiki page title. Use a binary ranking of terms existing or not in the title. More terms in title get priorotize.
- Get pageview: retrive a specific wiki page amount of views.
- Get pagerank:  retrive a specific wiki page rank.

### Endpoints
Our search engine supports 5 different requests, including:
- [GET] request, route: /search. Insert your query through the 'query' parameter.
- [GET] request, route: /search_body. Insert your query through the 'query' parameter.
- [GET] request, route: /search_anchor. Insert your query through the 'query' parameter.
- [POST] request, route: /get_pageview. Insert wiki id's in the body of the request in a parameter named 'json'.
- [POST] request, route: /get_pagerank. Insert wiki id's in the body of the request in a parameter named 'json'.

![image](https://user-images.githubusercontent.com/74815296/148820407-131e9453-bba0-4ccf-8f9e-e0c63c2170c4.png)

### Evaluation
We evaluated our engine using MAP@40. The results reached an average of 0.55 at the submission of the project.
Retrival time was 0.737 on average.

![image](https://user-images.githubusercontent.com/74815296/148820529-6de58eeb-72ac-4646-b6b8-b004317ab8ec.png)




### References
- Python packages - including pickle, json, nltk and flask.
- Google storage, a link to our project bucket: https://console.cloud.google.com/storage/browser/ir-project-bucket-313191645-201013310
- Virtual machine external IP: http://34.66.225.79:8080 , can be activated and queried through /search?query=YOUR_QUERY ! Email us for activating the VM.

Ben Aidlin: benaid@post.bgu.ac.il
Shahar Kramer: kramers@post.bgu.ac.il
