
session1 = [('python', 7.946138143539429, 0.168), ('data science', 7.868347883224487, 0.091), ('migraine', 7.820008039474487, 0.234),
            ('chocolate', 7.9894118309021, 0.159), ('how to make pasta', 0.14537549018859863, None), ('Does pasta have preservatives?', 0.09855914115905762, None),
            ('how google works', 0.0984957218170166, None), ('what is information retrieval', 0.09599184989929199, None), ('NBA', 0.11116623878479004, None),
            ('yoga', 7.770323753356934, 0.522), ('how to not kill plants', 0.10992264747619629, None), ('masks', 7.782826900482178, 0.216),
            ('black friday', 8.134378671646118, 0.0), ('why do men have nipples', 0.09868407249450684, None), ('rubber duck', 7.961754322052002, 0.0),
            ('michelin', 7.934900760650635, 0.143), ('what to watch', 0.1155393123626709, None), ('best marvel movie', 7.940219402313232, 0.063),
            ('how tall is the eiffel tower', 0.11460757255554199, None), ('where does vanilla flavoring come from', 0.09833073616027832, None),
            ('best ice cream flavour', 8.125947713851929, 0.09), ('how to tie a tie', 0.11406922340393066, None), ('how to earn money online', 0.09995436668395996, None),
            ('what is critical race theory', 0.0996241569519043, None), ('what space movie was made in 1992', 0.09851598739624023, None), ('how to vote', 0.10283184051513672, None),
            ('google trends', 7.901169776916504, 0.143), ('dim sum', 7.812884569168091, 0.102), ('ted', 7.8077898025512695, 0.029), ('fairy tale', 7.889822721481323, 0.113)]


####### clean up query -> remove from query stopwords, make query lowercase, handle unexistant terms in the index

session2 = [('python', 12.294742822647095, 0.168), ('data science', 7.939196348190308, 0.091), ('migraine', 7.624596118927002, 0.234), ('chocolate', 7.866066217422485, 0.159),
            ('how to make pasta', 7.86054253578186, 0.277), ('Does pasta have preservatives?', 7.693466901779175, 0.0), ('how google works', 7.893348693847656, 0.227),
            ('what is information retrieval', 7.770978212356567, 0.258), ('NBA', 7.712854862213135, 0.135), ('yoga', 7.680114507675171, 0.522),
            ('how to not kill plants', 7.916727781295776, 0.111), ('masks', 7.842602491378784, 0.216), ('black friday', 7.76887321472168, 0.0),
            ('why do men have nipples', 7.8874571323394775, 0.078), ('rubber duck', 7.925971508026123, 0.0), ('michelin', 7.7022483348846436, 0.143),
            ('what to watch', 7.773510217666626, 0.067), ('best marvel movie', 7.8394997119903564, 0.063), ('how tall is the eiffel tower', 8.07307481765747, 0.236),
            ('where does vanilla flavoring come from', 8.160261869430542, 0.069), ('best ice cream flavour', 8.065742015838623, 0.09), ('how to tie a tie', 7.746157169342041, 0.378),
            ('how to earn money online', 8.048466920852661, 0.083), ('what is critical race theory', 7.924978971481323, 0.0),
            ('what space movie was made in 1992', 8.310042381286621, 0.036), ('how to vote', 8.201421737670898, 0.215), ('google trends', 7.699093818664551, 0.143),
            ('dim sum', 7.745569705963135, 0.102), ('ted', 7.671856641769409, 0.029), ('fairy tale', 7.837843656539917, 0.113)]

# MAP@40: 0.14143333333333336
# AVG run time: 8.015910243988037