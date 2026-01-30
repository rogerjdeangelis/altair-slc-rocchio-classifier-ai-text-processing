# altair-slc-rocchio-classifier-ai-text-processing
Altair slc rocchio classifier ai text processing
    %let pgm=altair-slc-rocchio-classifier-ai-text-processing;

    %stop_submission;

    Altair slc rocchio classifier ai text processing

    Too long to post here,see github
    https://github.com/rogerjdeangelis/altair-slc-rocchio-classifier-ai-text-processing

    Response to these
    https://community.altair.com/discussion/44579
    https://community.altair.com/discussion/48900

    PROBLEM Categorize tasks as either programming or AI

     TASKS
       neural networks are powerful
       coding in python is enjoyable
       java code runs everywhere
       deep neural networks learn

     TRAINING DATA

      ADD Category                             Category

      machine learning is great                ai
      deep learning is amazing                 ai
      artificial intelligence is the future    ai
      python programming is fun                programming
      java programming is robust               programming
      c++ programming is fast                  programming

     SOLUTION (ADD CATEGORY TO TRAINING DATA)

               TASK                         prob_ai    prob_programming    prediction

      neural networks are powerful              0.5               0.5      ai
      coding in python is enjoyable    0.4341858128      0.5658141871      programming
      java code runs everywhere         0.428583374       0.571416626      programming
      deep neural networks learn       0.5748150653      0.4251849346      ai


    COMMENT FROM PERPLEXITY?

    The Rocchio classifier is a simple, prototype-based method used in text classification,
    particularly in vector space models like those for information retrieval.
    It computes a centroid (average vector) for each class from the training documents
    feature vectors, such as tf-idf weights, and classifies new documents
    by assigning them to the nearest centroid

    /*                   _
    (_)_ __  _ __  _   _| |_
    | | `_ \| `_ \| | | | __|
    | | | | | |_) | |_| | |_
    |_|_| |_| .__/ \__,_|\__|
            |_|
    */

    /*--- clear work persistent work library ---*/
    proc datasets lib=workx kill;
    run;quit;

    %utlfkil(d:/wpswrkx/final_results.parquet); /*--- export file for python 310 ---*/

    /*--- works ---*/

    options validvarname=v7;
    data workx.training;
        input doc & $100. label  $20.;
    cards4;
    machine learning is great  ai
    deep learning is amazing  ai
    artificial intelligence is the future  ai
    python programming is fun  programming
    java programming is robust  programming
    c++ programming is fast  programming
    ;;;;
    run;

    data workx.testing;
        input doc & $100.;
    cards4;
    neural networks are powerful
    coding in python is enjoyable
    java code runs everywhere
    deep neural networks learn
    ;;;;
    run;

    /*************************************************************************************************************************/
    /*                                                                                                                       */
    /* WORKX.TRAINING total obs=6 30JAN2026:08:42:49                                                                         */
    /*                                                                                                                       */
    /*       DOC                                      LABEL                                                                  */
    /*                                                                                                                       */
    /* 1     machine learning is great                ai                                                                     */
    /* 2     deep learning is amazing                 ai                                                                     */
    /* 3     artificial intelligence is the future    ai                                                                     */
    /* 4     python programming is fun                programming                                                            */
    /* 5     java programming is robust               programming                                                            */
    /* 6     c++ programming is fast                  programming                                                            */
    /*                                                                                                                       */
    /*                                                                                                                       */
    /* WORKX.TESTING total obs=4 30JAN2026:08:43:37                                                                          */
    /*                                                                                                                       */
    /* Obs                 doc                                                                                               */
    /*                                                                                                                       */
    /*  1     neural networks are powerful                                                                                   */
    /*  2     coding in python is enjoyable                                                                                  */
    /*  3     java code runs everywhere                                                                                      */
    /*  4     deep neural networks learn                                                                                     */
    /*                                                                                                                       */
    /*************************************************************************************************************************/

    /*
     _ __  _ __ _ __   ___ ___  ___ ___
    | `_ \| `__| `_ \ / __/ _ \/ __/ __|
    | |_) | |  | |_) | (_|  __/\__ \__ \
    | .__/|_|  | .__/ \___\___||___/___/
    |_|        |_|
    */

    options set=PYTHONHOME "D:\py314";

    proc python;
    submit;
    import pyarrow
    import pandas as pd
    import numpy as np
    import re
    from collections import defaultdict, Counter
    import math
    import pyreadstat as ps
    train_df,meta = ps.read_sas7bdat('d:/wpswrkx/training.sas7bdat')
    test_df,meta = ps.read_sas7bdat('d:/wpswrkx/testing.sas7bdat')
    class SimpleTFIDFVectorizer:
        """A simple TF-IDF vectorizer without scikit-learn dependency."""

        def __init__(self):
            self.vocabulary_ = {}
            self.idf_ = None

        def fit_transform(self, documents):
            # Tokenization - simple regex to get words
            tokenized_docs = []
            for doc in documents:
                # Convert to lowercase and split into words
                words = re.findall(r'\b\w+\b', doc.lower())
                tokenized_docs.append(words)

            # Build vocabulary
            all_words = set()
            for words in tokenized_docs:
                all_words.update(words)

            # Create vocabulary dictionary
            self.vocabulary_ = {word: idx for idx, word in enumerate(sorted(all_words))}
            vocab_size = len(self.vocabulary_)

            # Initialize matrices
            n_docs = len(documents)
            tf_matrix = np.zeros((n_docs, vocab_size))
            doc_freq = np.zeros(vocab_size)

            # Calculate term frequencies
            for i, words in enumerate(tokenized_docs):
                word_counts = Counter(words)
                for word, count in word_counts.items():
                    if word in self.vocabulary_:
                        idx = self.vocabulary_[word]
                        tf_matrix[i, idx] = count

            # Calculate document frequency
            for i in range(vocab_size):
                doc_freq[i] = np.count_nonzero(tf_matrix[:, i])

            # Calculate IDF
            self.idf_ = np.log((n_docs + 1) / (doc_freq + 1)) + 1

            # Calculate TF-IDF
            tfidf_matrix = tf_matrix * self.idf_

            # Normalize (L2 norm)
            for i in range(n_docs):
                norm = np.linalg.norm(tfidf_matrix[i, :])
                if norm > 0:
                    tfidf_matrix[i, :] = tfidf_matrix[i, :] / norm

            return tfidf_matrix

        def transform(self, documents):
            # Tokenize new documents
            tokenized_docs = []
            for doc in documents:
                words = re.findall(r'\b\w+\b', doc.lower())
                tokenized_docs.append(words)

            n_docs = len(documents)
            vocab_size = len(self.vocabulary_)
            tf_matrix = np.zeros((n_docs, vocab_size))

            # Calculate term frequencies
            for i, words in enumerate(tokenized_docs):
                word_counts = Counter(words)
                for word, count in word_counts.items():
                    if word in self.vocabulary_:
                        idx = self.vocabulary_[word]
                        tf_matrix[i, idx] = count

            # Apply IDF
            tfidf_matrix = tf_matrix * self.idf_

            # Normalize
            for i in range(n_docs):
                norm = np.linalg.norm(tfidf_matrix[i, :])
                if norm > 0:
                    tfidf_matrix[i, :] = tfidf_matrix[i, :] / norm

            return tfidf_matrix

    class RocchioClassifier:
        """Rocchio classifier implementation."""

        def __init__(self, metric='cosine'):
            self.metric = metric
            self.prototypes = {}
            self.classes = []
            self.vectorizer = SimpleTFIDFVectorizer()

        def fit(self, X_train, y_train):
            # Convert to numpy arrays if needed
            X_train = np.array(X_train)
            y_train = np.array(y_train)

            # Vectorize documents
            X_train_vec = self.vectorizer.fit_transform(X_train)

            # Get unique classes
            self.classes = np.unique(y_train)

            # Calculate prototype vector for each class
            for class_label in self.classes:
                # Get indices of documents in this class
                mask = y_train == class_label
                class_indices = np.where(mask)[0]

                if len(class_indices) > 0:
                    # Get vectors for this class
                    class_vectors = X_train_vec[class_indices]

                    # Average vectors to create prototype
                    prototype = np.mean(class_vectors, axis=0)
                    self.prototypes[class_label] = prototype
                else:
                    # If no documents for this class, create zero vector
                    vocab_size = len(self.vectorizer.vocabulary_)
                    self.prototypes[class_label] = np.zeros(vocab_size)

            return self

        def predict(self, X_test):
            # Vectorize test documents
            X_test_vec = self.vectorizer.transform(X_test)

            predictions = []

            # For each test document
            for i in range(len(X_test)):
                test_vector = X_test_vec[i]
                best_similarity = -float('inf')
                best_class = None

                # Compare to each prototype
                for class_label, prototype in self.prototypes.items():
                    if self.metric == 'cosine':
                        # Cosine similarity
                        similarity = np.dot(test_vector, prototype)
                        norm_test = np.linalg.norm(test_vector)
                        norm_proto = np.linalg.norm(prototype)

                        if norm_test > 0 and norm_proto > 0:
                            similarity = similarity / (norm_test * norm_proto)
                        else:
                            similarity = 0
                    else:
                        # Euclidean distance (convert to similarity)
                        distance = np.linalg.norm(test_vector - prototype)
                        similarity = -distance

                    # Keep the best match
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_class = class_label

                predictions.append(best_class)

            return predictions

        def predict_proba(self, X_test):
            """Return probability-like scores for each class."""
            X_test_vec = self.vectorizer.transform(X_test)
            n_classes = len(self.classes)
            n_samples = len(X_test)

            scores = np.zeros((n_samples, n_classes))

            for i in range(n_samples):
                test_vector = X_test_vec[i]

                for j, class_label in enumerate(self.classes):
                    prototype = self.prototypes[class_label]

                    if self.metric == 'cosine':
                        similarity = np.dot(test_vector, prototype)
                        norm_test = np.linalg.norm(test_vector)
                        norm_proto = np.linalg.norm(prototype)

                        if norm_test > 0 and norm_proto > 0:
                            similarity = similarity / (norm_test * norm_proto)
                        else:
                            similarity = 0
                    else:
                        distance = np.linalg.norm(test_vector - prototype)
                        similarity = np.exp(-distance)  # Convert to similarity

                    scores[i, j] = similarity

            # Normalize scores to sum to 1 (softmax-like)
            # Add small epsilon to avoid division by zero
            scores_exp = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            proba = scores_exp / (np.sum(scores_exp, axis=1, keepdims=True) + 1e-10)

            return proba

    # Example usage - no external dependencies needed
    print("=" * 60)
    print("Rocchio Classifier Demo")
    print("=" * 60)


    # Sample training data
    documents = train_df['doc'].tolist()
    labels = train_df['label'].tolist()

    # Sample test data
    test_docs = test_df['doc'].tolist()


    print("\nTraining documents:")
    for doc, label in zip(documents, labels):
        print(f"  '{doc}' -> {label}")

    print("\nTest documents:")
    for doc in test_docs:
        print(f"  '{doc}'")

    # Create and train the classifier
    print("\nTraining Rocchio classifier...")
    classifier = RocchioClassifier(metric='cosine')
    classifier.fit(documents, labels)

    print(f"Vocabulary size: {len(classifier.vectorizer.vocabulary_)}")
    print(f"Classes: {list(classifier.classes)}")

    # Make predictions
    print("\nMaking predictions...")
    predictions = classifier.predict(test_docs)

    print("\nPredictions:")
    for doc, pred in zip(test_docs, predictions):
        print(f"  '{doc}' -> {pred}")

    # Get probability scores
    print("\nProbability scores:")
    proba = classifier.predict_proba(test_docs)
    for i, doc in enumerate(test_docs):
        print(f"\n  Document: '{doc}'")
        for j, class_label in enumerate(classifier.classes):
            print(f"    P({class_label}) = {proba[i, j]:.4f}")

    print("\n" + "=" * 60)

    predictions = classifier.predict(test_docs)
    proba = classifier.predict_proba(test_docs)

    # Create DataFrame
    final_results = pd.DataFrame({
        'document': test_docs,
        'prob_ai': proba[:, 0],
        'prob_programming': proba[:, 1],
        'prediction': predictions
    })

    print("\nFinal Results DataFrame:")
    print(final_results.head())

    final_results.to_parquet('d:/wpswrkx/final_results.parquet', engine='pyarrow')
    print("Demo complete!")
    print("=" * 60)
    endsubmit;
    run;

    options set=PYTHONHOME "D:\py310";
    proc python;
    submit;
    import pyarrow
    import pandas as pd
    final_results = pd.read_parquet('d:/wpswrkx/final_results.parquet', engine='pyarrow');
    endsubmit;
    import python=final_results data=workx.final_results;
    run;

    proc print data=workx.final_results;
    run;

    /*           _               _
      ___  _   _| |_ _ __  _   _| |_
     / _ \| | | | __| `_ \| | | | __|
    | (_) | |_| | |_| |_) | |_| | |_
     \___/ \__,_|\__| .__/ \__,_|\__|
                    |_|
    */

    /**************************************************************************************************************************/
    /*                                                                                                                        */
    /* Altair SLC                                                                                                             */
    /*                                                                                                                        */
    /* WORKX.FINAL_RESULTS total obs=4                                                                                        */
    /*                                                 PROB_                                                                  */
    /*            DOCUMENT               PROB_AI    PROGRAMMING    PREDICTION                                                 */
    /*                                                                                                                        */
    /*  neural networks are powerful     0.50000      0.50000      ai                                                         */
    /*  coding in python is enjoyable    0.43419      0.56581      programming                                                */
    /*  java code runs everywhere        0.42858      0.57142      programming                                                */
    /*  deep neural networks learn       0.57482      0.42518      ai                                                         */
    /*                                                                                                                        */
    /* Altair SLC                                                                                                             */
    /* LIST: 8:35:19                                                                                                          */
    /*                                                                                                                        */
    /* The PYTHON Procedure                                                                                                   */
    /*                                                                                                                        */
    /* ============================================================                                                           */
    /*                                                                                                                        */
    /* Rocchio Classifier Demo                                                                                                */
    /*                                                                                                                        */
    /* ============================================================                                                           */
    /*                                                                                                                        */
    /*                                                                                                                        */
    /* Training documents:                                                                                                    */
    /*                                                                                                                        */
    /*   'machine learning is great' -> ai                                                                                    */
    /*                                                                                                                        */
    /*   'deep learning is amazing' -> ai                                                                                     */
    /*                                                                                                                        */
    /*   'artificial intelligence is the future' -> ai                                                                        */
    /*                                                                                                                        */
    /*   'python programming is fun' -> programming                                                                           */
    /*                                                                                                                        */
    /*   'java programming is robust' -> programming                                                                          */
    /*                                                                                                                        */
    /*   'c++ programming is fast' -> programming                                                                             */
    /*                                                                                                                        */
    /*                                                                                                                        */
    /* Test documents:                                                                                                        */
    /*                                                                                                                        */
    /*   'neural networks are powerful'                                                                                       */
    /*                                                                                                                        */
    /*   'coding in python is enjoyable'                                                                                      */
    /*                                                                                                                        */
    /*   'java code runs everywhere'                                                                                          */
    /*                                                                                                                        */
    /*   'deep neural networks learn'                                                                                         */
    /*                                                                                                                        */
    /*                                                                                                                        */
    /* Training Rocchio classifier...                                                                                         */
    /*                                                                                                                        */
    /* Vocabulary size: 17                                                                                                    */
    /*                                                                                                                        */
    /* Classes: [np.str_('ai'), np.str_('programming')]                                                                       */
    /*                                                                                                                        */
    /*                                                                                                                        */
    /* Making predictions...                                                                                                  */
    /*                                                                                                                        */
    /*                                                                                                                        */
    /* Predictions:                                                                                                           */
    /*                                                                                                                        */
    /*   'neural networks are powerful' -> ai                                                                                 */
    /*                                                                                                                        */
    /*   'coding in python is enjoyable' -> programming                                                                       */
    /*                                                                                                                        */
    /*   'java code runs everywhere' -> programming                                                                           */
    /*                                                                                                                        */
    /*   'deep neural networks learn' -> ai                                                                                   */
    /*                                                                                                                        */
    /*                                                                                                                        */
    /* Probability scores:                                                                                                    */
    /*                                                                                                                        */
    /*                                                                                                                        */
    /*   Document: 'neural networks are powerful'                                                                             */
    /*                                                                                                                        */
    /*     P(ai) = 0.5000                                                                                                     */
    /*                                                                                                                        */
    /*     P(programming) = 0.5000                                                                                            */
    /*                                                                                                                        */
    /*                                                                                                                        */
    /*   Document: 'coding in python is enjoyable'                                                                            */
    /*                                                                                                                        */
    /*     P(ai) = 0.4342                                                                                                     */
    /*                                                                                                                        */
    /*     P(programming) = 0.5658                                                                                            */
    /*                                                                                                                        */
    /*                                                                                                                        */
    /*   Document: 'java code runs everywhere'                                                                                */
    /*                                                                                                                        */
    /*     P(ai) = 0.4286                                                                                                     */
    /*                                                                                                                        */
    /*     P(programming) = 0.5714                                                                                            */
    /*                                                                                                                        */
    /*                                                                                                                        */
    /*   Document: 'deep neural networks learn'                                                                               */
    /*                                                                                                                        */
    /*     P(ai) = 0.5748                                                                                                     */
    /*                                                                                                                        */
    /*     P(programming) = 0.4252                                                                                            */
    /*                                                                                                                        */
    /*                                                                                                                        */
    /* ============================================================                                                           */
    /*                                                                                                                        */
    /*                                                                                                                        */
    /* Final Results DataFrame:                                                                                               */
    /*                                                                                                                        */
    /*                         document   prob_ai  prob_programming   prediction                                              */
    /* 0   neural networks are powerful  0.500000          0.500000           ai                                              */
    /* 1  coding in python is enjoyable  0.434186          0.565814  programming                                              */
    /* 2      java code runs everywhere  0.428583          0.571417  programming                                              */
    /* 3     deep neural networks learn  0.574815          0.425185           ai                                              */
    /*                                                                                                                        */
    /* Demo complete!                                                                                                         */
    /*                                                                                                                        */
    /* ============================================================                                                           */
    /**************************************************************************************************************************/

    /*
    | | ___   __ _
    | |/ _ \ / _` |
    | | (_) | (_| |
    |_|\___/ \__, |
             |___/
    */

    1                                          Altair SLC        08:35 Friday, January 30, 2026

    NOTE: Copyright 2002-2025 World Programming, an Altair Company
    NOTE: Altair SLC 2026 (05.26.01.00.000758)
          Licensed to Roger DeAngelis
    NOTE: This session is executing on the X64_WIN11PRO platform and is running in 64 bit mode

    NOTE: AUTOEXEC processing beginning; file is C:\wpsoto\autoexec.sas
    NOTE: AUTOEXEC source line
    1       +  ï»¿ods _all_ close;
               ^
    ERROR: Expected a statement keyword : found "?"
    NOTE: Library workx assigned as follows:
          Engine:        SAS7BDAT
          Physical Name: d:\wpswrkx

    NOTE: Library slchelp assigned as follows:
          Engine:        WPD
          Physical Name: C:\Progra~1\Altair\SLC\2026\sashelp


    LOG:  8:35:19
    NOTE: 1 record was written to file PRINT

    NOTE: The data step took :
          real time : 0.025
          cpu time  : 0.015


    NOTE: AUTOEXEC processing completed


    Altair SLC

    The DATASETS Procedure

             Directory

    Libref           WORKX
    Engine           SAS7BDAT
    Physical Name    d:\wpswrkx

                                      Members

                                 Member
      Number    Member Name      Type         File Size      Date Last Modified

    ---------------------------------------------------------------------------

           1    FINAL_RESULTS    DATA              9216      30JAN2026:08:32:02
           2    TESTING          DATA              9216      30JAN2026:08:32:00
           3    TRAINING         DATA             13312      30JAN2026:08:32:00
    1         /*--- clear work persistent work library ---*/
    2         proc datasets lib=workx kill;
    3         run;quit;
    NOTE: Deleting WORKX.final_results (type=DATA)
    NOTE: Deleting WORKX.testing (type=DATA)
    NOTE: Deleting WORKX.training (type=DATA)
    NOTE: Procedure datasets step took :
          real time : 0.039
          cpu time  : 0.031


    4
    5         %utlfkil(d:/wpswrkx/final_results.parquet); /*--- export file for python 310 ---*/
    6
    7         /*--- works ---*/
    8
    9         options validvarname=v7;
    10        data workx.training;
    11            input doc & $100. label  $20.;
    12        cards4;

    NOTE: Data set "WORKX.training" has 6 observation(s) and 2 variable(s)
    NOTE: The data step took :
          real time : 0.006
          cpu time  : 0.000


    13        machine learning is great  ai
    14        deep learning is amazing  ai
    15        artificial intelligence is the future  ai
    16        python programming is fun  programming
    17        java programming is robust  programming
    18        c++ programming is fast  programming
    19        ;;;;

    2                                                                                                                         Altair SLC

    20        run;
    21
    22        data workx.testing;
    23            input doc & $100.;
    24        cards4;

    NOTE: Data set "WORKX.testing" has 4 observation(s) and 1 variable(s)
    NOTE: The data step took :
          real time : 0.006
          cpu time  : 0.000


    25        neural networks are powerful
    26        coding in python is enjoyable
    27        java code runs everywhere
    28        deep neural networks learn
    29        ;;;;
    30        run;
    31
    32        options set=PYTHONHOME "D:\py314";
    33
    34        proc python;
    35        submit;
    36        import pyarrow
    37        import pandas as pd
    38        import numpy as np
    39        import re
    40        from collections import defaultdict, Counter
    41        import math
    42        import pyreadstat as ps
    43        train_df,meta = ps.read_sas7bdat('d:/wpswrkx/training.sas7bdat')
    44        test_df,meta = ps.read_sas7bdat('d:/wpswrkx/testing.sas7bdat')
    45        class SimpleTFIDFVectorizer:
    46            """A simple TF-IDF vectorizer without scikit-learn dependency."""
    47
    48            def __init__(self):
    49                self.vocabulary_ = {}
    50                self.idf_ = None
    51
    52            def fit_transform(self, documents):
    53                # Tokenization - simple regex to get words
    54                tokenized_docs = []
    55                for doc in documents:
    56                    # Convert to lowercase and split into words
    57                    words = re.findall(r'\b\w+\b', doc.lower())
    58                    tokenized_docs.append(words)
    59
    60                # Build vocabulary
    61                all_words = set()
    62                for words in tokenized_docs:
    63                    all_words.update(words)
    64
    65                # Create vocabulary dictionary
    66                self.vocabulary_ = {word: idx for idx, word in enumerate(sorted(all_words))}
    67                vocab_size = len(self.vocabulary_)
    68
    69                # Initialize matrices
    70                n_docs = len(documents)
    71                tf_matrix = np.zeros((n_docs, vocab_size))
    72                doc_freq = np.zeros(vocab_size)
    73
    74                # Calculate term frequencies
    75                for i, words in enumerate(tokenized_docs):

    3                                                                                                                         Altair SLC

    76                    word_counts = Counter(words)
    77                    for word, count in word_counts.items():
    78                        if word in self.vocabulary_:
    79                            idx = self.vocabulary_[word]
    80                            tf_matrix[i, idx] = count
    81
    82                # Calculate document frequency
    83                for i in range(vocab_size):
    84                    doc_freq[i] = np.count_nonzero(tf_matrix[:, i])
    85
    86                # Calculate IDF
    87                self.idf_ = np.log((n_docs + 1) / (doc_freq + 1)) + 1
    88
    89                # Calculate TF-IDF
    90                tfidf_matrix = tf_matrix * self.idf_
    91
    92                # Normalize (L2 norm)
    93                for i in range(n_docs):
    94                    norm = np.linalg.norm(tfidf_matrix[i, :])
    95                    if norm > 0:
    96                        tfidf_matrix[i, :] = tfidf_matrix[i, :] / norm
    97
    98                return tfidf_matrix
    99
    100           def transform(self, documents):
    101               # Tokenize new documents
    102               tokenized_docs = []
    103               for doc in documents:
    104                   words = re.findall(r'\b\w+\b', doc.lower())
    105                   tokenized_docs.append(words)
    106
    107               n_docs = len(documents)
    108               vocab_size = len(self.vocabulary_)
    109               tf_matrix = np.zeros((n_docs, vocab_size))
    110
    111               # Calculate term frequencies
    112               for i, words in enumerate(tokenized_docs):
    113                   word_counts = Counter(words)
    114                   for word, count in word_counts.items():
    115                       if word in self.vocabulary_:
    116                           idx = self.vocabulary_[word]
    117                           tf_matrix[i, idx] = count
    118
    119               # Apply IDF
    120               tfidf_matrix = tf_matrix * self.idf_
    121
    122               # Normalize
    123               for i in range(n_docs):
    124                   norm = np.linalg.norm(tfidf_matrix[i, :])
    125                   if norm > 0:
    126                       tfidf_matrix[i, :] = tfidf_matrix[i, :] / norm
    127
    128               return tfidf_matrix
    129
    130       class RocchioClassifier:
    131           """Rocchio classifier implementation."""
    132
    133           def __init__(self, metric='cosine'):
    134               self.metric = metric
    135               self.prototypes = {}
    136               self.classes = []
    137               self.vectorizer = SimpleTFIDFVectorizer()
    138

    4                                                                                                                         Altair SLC

    139           def fit(self, X_train, y_train):
    140               # Convert to numpy arrays if needed
    141               X_train = np.array(X_train)
    142               y_train = np.array(y_train)
    143
    144               # Vectorize documents
    145               X_train_vec = self.vectorizer.fit_transform(X_train)
    146
    147               # Get unique classes
    148               self.classes = np.unique(y_train)
    149
    150               # Calculate prototype vector for each class
    151               for class_label in self.classes:
    152                   # Get indices of documents in this class
    153                   mask = y_train == class_label
    154                   class_indices = np.where(mask)[0]
    155
    156                   if len(class_indices) > 0:
    157                       # Get vectors for this class
    158                       class_vectors = X_train_vec[class_indices]
    159
    160                       # Average vectors to create prototype
    161                       prototype = np.mean(class_vectors, axis=0)
    162                       self.prototypes[class_label] = prototype
    163                   else:
    164                       # If no documents for this class, create zero vector
    165                       vocab_size = len(self.vectorizer.vocabulary_)
    166                       self.prototypes[class_label] = np.zeros(vocab_size)
    167
    168               return self
    169
    170           def predict(self, X_test):
    171               # Vectorize test documents
    172               X_test_vec = self.vectorizer.transform(X_test)
    173
    174               predictions = []
    175
    176               # For each test document
    177               for i in range(len(X_test)):
    178                   test_vector = X_test_vec[i]
    179                   best_similarity = -float('inf')
    180                   best_class = None
    181
    182                   # Compare to each prototype
    183                   for class_label, prototype in self.prototypes.items():
    184                       if self.metric == 'cosine':
    185                           # Cosine similarity
    186                           similarity = np.dot(test_vector, prototype)
    187                           norm_test = np.linalg.norm(test_vector)
    188                           norm_proto = np.linalg.norm(prototype)
    189
    190                           if norm_test > 0 and norm_proto > 0:
    191                               similarity = similarity / (norm_test * norm_proto)
    192                           else:
    193                               similarity = 0
    194                       else:
    195                           # Euclidean distance (convert to similarity)
    196                           distance = np.linalg.norm(test_vector - prototype)
    197                           similarity = -distance
    198
    199                       # Keep the best match
    200                       if similarity > best_similarity:
    201                           best_similarity = similarity

    5                                                                                                                         Altair SLC

    202                           best_class = class_label
    203
    204                   predictions.append(best_class)
    205
    206               return predictions
    207
    208           def predict_proba(self, X_test):
    209               """Return probability-like scores for each class."""
    210               X_test_vec = self.vectorizer.transform(X_test)
    211               n_classes = len(self.classes)
    212               n_samples = len(X_test)
    213
    214               scores = np.zeros((n_samples, n_classes))
    215
    216               for i in range(n_samples):
    217                   test_vector = X_test_vec[i]
    218
    219                   for j, class_label in enumerate(self.classes):
    220                       prototype = self.prototypes[class_label]
    221
    222                       if self.metric == 'cosine':
    223                           similarity = np.dot(test_vector, prototype)
    224                           norm_test = np.linalg.norm(test_vector)
    225                           norm_proto = np.linalg.norm(prototype)
    226
    227                           if norm_test > 0 and norm_proto > 0:
    228                               similarity = similarity / (norm_test * norm_proto)
    229                           else:
    230                               similarity = 0
    231                       else:
    232                           distance = np.linalg.norm(test_vector - prototype)
    233                           similarity = np.exp(-distance)  # Convert to similarity
    234
    235                       scores[i, j] = similarity
    236
    237               # Normalize scores to sum to 1 (softmax-like)
    238               # Add small epsilon to avoid division by zero
    239               scores_exp = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    240               proba = scores_exp / (np.sum(scores_exp, axis=1, keepdims=True) + 1e-10)
    241
    242               return proba
    243
    244       # Example usage - no external dependencies needed
    245       print("=" * 60)
    246       print("Rocchio Classifier Demo")
    247       print("=" * 60)
    248
    249
    250       # Sample training data
    251       documents = train_df['doc'].tolist()
    252       labels = train_df['label'].tolist()
    253
    254       # Sample test data
    255       test_docs = test_df['doc'].tolist()
    256
    257
    258       print("\nTraining documents:")
    259       for doc, label in zip(documents, labels):
    260           print(f"  '{doc}' -> {label}")
    261
    262       print("\nTest documents:")
    263       for doc in test_docs:
    264           print(f"  '{doc}'")

    6                                                                                                                         Altair SLC

    265
    266       # Create and train the classifier
    267       print("\nTraining Rocchio classifier...")
    268       classifier = RocchioClassifier(metric='cosine')
    269       classifier.fit(documents, labels)
    270
    271       print(f"Vocabulary size: {len(classifier.vectorizer.vocabulary_)}")
    272       print(f"Classes: {list(classifier.classes)}")
    273
    274       # Make predictions
    275       print("\nMaking predictions...")
    276       predictions = classifier.predict(test_docs)
    277
    278       print("\nPredictions:")
    279       for doc, pred in zip(test_docs, predictions):
    280           print(f"  '{doc}' -> {pred}")
    281
    282       # Get probability scores
    283       print("\nProbability scores:")
    284       proba = classifier.predict_proba(test_docs)
    285       for i, doc in enumerate(test_docs):
    286           print(f"\n  Document: '{doc}'")
    287           for j, class_label in enumerate(classifier.classes):
    288               print(f"    P({class_label}) = {proba[i, j]:.4f}")
    289
    290       print("\n" + "=" * 60)
    291
    292       predictions = classifier.predict(test_docs)
    293       proba = classifier.predict_proba(test_docs)
    294
    295       # Create DataFrame
    296       final_results = pd.DataFrame({
    297           'document': test_docs,
    298           'prob_ai': proba[:, 0],
    299           'prob_programming': proba[:, 1],
    300           'prediction': predictions
    301       })
    302
    303       print("\nFinal Results DataFrame:")
    304       print(final_results.head())
    305
    306       final_results.to_parquet('d:/wpswrkx/final_results.parquet', engine='pyarrow')
    307       print("Demo complete!")
    308       print("=" * 60)
    309       endsubmit;

    NOTE: Submitting statements to Python:


    310       run;
    NOTE: Procedure python step took :
          real time : 0.987
          cpu time  : 0.031


    311
    312       options set=PYTHONHOME "D:\py310";
    313       proc python;
    314       submit;
    315       import pyarrow
    316       import pandas as pd
    317       final_results = pd.read_parquet('d:/wpswrkx/final_results.parquet', engine='pyarrow');
    318       endsubmit;

    7                                                                                                                         Altair SLC


    NOTE: Submitting statements to Python:


    319       import python=final_results data=workx.final_results;
    NOTE: Creating data set 'WORKX.final_results' from Python data frame 'final_results'
    NOTE: Data set "WORKX.final_results" has 4 observation(s) and 4 variable(s)

    320       run;
    NOTE: Procedure python step took :
          real time : 0.972
          cpu time  : 0.000


    321
    322       proc print data=workx.final_results;
    323       run;
    NOTE: 4 observations were read from "WORKX.final_results"
    NOTE: Procedure print step took :
          real time : 0.009
          cpu time  : 0.000


    324
    325
    ERROR: Error printed on page 1

    NOTE: Submitted statements took :
          real time : 2.124
          cpu time  : 0.156

    /*              _
      ___ _ __   __| |
     / _ \ `_ \ / _` |
    |  __/ | | | (_| |
     \___|_| |_|\__,_|

    */
