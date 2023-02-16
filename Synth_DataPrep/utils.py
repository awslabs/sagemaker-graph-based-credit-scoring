import numpy as np
import psutil
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


def construct_network_data(dataset, text_column_name, embedding_size, window_size=5, min_count=1, cutoff=0.5):
    """
    This function takes in a dataset with a text column and generates a graph where the nodes are
    linked based on similarity between embeddings for each node. 
    
    The dataset with a text column is used to generate a embedding matrix, where graph nodes on the rows and embedding dimension on the columns.
    Example: a graph with N nodes and embedding dimension of 300 will have embedding_matrix be of shape (N,300)
    
    cutoff: the threshold of cosine similarity to add a link between two nodes.
    
    Returns a dict with source and destination nodes, length of number of links in graph
    """
    docs_words = [d.split() for d in dataset[text_column_name].values.tolist()]
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(docs_words)]
    
    # initialize the model
    model = Doc2Vec(documents, vector_size=embedding_size, window=window_size, min_count=min_count, workers=psutil.cpu_count()-1)
    
    doc_len = len(documents)
    embedding_matrix = np.zeros((doc_len, embedding_size))
    
    def infer(i):
        embedding_matrix[i] = model.infer_vector(docs_words[i])

    with ThreadPoolExecutor(max_workers=psutil.cpu_count()-1) as executor:
        for i in range(doc_len):
            executor.submit(infer, i)    
    
    print("Finish generating the embedding matrix.")
    
    similars = cosine_similarity(embedding_matrix)
    np.fill_diagonal(similars, 0.0)
    n_nodes = embedding_matrix.shape[0]
    x = similars.reshape(n_nodes*n_nodes,1)
    n_links = len(x[x>cutoff])/2

    # Examine the distribution of cosine similarities and other statistics of the graph
    plt.hist(similars.reshape(n_nodes*n_nodes,1),100); plt.grid()
    print("Number of nodes =", n_nodes)
    print("Mean of cosine similarities =", np.mean(similars))
    print("Median of cosine similarities =", np.median(similars))  
    print("Number of nodes =", n_nodes)
    print("Number of links (symmetric) =",n_links)
    print("Average degree = ", 2*n_links/n_nodes)
    
    # Construct the source and destination node lists
    # Note that this delivers an undirected, symmetric graph - modify for directed graph
    src = []
    dst = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if similars[i,j]>cutoff:
                src.append(i)
                dst.append(j)
    print("Check: number of links =", len(src))
        
    return {'src': src, 'dst': dst} # dict so that it can be pickled

    