import numpy as np

def init(K, D, W):
    np.random.seed(0)

    sd = np.floor(K*np.random.random((D,1))).astype(int)
    swk = np.zeros((W,K))
    sk_docs = np.zeros((K,1))

    for d in np.arange(D):
        w = A[A[:,0]==d,1]      # unique words in doc d
        c = A[A[:,0]==d,2]      # counts
        k = sd[d]               # doc d is in mixture k
        swk[w,k] = swk[w,k] + c # num times word w is assigned to mixture component k
        sk_docs[k] = sk_docs[k] + 1

    sk_words = np.sum(swk,axis=0).T
    return sd, swk, sk_docs, sk_words

def sample_discrete(b):
    r = np.sum(b)*np.random.random()
    a = b[0].copy()
    i = 0
    while a < r:
        i += 1
        a += b[i]
    return i

from scipy.special import gammaln
def log_ll(B, d, swk, sk_docs, sk_words, alpha, gamma, W, K):
    w = B[B[:, 0] == d + 2000, 1]
    c = B[B[:, 0] == d + 2000, 2]

    t = np.zeros([K, W])
    for k in np.arange(K):
        th = (sk_docs[k] + alpha) / np.sum(sk_docs + alpha)
        ph = (swk[:, k] + gamma) / (sk_words[k] + gamma * W)
        t[k] = th * ph
    assert c.shape[0] == t[:, w].sum(0).shape[0]
    p = c * np.log(t[:, w].sum(0))
    return p.sum(0)

    # l = 0
    # l += np.sum(gammaln(sk_docs + alpha))
    # l -= gammaln(np.sum(sk_docs + alpha))
    # l += gammaln(K * alpha)
    # l -= K * gammaln(alpha)
    #
    # for k in np.arange(K):
    #     l += gammaln(gamma * W)
    #     l -= W * gammaln(gamma)
    #     l += np.sum(gammaln(swk[w, k] + gamma))
    #     l -= gammaln(np.sum(sk_words[k] + (W * gamma)))
    # return l




def perplexity(B, swk, sk_docs, sk_words, alpha, gamma, W, K):
    t = np.zeros([K, W])
    for k in np.arange(K):
        th = (sk_docs[k] + alpha) / np.sum(sk_docs + alpha)
        ph = (swk[:, k] + gamma) / (sk_words[k] + gamma * W)
        t[k] = th * ph


    p = 0
    for w in np.unique(B[:, 1]):
        c = B[B[:, 1] == w, 2].sum()

        p += c * np.log(t[:, w].sum(0))


    p = p / B[:, 2].sum(0)

    return np.exp(-p)

def run(A, K, D, W, alpha, gamma):
    sd, swk, sk_docs, sk_words = init(K, D, W)
    # This makes a number of Gibbs sampling sweeps through all docs and words
    num_sweeps = 30
    for i_sweep in np.arange(num_sweeps):
        print("gibbs sweep : {0}".format(i_sweep))
        for d in np.arange(D):
            w = A[A[:, 0] == d, 1]  # unique words in doc d
            c = A[A[:, 0] == d, 2]  # counts

            # remove doc d's contributions from count tables
            swk[w, sd[d]] = swk[w, sd[d]] - c
            sk_docs[sd[d]] = sk_docs[sd[d]] - 1
            sk_words[sd[d]] = sk_words[sd[d]] - np.sum(c)

            # log probability of doc d under each mixture component
            lb = np.zeros(K)
            for k in np.arange(K):
                ll = np.dot(c, (np.log(swk[w, k] + gamma) - np.log(sk_words[k] + gamma * W)))
                lb[k] = np.log(sk_docs[k] + alpha) + ll

            # assign doc d to a new component
            b = np.exp(lb - np.max(lb))
            kk = sample_discrete(b)

            # add back doc d's contributions from count tables
            swk[w, kk] = swk[w, kk] + c
            sk_docs[kk] = sk_docs[kk] + 1
            sk_words[kk] = sk_words[kk] + np.sum(c)
            sd[d] = kk

    return sd, swk, sk_docs, sk_words
if __name__ == '__main__':
    A = np.load('data/mat_A.npy')
    B = np.load('data/mat_B.npy')
    words = np.load('data/words.npy')


    W = np.max(np.hstack((A[:, 1], B[:, 1]))) + 1  # number of unique words
    D = np.max(A[:, 0]) + 1  # number of documents in A
    K = 20  # number of mixture components we will use

    alpha = 1.   # parameter of the Dirichlet over mixture components
    gamma = 0.1  # parameter of the Dirichlet over words

    sd, swk, sk_docs, sk_words = run(A, K, D, W, alpha, gamma)

    print(f"doc:{1} \t log_p:\t{log_ll(B, 1, swk, sk_docs, sk_words, alpha, gamma, W, K)}")
    print(f"doc:{0} \t log_p:\t{log_ll(B, 0, swk, sk_docs, sk_words, alpha, gamma, W, K)}")
    print(f"doc:{100} \t log_p:\t{log_ll(B, 100, swk, sk_docs, sk_words, alpha, gamma, W, K)}")

    print(f"perplexity: \t {perplexity(B, swk, sk_docs, sk_words, alpha, gamma, W, K)}")