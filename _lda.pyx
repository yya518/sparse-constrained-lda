#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

from cython.operator cimport preincrement as inc, predecrement as dec
from libc.stdlib cimport malloc, free, calloc

cdef extern from "gamma.h":
    cdef double lda_lgamma(double x) nogil


cdef double lgamma(double x) nogil:
    if x <= 0:
        with gil:
            raise ValueError("x must be strictly positive")
    return lda_lgamma(x)


cdef int searchsorted(double* arr, int length, double value) nogil:
    """Bisection search (c.f. numpy.searchsorted)

    Find the index into sorted array `arr` of length `length` such that, if
    `value` were inserted before the index, the order of `arr` would be
    preserved.
    """
    cdef int imin, imax, imid
    imin = 0
    imax = length
    while imin < imax:
        imid = imin + ((imax - imin) >> 2)
        if value > arr[imid]:
            imin = imid + 1
        else:
            imax = imid
    return imin


def _sample_topics_sparse(int[:] WS, int[:] DS, int[:] ZS, int[:, :] nzw, int[:, :] ndz, int[:] nz,
                   double alpha, double beta, double beta_sum, double[:] rands,
                   double[:, :, :] potential):
    """SparseLDATrainGibbsSampler implements the SparseLDA gibbs sampling
    training algorithm.  In gibbs sampling formula:
     (0) p(z|w) --> p(z|d) * p(w|z) * potential(z, w, d)
                --> (alpha(z) + N(z|d)) * p(w|z)
                 =  (alpha(z) + N(z|d)) * p(w|z) *
                    (beta + N(w|z)) / (beta * |V| + N(z))
                 =  alpha(z) * beta / (beta * |V| + N(z)) +
                    N(z|d) * beta / (beta * |V| + N(z)) +
                    (alpha(z) + N(z|d)) * N(w|z) / (beta * |V| + N(z))
     (1) s(z) = alpha(z) * beta / (beta * |V| + N(z)), it can be pre-computed. only needs to be updated once for each token sampling
     (2) r(z, d) = N(z|d) * beta / (beta * |V| + N(z)), it's nonzero only for n(d, z) that is nonezero
     (3) q(z, w, d) = N(w|z) * (alpha(z) + N(z|d)) / (beta * |V| + N(z)), it's nonzero only for n(z,w) is nonzero
     (4) q_coefficient(z, d) = (alpha(z) + N(z|d)) / (beta * |V| + N(z))
                             = alpha(z)/ (beta * |V| + N(z)) +  N(z|d)/(beta * |V| + N(z))
                             the first part is global value, and the second part is document-specific
    This process divides the full sampling mass into three buckets, where s(z)
    is a smoothing-only bucket, r(z, d) is a document-topic bucket, and
    q(z, w, d) is a topic-word bucket.
    """
    cdef int i, k, w, d, z, z_new
    cdef int d_idx = -1
    cdef double sample
    cdef int N = WS.shape[0]
    cdef int D = ndz.shape[0]
    cdef int n_rand = rands.shape[0]
    cdef int n_topics = nz.shape[0]
    cdef double smoothing_only_sum = 0
    cdef double doc_topic_sum = 0
    cdef double topic_word_sum = 0
    cdef double tmp_smoothing_only_sum, tmp_doc_topic_sum, tmp_topic_word_sum
    cdef double* tmp_smoothing_only_bucket = <double*> calloc(n_topics, sizeof(double))
    cdef double* tmp_topic_word_coef = <double*> calloc(n_topics, sizeof(double))
    cdef double* tmp_doc_topic_bucket = <double*> calloc(n_topics, sizeof(double))
    cdef double* tmp_topic_word_bucket = <double*> calloc(n_topics, sizeof(double))

    cdef double* smoothing_only_bucket = <double*> calloc(n_topics, sizeof(double))
    cdef double* topic_word_coef = <double*> calloc(n_topics, sizeof(double))
    cdef double* doc_topic_bucket = <double*> calloc(n_topics, sizeof(double))
    cdef double* topic_word_bucket = <double*> calloc(n_topics, sizeof(double))

    if smoothing_only_bucket is NULL:
        raise MemoryError("Could not allocate memory during sampling.")
    with nogil:
        """ compute smoothing-only bucket
            s(z) = alpha(z) * beta / (beta * |V| + N(z))
        """
        for k in range(n_topics):
            smoothing_only_bucket[k] = alpha * beta / ( beta_sum + nz[k] )
            smoothing_only_sum += smoothing_only_bucket[k]

        """ initial topic word coefficient
            q_coefficient(z) = alpha(z) / (beta * |V| + N(z))
        """
        for k in range(n_topics):
            topic_word_coef[k] = alpha / (beta_sum + nz[k])

        for i in range(N):
            d = DS[i]
            if d != d_idx: # a new document
                d_idx +=1
                doc_topic_sum = 0
                """ compute doc topic bucket & update topic word coefficient
                   r(z, d) = N(z|d) * beta / (beta * |V| + N(z))
                   q_coefficient(z, d) = (alpha(z) + N(z|d)) / (beta * |V| + N(z))
                """
                for k in range(n_topics):
                    if ndz[d,k] != 0:
                        doc_topic_bucket[k] = beta * ndz[d,k] / (beta_sum + nz[k])
                        doc_topic_sum += doc_topic_bucket[k]
                        topic_word_coef[k] = (alpha + ndz[d,k]) / (beta_sum + nz[k])

            w = WS[i]
            z = ZS[i]
            #remove word topic and update bucket values related to z
            dec(nzw[z, w])
            dec(ndz[d, z])
            dec(nz[z])
            smoothing_only_sum -= smoothing_only_bucket[z]
            smoothing_only_bucket[z] = alpha * beta / ( beta_sum + nz[z] )
            smoothing_only_sum += smoothing_only_bucket[z]
            doc_topic_sum -= doc_topic_bucket[z]
            doc_topic_bucket[z] = beta * ndz[d,z] / (beta_sum + nz[z])
            doc_topic_sum += doc_topic_bucket[z]
            topic_word_coef[z] = (alpha + ndz[d,z]) / (beta_sum + nz[z])

            """ compute topic word bucket
             q(z, w, d) = N(w|z) * (alpha(z) + N(z|d)) / (beta * |V| + N(z))
                      = N(w|z) * q_coefficient(z, d)
            """
            topic_word_sum = 0.0
            for k in range(n_topics):
                if nzw[k, w] != 0:
                    topic_word_bucket[k] = nzw[k, w] * topic_word_coef[k]
                    topic_word_sum += topic_word_bucket[k]

            # incorporate potential
            tmp_smoothing_only_bucket = smoothing_only_bucket
            tmp_smoothing_only_sum = smoothing_only_sum
            tmp_doc_topic_bucket = doc_topic_bucket
            tmp_doc_topic_sum = doc_topic_sum
            tmp_topic_word_bucket = topic_word_bucket
            tmp_topic_word_sum = topic_word_sum
            for k in range(n_topics):
                if potential[k, d, w] != 1:
                    tmp_smoothing_only_sum -= smoothing_only_bucket[k]
                    tmp_smoothing_only_bucket[k] = smoothing_only_bucket[k] * potential[k, d, w]
                    tmp_smoothing_only_sum += smoothing_only_bucket[k]

                    tmp_doc_topic_sum -= tmp_doc_topic_bucket[k]
                    tmp_doc_topic_bucket[k] = doc_topic_bucket[k] * potential[k,d,w]
                    tmp_doc_topic_sum += tmp_doc_topic_bucket[k]

                    tmp_topic_word_sum -= tmp_topic_word_bucket[k]
                    tmp_topic_word_bucket[k] = topic_word_bucket[k] * potential[k,d,w]
                    tmp_topic_word_sum += tmp_topic_word_bucket[k]

            # sample new topic
            total_mass = tmp_smoothing_only_sum + tmp_doc_topic_sum + tmp_topic_word_sum
            sample =  rands[i % n_rand] * total_mass
            if sample < tmp_topic_word_sum:
                for k in range(n_topics):
                    if nzw[k, w]!= 0:
                        sample -= tmp_topic_word_bucket[k]
                        if sample <= 0:
                            z_new = k
                            break
            else:
                sample -= tmp_topic_word_sum
                if sample < tmp_doc_topic_sum:
                    for k in range(n_topics):
                        if ndz[d, k] != 0:
                            sample -= tmp_doc_topic_bucket[k]
                            if sample <= 0:
                                z_new = k
                                break
                else:
                    sample -= tmp_doc_topic_sum
                    for k in range(n_topics):
                        sample -= tmp_smoothing_only_bucket[k]
                        if sample <=0:
                            z_new = k
                            break
            ZS[i] = z_new
            #add word topic
            inc(nzw[z_new, w])
            inc(ndz[d, z_new])
            inc(nz[z_new])
            smoothing_only_sum -= smoothing_only_bucket[z_new]
            smoothing_only_bucket[z_new] = alpha * beta / ( beta_sum + nz[z_new] )
            smoothing_only_sum += smoothing_only_bucket[z_new]
            doc_topic_sum -= doc_topic_bucket[z_new]
            doc_topic_bucket[z_new] = beta * ndz[d,z_new] / (beta_sum + nz[z_new])
            doc_topic_sum += doc_topic_bucket[z_new]
            topic_word_coef[z_new] = (alpha + ndz[d,z_new]) / (beta_sum + nz[z_new])

            """if next token is in a new document, we need to update topic_word coefficient
            q_coefficient(z) = alpha(z) / (beta * |V| + N(z))
            """
            if i != N-1 and DS[i+1] > d:
                for k in range(n_topics):
                    if ndz[d, k] != 0:
                        topic_word_coef[k] = alpha / (beta_sum + nz[k])
        free(doc_topic_bucket)
        free(topic_word_coef)
        free(topic_word_bucket)
        free(smoothing_only_bucket)


def _sample_topics(int[:] WS, int[:] DS, int[:] ZS, int[:, :] nzw, int[:, :] ndz, int[:] nz,
                   double alpha, double beta, double beta_sum, double[:] rands,
                   double[:, :, :] potential):
    cdef int i, k, w, d, z, z_new
    cdef double r, dist_cum
    cdef int N = WS.shape[0]
    cdef int n_rand = rands.shape[0]
    cdef int n_topics = nz.shape[0]
    #cdef double eta_sum = 0
    cdef double* dist_sum = <double*> malloc(n_topics * sizeof(double))
    if dist_sum is NULL:
        raise MemoryError("Could not allocate memory during sampling.")
    with nogil:
        #for i in range(eta.shape[0]):
        #    eta_sum += eta[i]

        for i in range(N):
            w = WS[i]
            d = DS[i]
            z = ZS[i]

            dec(nzw[z, w])
            dec(ndz[d, z])
            dec(nz[z])

            dist_cum = 0
            for k in range(n_topics):
                dist_cum += (nzw[k, w] + beta) / (nz[k] + beta_sum) * (ndz[d, k] + alpha) * potential[k,d,w]
                dist_sum[k] = dist_cum

            r = rands[i % n_rand] * dist_cum  # dist_cum == dist_sum[-1]
            z_new = searchsorted(dist_sum, n_topics, r)

            ZS[i] = z_new
            inc(nzw[z_new, w])
            inc(ndz[d, z_new])
            inc(nz[z_new])

        free(dist_sum)


cpdef double _loglikelihood(int[:, :] nzw, int[:, :] ndz, int[:] nz, int[:] nd, double alpha, double eta) nogil:
    cdef int k, d
    cdef int D = ndz.shape[0]
    cdef int n_topics = ndz.shape[1]
    cdef int vocab_size = nzw.shape[1]

    cdef double ll = 0

    # calculate log p(w|z)
    cdef double lgamma_eta, lgamma_alpha
    with nogil:
        lgamma_eta = lgamma(eta)
        lgamma_alpha = lgamma(alpha)

        ll += n_topics * lgamma(eta * vocab_size)
        for k in range(n_topics):
            ll -= lgamma(eta * vocab_size + nz[k])
            for w in range(vocab_size):
                # if nzw[k, w] == 0 addition and subtraction cancel out
                if nzw[k, w] > 0:
                    ll += lgamma(eta + nzw[k, w]) - lgamma_eta

        # calculate log p(z)
        for d in range(D):
            ll += (lgamma(alpha * n_topics) -
                    lgamma(alpha * n_topics + nd[d]))
            for k in range(n_topics):
                if ndz[d, k] > 0:
                    ll += lgamma(alpha + ndz[d, k]) - lgamma_alpha
        return ll