# sparse-constrained-lda
The detailed instructions will be updated soon.

The sparse constrained LDA code is designed and built upon python package lda: https://pypi.python.org/pypi/lda, as it utilizes Cython for fast arithmetic operations.

It currently supports four types of constraints:
* Document Seed
* Document Pairwise
* Word Seed
* Word Pairwise



Reference:

1. Limin Yao, David Mimno, and Andrew McCallum. Efficient methods for topic model inference on streaming document collections.
KDD, 2009

2. Yi Yang, Doug Downey, and Jordan Boyd-Graber. Efficient Methods for Incorporating Knowledge into Topic Models. EMNLP, 2015.
