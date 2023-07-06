import numpy as np

def matrix_block_inversion(Ainv,B,C,D):
    '''
    NB: the inverse of a matrix M is here indicated as M'

    we perform the inverse of a matrix
    
    X = [A B]
        [C D]
    
    by using Block Matrix Inversion.
    The algorithm requires A and its schur complement D-CA'B to be invertible.
    If not, one could use D and its schur complement A-BD'C instead, being sure those are invertible.

    If A and its Schur complement S are invertible, then the inverse of X is
    (see https://arxiv.org/pdf/2305.11103.pdf)

    X'= [  A' + A'BS'CA'   -A'BS'  ]
        [     -S'CA'         S'    ]

    where only the inverse of A and S are required.

    This come in handy since A' is already known in this application and S' in very much smaller than X'.
    '''

    assert Ainv.shape[0] == Ainv.shape[1]
    assert D.shape[0] == D.shape[1]
    assert B.shape[0] == Ainv.shape[0] and B.shape[1] == D.shape[0]
    assert C.shape[0] == D.shape[0] and C.shape[1] == Ainv.shape[0]


    S = D - C @ Ainv @ B
    Sinv = np.linalg.inv(S)

    Xinv00 = Ainv + Ainv @ B @ Sinv @ C @ Ainv
    Xinv01 = -Ainv @ B @ Sinv
    Xinv10 = -Sinv @ C @ Ainv
    Xinv11 = Sinv

    Xinv = np.block([   [Xinv00, Xinv01],
                        [Xinv10, Xinv11]   ])

    return Xinv