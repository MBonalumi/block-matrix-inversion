# Block Matrix Inversion

This function performs the inverted of a matrix X, defined by blocks as:

```math
X = 
\begin{bmatrix}
   A & B \\
   C & D
\end{bmatrix}
```

The algorithm performed is Block Matrix Inversion.
(refer to https://arxiv.org/pdf/2305.11103.pdf for more information)

This repository implements two functions:
- The first, block_matrix_inversion(), requires  $A$ and its schur complement  $S_A = D-CA^{-1}B$ to be invertible.  
  It takes in input $A^{-1},~B,~C,~D$
- The second, block_matrix_inversion_D(), requires  $D$ and its schur complement  $S_D = A-BD^{-1}C$ to be invertible, instead.  
It takes in input $A,~B,~C,~D^{-1}$

If $A$ and its Schur complement $S_A$ are invertible, then the inverse of $X$ is:

```math
 X^{-1} =
\begin{bmatrix}
  A^{-1} + A^{-1}B~S_A^{-1}C~A^{-1}   &   -A^{-1}B~S_A^{-1} \\
  -S_A^{-1}C~A^{-1}    &    S_A^{-1}
\end{bmatrix}
```

If, instead, you want to use $D$ and $S_D$, the inverse of $X$ will be:
```math
 X^{-1} =
\begin{bmatrix}
  S_D^{-1}   &   -S_D^{-1}B~D^{-1} \\
  -D^{-1}C~S_D^{-1}   &   D^{-1}+D^{-1}C~S_D^{-1}B~D^{-1}
\end{bmatrix}
```


The advantage is that only the inverse of $A$ and $S$ are required.
This is convenient when $A^{-1}$ is already known.

An example is when expanding the kernel matrix in Gaussian Processes techniques.
(see https://github.com/MBonalumi/Batch-Incremental-Gaussian-Process-Regression-BIGPR)
