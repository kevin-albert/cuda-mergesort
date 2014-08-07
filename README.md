gpu-mergesort
=============

A parallel implementation of mergesort, using CUDA to perform the sorting on the GPU.

It reads in signed integers from stdin. There is no checking if the characters it reads are actually numerical, so if you give him
non-numeric characters, it will read them according to their character code and happily keep going :)

Example of how it sorts:
```
Input:          8 3 1 9 1 2 7 5 9 3 6 4 2 0 2 5
Threads: |    t1    |    t2    |    t3    |    t4    |
         | 8 3 1 9  | 1 2 7 5  | 9 3 6 4  | 2 0 2 5  |
         |  38 19   |  12 57   |  39 46   |  02 25   |
         |   1398   |   1257   |   3469   |   0225   |
         +----------+----------+----------+----------+
         |          t1         |          t2         |
         |       11235789      |       02234569      |
         +---------------------+---------------------+
         |                     t1                    |
         |      0 1 1 2 2 2 3 3 4 5 5 6 7 8 9 9      |

Output is the original list, sorted.
```
