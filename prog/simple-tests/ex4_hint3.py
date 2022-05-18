
import numpy as np

N = 12
M = 10
kn = np.zeros(shape = (M, N))
print(kn, '\n')

selected_rows = [1, 4, 7]        # spacing  = 3
selected_cols = [2, 4, 6, 8, 10] # even numbers



off1 = 1  # on rows
off2 = 2
sp1 = 3   # on rows
sp2 = 2

for i in range(M):
    if i>=off1:
        if (i - off1) % sp1 == 0:
            print(i)


kn2 = kn.copy()
for i in range(M):
    for j in range(N):
        if i>=off1 and j>=off2:
            if (i - off1) % sp1 == 0 and (j - off2) % sp2 == 0:
                kn2[i,j] = 1
print(kn2)

kn3 = kn.copy()
for i in range(off1, M, sp1):
    for j in range(off2, N, sp2):
        kn3[i,j] = 1
print(kn3)

print(np.allclose(kn2,kn3))