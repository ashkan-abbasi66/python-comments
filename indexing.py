a=np.array([[1,2,3],[4,5,6]])
b=a[np.c_[0:2],np.r_[0:2]]

# stack/concatenate along a new dimension.
a=np.arange(0,6).reshape((2,3))
b=np.arange(0,6).reshape((2,3))+10
d=np.stack((a,b),axis=2)
print(d[:,:,0])

# concatenate along existing dimensions
c=np.concatenate((a,b), axis=1) # along columns

# access a tuple inside a list
# example 1
L = ['a', [(1, 2), ([3], 4)], 5]

print(L[0])
print(L[1]) # access the nested list
print(L[1][0]) # the first element of the nested list which is a tuple
print(L[1][0][0])
print(L[1][0][1])

print(L[1][1])# the second element of the nested list which is a tuple
print(L[1][1][0])
print(L[1][1][0][0])
print(L[1][1][1])

# example 2
L=[(1,2,3),(4,(5,6))]
print(L[0]) # a tuple
t=list(L[0]) # convert the tuple to list
print(t[2])

# example 3
# How do I return the 2nd value from each tuple inside this list?
L=[(1,2),(2,3),(4,5),(3,4),(6,7),(6,7),(3,8)]
t=[]
for i in range(0,len(L)):
    t.append(L[i][1])
print(t)
# list comprehension
t=[x[1] for x in L]
print(t)

# Example: np.partition and np.argpartition 
# Rearrange the array in such a way that the value in the k-th position
# matches with a sorted version of the input array. The order of other elements
# are not important.
a=np.array([8,7,1,6,9,4])
print("a = ",a)
print(np.partition(a,3))
print(np.partition(a,4))

# similarly, we can get the array of indices for this task:
indices=np.argpartition(a,4)
print(a[indices])

# we can use negative indices. let's get the last three elements:
print("a[-3:] = ",a[-3:])
# the following will put -3-th element in its correct position: 
#   (Note the sorted array is [1,4,6,7,8,9]) 
print(np.partition(a,-3))
