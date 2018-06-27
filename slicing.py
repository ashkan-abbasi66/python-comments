a=np.array([[1,2,3],[4,5,6]])
b=a[np.c_[0:2],np.r_[0:2]]

# stack/concatenate along a new dimension.
a=np.arange(0,6).reshape((2,3))
b=np.arange(0,6).reshape((2,3))+10
d=np.stack((a,b),axis=2)
print(d[:,:,0])

# concatenate along existing dimensions
c=np.concatenate((a,b), axis=1) # along columns
