from PIL import Image
fname = 'data/bird.png'
I = Image.open(fname) #.convert('L') can be used to convert an rgb image to gray-scale

# I.show()

print('Minimum and Maximum intensity values for each channel:',I.getextrema())
t = I.getextrema() # tuple --- list ???
print(t[0])
