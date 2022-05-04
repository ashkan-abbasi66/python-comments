"""
An iterator is an object which implements __iter__() and __next__().


https://www.w3schools.com/python/python_iterators.asp
"""

# Tuple is an iterable object. So, you can get an iterator from it.
mytuple = ("apple", "banana", "cherry")
myit = iter(mytuple)

try:
    print(next(myit))
    print(next(myit))
    print(next(myit))
    # print(next(myit))
except:
    raise StopIteration('There are only %d elements.'%len(mytuple))
print()

for x in mytuple:
    print(x)
print()


"""
Creating an iterator class

Only "__iter__" and "__next__" are mandatory.
"""

class myCounter:
    def __iter__(self):
        self.counter = 1
        # must always return the iterator object itself.
        return self

    def __next__(self):
        # must return the next item in the sequence.
        out = self.counter
        self.counter += 1
        return out


counter_object = myCounter()
counter = iter(counter_object) # get an iterator

print(next(counter))
print(next(counter))
print(next(counter), '\n')


"""
Let's use __init__ to initialize the first value of the counter.
"""

class myCounter2:

    def __init__(self, initial_value):
        self.counter = initial_value

    def __iter__(self):
        # must always return the iterator object itself.
        return self

    def __next__(self):
        # must return the next item in the sequence.
        out = self.counter
        self.counter += 1
        return out


counter_object2 = myCounter2(10)
counter2 = iter(counter_object2) # get an iterator

print(next(counter2))
print(next(counter2))
print(next(counter2), '\n')