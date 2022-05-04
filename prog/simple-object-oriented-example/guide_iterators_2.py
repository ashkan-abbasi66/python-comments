"""

1. Generator function contains one or more yield statements.
2. When called, it returns an object (iterator) but does not start execution immediately.
3. Methods like __iter__() and __next__() are implemented automatically.
    => you can iterate through `next()`
    => for loops can be used.
4. Once the function yields, the function is paused and the control is transferred to the caller.
    4.1. Local variables and their states are remembered between successive calls.
         (in normal functions all local variables are destroyed when the function returns)
    4.2. Finally, when the function terminates, StopIteration is raised automatically on further calls.
5. To restart the process we need to create another generator object.


"""


def my_gen():
    n = 1
    print('This is printed first')
    # Generator function contains yield statements
    yield n

    n += 1
    print('This is printed second')
    yield n

    n += 1
    print('This is printed at last')
    yield n

g = my_gen()

print(g, '\n')

next(g)
next(g)
next(g)
# next(g) # StopIteration
print()

g2 = my_gen()
next(g2)

"""
Define an iterator and its equivalent generator 
for Fibonachi series 
"""


class fibo:
    def __init__(self):
        self.a = 1
        self.b = 1
        self.counter = 1

    def __iter__(self):
        return self

    def __next__(self):
        if self.counter <= 2:
            out = self.a
        else:
            out = self.a + self.b
            self.a = self.b
            self.b = out
        self.counter += 1
        return out

fibo_obj = fibo()
fi = iter(fibo_obj)

for i in range(10):
    print(i + 1, next(fi))
print()
# If you want to restrict it, you need to modify the __next__


def fibo_gen(seri_length):
    a, b = 1, 1
    for _ in range(seri_length):
        yield a
        tmp = b
        b = a + b
        a = tmp


fibo_gen_obj = fibo_gen(10)
for element in fibo_gen_obj:
    print(element)

print("end", next(fibo_gen_obj))  # StopIteration

