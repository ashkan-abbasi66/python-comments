"""
SAMPLE OUTPUT:
my_method1 took 0.016955 seconds
##########Hello!##########
my_method2 took 0.016954 seconds

"""

from functools import wraps
import time

# def timer(func):
#   """Decorator to measure the runtime of a function."""
#   @wraps(func)
#   def wrapper(*args, **kwargs):
#     start_time = time.time()
#     result = func(*args, **kwargs)
#     end_time = time.time()
#     print(f"{func.__name__} took {end_time - start_time:.6f} seconds")
#     return result
#   return wrapper

def timer(func):
  """Decorator to measure the runtime of a function."""
  def wrapper(*args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    print(f"{func.__name__} took {end_time - start_time:.6f} seconds")
    return result
  return wrapper

@timer
def my_method1(arg1, arg2):
  for i in range(1000000):
    pass
  return arg1 + arg2

@timer
def my_method2(arg1):
  for i in range(1000000):
    pass
  print("#"*10 + arg1 + "#"*10)

my_method1(10, 20)
my_method2("Hello!")