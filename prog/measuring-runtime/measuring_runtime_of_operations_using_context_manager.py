"""
SAMPLE OUTPUT:
Code block took 0.016984 seconds
##########Hello!##########
Code block took 0.015957 seconds
"""

import time

class Timer:
  """Context manager to measure code execution time."""
  def __enter__(self):
    self.start_time = time.time()
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    if exc_type is not None:
      print("Error during timing:", exc_val)
    else:
      end_time = time.time()
      print(f"Code block took {end_time - self.start_time:.6f} seconds")

def my_method1(arg1, arg2):
  for i in range(1000000):
    pass
  return arg1 + arg2

def my_method2(arg1):
  for i in range(1000000):
    pass
  print("#"*10 + arg1 + "#"*10)

with Timer():
  my_method1(10, 20)

with Timer():
  my_method2("Hello!")