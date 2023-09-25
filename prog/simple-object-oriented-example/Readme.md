# Notes on object-oriented programming in Python

## class method

- Class method can be used to create different initializers 

  ```python
  import math
  
  
  class Circle:
      def __init__(self, radius):
          self.radius = radius
  
      @classmethod
      def from_diameter(cls, diameter):
          radius = diameter/2
          return cls(radius=radius)
  
      def area(self):
          return math.pi * self.radius ** 2

  if __name__ == "__main__":
    print(Circle(2).area())                   # 12.566370614359172
    print(Circle.from_diameter(4).area())     # 12.566370614359172
  ```
