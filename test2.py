import numpy as np

# Your initial 100 x 100 matrix
a = np.zeros((100, 100))


np.set_printoptions(edgeitems=30, linewidth=10000)
np.set_printoptions(edgeitems=100, linewidth=180)
np.set_printoptions(precision=100)


# Your initial 100 x 100 matrix
a = np.zeros((100, 100))


for i in range(10):
  # the 10 x 10 generated matrix with "random" number
  # I'm creating it with ones for checking if the code works
  b = np.ones((10, 10)) * (i + 1)
  print(b)
  # The random version would be:
  #  b = np.random.rand(10, 10)
  # Diagonal insertion
  a[i*10:(i+1)*10,i*10:(i+1)*10] = b

print(a)


class test:
  def __init__(self):
    self.k=9

  def fun(self):
    rezult = self.k+self.b
    print(rezult)
    return rezult

a=test()
a.b=4
a.fun()
