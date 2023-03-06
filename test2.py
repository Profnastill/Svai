import numpy as np

# Your initial 100 x 100 matrix
a = np.zeros((26, 26))
print(a)

np.set_printoptions(edgeitems=30, linewidth=10000)
np.set_printoptions(edgeitems=100, linewidth=180)
np.set_printoptions(precision=100)


for i in range(4):
  # the 10 x 10 generated matrix with "random" number
  # I'm creating it with ones for checking if the code works
  b = np.ones((5, 5)) * (i + 1)
  n = b.shape[0]
  print(b)
  print(b.shape)
  # The random version would be:
  #  b = np.random.rand(10, 10)
  # Diagonal insertion
  test=a[i*n:(i+1)*n,i*n:(i+1)*n]
  print(f"test {i},\n {test}")

print(a)