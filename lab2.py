import numpy as np
import time

# Vector creation

a = np.zeros(4);
print(f"1 np.zeros(4) : a = {a}, a.shape = {a.shape}, a.shape[0] = {a.shape[0]}, a.dtype = {a.dtype}")

a = np.zeros((4,))
print(f"2 np.zeros(4) : a = {a}, a.shape = {a.shape}, a.shape[0] = {a.shape[0]}, a.dtype = {a.dtype}")

a = np.random.random_sample((4,))
print(f"np.zeros(4) : a = {a}, a.shape = {a.shape}, a.shape[0] = {a.shape[0]}, a.dtype = {a.dtype}")

a = np.arange(4,)
print(f"np.zeros(4) : a = {a}, a.shape = {a.shape}, a.shape[0] = {a.shape[0]}, a.dtype = {a.dtype}")

a = np.random.rand(4)
print(f"np.zeros(4) : a = {a}, a.shape = {a.shape}, a.shape[0] = {a.shape[0]}, a.dtype = {a.dtype}")

# Operations on Vector
# Idexing

a = np.arange(10)
print(f"np.zeros(4) : a = {a}, a.shape = {a.shape}, a.shape[0] = {a.shape[0]}, a.dtype = {a.dtype}")

print(f"a[2] = {a[2]}")
print(f'a[-1] = {a[-1]}')

try:
    print(a[10])
except Exception as e:
    print('The error message you will see is :')
    print(e)

# Slicing

a = np.arange(32)
print(f'a = {a}')

print(f'a[5:7:1] = {a[5:7:1]}')

print(f'a[3:] = {a[3:]}')

print(f'a[:10] = {a[:10]}');

print(f'a[:] = {a[:]}');

# Single vector operations

a = np.array([1,2,3,4])
print(a);

b = -a
print(b)

a = np.array([1,2,3,4])
b = np.sum(a)
print(b)

b = np.mean(a)
print(b)

b = a**2
print(b)

a = np.array([1,2,3,4])
b = np.array([-1,-2,3,4])
print(f'Binary operator work element wise : {a + b}')

c = np.array([1,2])

try:
    d = a + c
except Exception as e:
    print('The error message you will see is:')
    print(e)

b = 5 * a
print(f'b = 5 * a = {b}')

def my_dot(a, b):
    x = 0
    for i in range(a.shape[0]):
        x = x + a[i] * b[i]
    return x

a = np.array([1,2,3,4])
b = np.array([-1, 4, 3,2])
print(f'my_dot(a,b) = {my_dot(a,b)}')

c = np.dot(a,b)
print(f'Numpy 1-D np.dot(a,b) = {c} np.dot(a,b).shape = {c.shape}')

c = np.dot(b,a)
print(c)

np.random.seed(1)
a = np.random.rand(10000000)
b = np.random.rand(10000000)

tic = time.time()
c = np.dot(a,b)
toc = time.time()

print(c)
print(f'{c:.3f}')
print(toc-tic)
print(f'Vectorised version duration : {1000*(toc-tic):.4f} ms')

tic = time.time()
c = my_dot(a,b)
toc = time.time()

print(c)
print(f'{c:.3f}')
print(toc-tic)
print(f'Vectorised version duration : {1000*(toc-tic):.4f} ms')

del(a); del(b);

a = np.random.rand(3,5)
print(f'np.random.rand(3,5) :  {a}')

a = np.random.random_sample((3,5))
print(f'np.random.random_sample((3,5)) : {a}')

rng = np.random.default_rng(42)

x = rng.random()
print(x)

x = rng.random(5)
print(x)

x = rng.random((3,5))
print(x)

a = np.zeros((1,5))
print(f'a shape = {a.shape} a = {a} a.dtype = {a.dtype}')

a = np.zeros((2,5))
print(f'a shape = {a.shape} a = {a} a.dtype = {a.dtype}')

a = np.array([[5,4], [4,3], [3,1]]);
print(f'a shape = {a.shape} a = {a} a.dtype = {a.dtype}')

a = np.arange(6).reshape(-1, 2)
print(f'a shape = {a.shape} a = {a} a.dtype = {a.dtype}')

print(f'\na[2,0].shape = {a[2,0].shape}, a[2,0] = {a[2,0]} type(a[2,0]) = {type(a[2,0])}')

# Access row
print(f'\na[2].shape = {a[2].shape}, a[2] = {a[2]} type(a[2]) = {type(a[2])}')

a = np.arange(20).reshape(-1, 10)
print(a)

# access 5 cosecutive elements (start:stop:step)
print(f'a[0, 2:7:1] = {a[0, 2:7:1]} a[0, 2:7:1].shape = {a[0, 2:7:1].shape} a 1-D array')

# access 5 consecutive elements from two rows (start:stop:step)
print(f'a[:, 2:7:1] = {a[:, 2:7:1]} a[0, 2:7:1].shape = {a[:, 2:7:1].shape} a 2-D array')

# access all elements
print(f'a[:,:] = {a[:,:]} a[:,:].shape = {a[:,:].shape}')

# access all elements in one row (very common usage)
print(f'a[1,:] = {a[1,:]} a[1,:].shape = {a[1,:].shape} a 1-D array')

# same as
print(f'a[1] = {a[1]} a[1].shape = {a[1].shape} a 1-D array')







      






