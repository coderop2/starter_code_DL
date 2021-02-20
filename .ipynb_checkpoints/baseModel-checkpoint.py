class Loss:
    def __call__(self, y_true, y_pred):
        self.y_pred = y_pred
        self.y_true = y_true
        return ((y_true - y_pred)**2).mean()
    
    def gradient(self):
        return 2 * (self.y_pred - self.y_true) / self.y_true.shape[0]
    
class Layer:
    def __init__(self, input_dim, num_hidden = 1):
        self.weights = np.random.randn(input_dim, num_hidden) * np.sqrt(2. / input_dim)
        self.bias = np.zeros(num_hidden)
        
    def __call__(self, x):
        self.x = x
        z = x @ self.weights + self.bias
        return z
    
    def backward(self, gradient):
        self.weights_grad = self.x.T @ gradient
        self.bias_grad = gradient.sum(axis = 0)
        self.x_grad = gradient @ self.weights.T
        return self.x_grad
    
    def update(self, lr):
        self.weights = self.weights - lr * self.weights_grad
        self.bias = self.bias - lr * self.bias_grad
        
x = np.random.uniform(-1, 1, (100, 2))
weights_true = np.array([[5,10],]).T
bias_true = np.array([10])

y_true = x @ weights_true + bias_true
loss = Loss()
linear = Layer(2)
for i in range(1):
    pred = linear(x)
    loss_out = loss(y_true, pred)
    if i % 5 == 0:
        print(loss_out)
    grad = loss.gradient()
    print(grad.shape)
    print(grad.sum(axis=0))
    a = linear.backward(grad)
    linear.update(0.1)
    
    
##### Multilayer Base


class model:
    def __init__(self):
        self.l1 = Layer(2,100)
        self.l2 = Layer(100)
        
    def __call__(self, x):
        self.x = x
        self.z1 = self.l1(x)
        self.z2 = self.l2(self.z1)
        return self.z2
    
    def backward(self, grad):
        self.l2_x_grad = self.l2.backward(grad)
        self.l1_x_grad = self.l1.backward(self.l2_x_grad)
        return self.l1_x_grad
    
    def update(self, lr):
        self.l1.update(lr)
        self.l2.update(lr)
        
x = np.random.uniform(-1, 1, (100, 2))
weights_true = np.array([[5,10],]).T
bias_true = np.array([10])

y_true = (x**2) @ weights_true + x @ weights_true + bias_true
loss = Loss()
linear = model()
for i in range(50):
    pred = linear(x)
    loss_out = loss(y_true, pred)
    if i % 5 == 0:
        print(loss_out)
    grad = loss.gradient()
    a = linear.backward(grad)
    linear.update(0.03)