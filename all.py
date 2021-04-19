import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston


class Layers:

    def __init__(self, nodes_num=0, name=None, is_trainable=False):
        self.nodes_num = nodes_num
        self.name = name
        self.is_trainable = is_trainable
        self.value = None
        self.gradients = {}

    def __repr__(self):
        return '{}'.format(self.name)


class Placeholder(Layers):

    def __init__(self, nodes_num=0, inputs=None, name=None, is_trainable=False):
        Layers.__init__(self, nodes_num=nodes_num, name=name, is_trainable=is_trainable)
        self.x = inputs
        self.outputs = []

    def forward(self):
        self.value = self.x

    def backward(self):
        for n in self.outputs:
            self.gradients[self] = n.gradients[self] * 1


class Sigmoid(Layers):

    def __init__(self, nodes_num=0, inputs=None, name=None, is_trainable=False):
        Layers.__init__(self, nodes_num=nodes_num, name=name, is_trainable=is_trainable)
        self.x = inputs
        self.w_matrix = np.random.normal(size=[self.nodes_num, self.x.nodes_num])
        self.b = np.random.randint(0, 9)
        self.outputs = []
        self.x.outputs.append(self)

    def x_value_before_activate(self):
        return np.dot(self.w_matrix, self.x.value) + self.b

    def _sigmoid(self, x):
        return 1. / (1 + np.exp(-1 * x))

    def partial(self):
        return self._sigmoid(self.x_value_before_activate()) * (1 - self._sigmoid(self.x_value_before_activate()))

    def forward(self):
        self.value = self._sigmoid(self.x_value_before_activate())

    def backward(self):
        for n in self.outputs:
            x = np.array([self.x.value])
            before_activate = n.gradients[self] * self.partial()
            before_activate_m = np.transpose(np.array([before_activate]))
            self.gradients[self.x] = np.dot(np.transpose(self.w_matrix), before_activate)
            self.gradients['w_matrix'] = np.matmul(before_activate_m, x)
            self.gradients['b'] = np.sum(before_activate)


class ReLU(Layers):

    def __init__(self, nodes_num=0, inputs=None, name=None, is_trainable=False):
        Layers.__init__(self, nodes_num=nodes_num, name=name, is_trainable=is_trainable)
        self.x = inputs
        self.w_matrix = np.random.normal(size=[self.nodes_num, self.x.nodes_num])
        self.b = np.random.randint(0, 9)
        self.outputs = []
        self.x.outputs.append(self)

    def x_value_before_activate(self):
        return np.dot(self.w_matrix, self.x.value) + self.b

    def partial(self):
        p_vector = self.x_value_before_activate()
        p_vector[p_vector <= 0] = 0
        p_vector[p_vector > 0] = 1
        return p_vector

    def forward(self):
        self.value = self.x_value_before_activate()
        self.value[self.value <= 0] = 0

    def backward(self):
        for n in self.outputs:
            before_activate = n.gradients[self] * self.partial()
            x = np.array([self.x.value])
            before_activate_m = np.transpose(np.array([before_activate]))
            self.gradients[self.x] = np.dot(np.transpose(self.w_matrix), before_activate)
            self.gradients['w_matrix'] = np.matmul(before_activate_m, x)
            self.gradients['b'] = np.sum(before_activate)


class Mean(Layers):

    def __init__(self, nodes_num=0, y=None, x=None, name=None, is_trainable=False):
        Layers.__init__(self, nodes_num=nodes_num, name=name, is_trainable=is_trainable)
        self.x = x
        self.y = y
        self.w_matrix = np.random.normal(size=[self.nodes_num, self.x.nodes_num])
        self.b = np.random.randint(0, 9)
        self.x.outputs.append(self)

    def y_hat_value(self):
        return np.dot(self.w_matrix, self.x.value) + self.b

    def forward(self):
        self.value = np.mean((self.y.value - self.y_hat_value()) ** 2)

    def backward(self):
        x = np.array([self.x.value])
        before_activate = -2 * (self.y.value - self.y_hat_value())
        before_activate_m = np.transpose(np.array([before_activate]))
        self.gradients[self.y] = 2 * (self.y.value - self.y_hat_value())
        self.gradients[self.x] = np.dot(np.transpose(self.w_matrix), before_activate)
        self.gradients['w_matrix'] = np.matmul(before_activate_m, x)
        self.gradients['b'] = np.sum(before_activate)


class SoftMax(Layers):

    def __init__(self, nodes_num=0, y=None, x=None, name=None, is_trainable=False):
        Layers.__init__(self, nodes_num=nodes_num, name=name, is_trainable=is_trainable)
        self.x = x
        self.y = y
        self.w_matrix = np.random.normal(size=[self.nodes_num, self.x.nodes_num])
        self.b = np.random.randint(0, 9)
        self.x.outputs.append(self)

    def y_hat_value(self):
        x_value_before_activate = np.exp(np.dot(self.w_matrix, self.x.value) + self.b)
        total = np.sum(x_value_before_activate)
        return x_value_before_activate / total

    def forward(self):
        self.value = - np.dot(self.y.value, np.log(self.y_hat_value()))

    def backward(self):
        x = np.array([self.x.value])
        before_activate = self.y_hat_value() * np.sum(self.y.value) - self.y.value
        before_activate_m = np.transpose(np.array([before_activate]))
        self.gradients[self.x] = np.dot(np.transpose(self.w_matrix), before_activate)
        self.gradients['w_matrix'] = np.matmul(before_activate_m, x)
        self.gradients['b'] = np.sum(before_activate)


def sgd(layers, learning_rate=1e-2):
    for l in layers:
        if l.is_trainable:
            w_matrix = np.transpose(l.w_matrix)
            w_gradients = np.transpose(l.gradients['w_matrix'])
            l.w_matrix = np.transpose(w_matrix - 1 * w_gradients * learning_rate)
            l.b += -1 * l.gradients['b'] * learning_rate


def forward_and_backward(order, monitor=False, predict_mode=False):
    if not predict_mode:
        # 整体的参数更新一次
        for layer in order:
            if monitor:
                print("前向计算Node：{}".format(layer))
            layer.forward()

        for layer in order[::-1]:
            if monitor:
                print("后向传播Node：{}".format(layer))
            layer.backward()
    else:
        for n in range(len(order) - 1):
            if monitor:
                print("前向计算Node：{}".format(order[n]))
            order[n].forward()


def predict(node, Loss, test, order, monitor=False):
    Loss.y.value = 0
    node.x = test
    forward_and_backward(order, monitor=monitor, predict_mode=True)
    return np.max(Loss.y_hat_value()), np.argmax(Loss.y_hat_value()), Loss.y_hat_value()


data = load_boston()
X_, y_ = data['data'], data['target']
X_rm = X_[:, 5]

# 网络框架搭建
x = Placeholder(nodes_num=5, inputs=None, name='x', is_trainable=False)
y = Placeholder(nodes_num=5, inputs=None, name='y', is_trainable=False)

Layer1 = Sigmoid(nodes_num=100, inputs=x, name='Layer1', is_trainable=True)
Layer2 = Sigmoid(nodes_num=100, inputs=Layer1, name='Layer2', is_trainable=True)
Loss = Mean(nodes_num=5, y=y, x=Layer2, name='Loss', is_trainable=True)

order = [x, y, Layer1, Layer2, Loss]

# 开始训练模型
losses = []
EPOCHS = 100

for e in range(EPOCHS):

    print('这是第{}轮'.format(e+1))

    batch_loss = 0

    batch_size = 100

    for b in range(batch_size):
        LOSS = 0
        index = np.random.choice(range(len(X_rm) - 5))
        x.x = X_rm[index:index + 5]
        y.x = y_[index:index + 5]

        forward_and_backward(order, monitor=False)
        sgd(order, learning_rate=1e-3)

        batch_loss += Loss.value

    losses.append(batch_loss / batch_size)
    print('本轮Loss：{}'.format(batch_loss / batch_size))

# 利用模型进行预测
rand_num = np.random.random((5, 1))
_, _, pre_num = predict(x, Loss, rand_num, order, monitor=False)
print(pre_num)

