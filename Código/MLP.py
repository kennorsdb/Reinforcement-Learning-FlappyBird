import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split

class MnistLoader:

    def __init__(self, folder):
        mnist = fetch_mldata('MNIST original', data_home=folder)

        self.train_img, self.test_img, self.train_lbl, self.test_lbl = train_test_split(mnist.data, mnist.target, test_size=1 / 7.0, random_state=0)

    def getTrain(self):
        return self.train_img, self.train_lbl

    def getTest(self):
        return self.test_img, self.test_lbl


class NeuralNet:

    def __init__(self, X, Y, inputSize, h1size, h2size, outSize):
        self.input = inputSize
        self.hl1 = h1size
        self.hl2 = h2size
        self.outSize = outSize

        self.__XaviersInit__()

    def __XaviersInit__(self):
        self.w1 = np.random.randn(self.input, self.hl1)/np.sqrt(self.hl1)
        self.w2 = np.random.randn(self.hl1, self.hl2)/np.sqrt(self.hl2)
        self.w3 = np.random.randn(self.hl2, self.outSize)/np.sqrt(self.outSize)

    def forwardPass(self, input, y):
        z1 = np.dot(input, self.w1)
        act1 = self.reLu(z1)
        z2 = np.dot(act1, self.w2)
        act2 = self.reLu(z2)
        z3 = np.dot(act2, self.w3)
        output = self.softMax(z3)

    def reLu(self, n):
        zeros = np.zeros(n.shape)
        return np.max([zeros,n], axis=0)

    def reLuPrime(self, n):
        pass

    def softMax(self, n):
        exp = np.exp(n)
        return exp/np.sum(exp)

    def crossEntropy(self, s, y):
        yi = y.argmax()
        x = s[yi]
        loss = - np.log(x)
        return loss

    def crossPrime(self, pi):
        pi - 1

    def backward(self):
        pass



def main():
    dataset = MnistLoader("./NMIST")
    train, rl = dataset.getTrain()
    print train.shape
    print rl.shape

    rl = rl.astype(int)
    tm = np.zeros([rl.shape[0], 10], dtype=int)
    tm[np.arange(rl.shape[0]), rl] = 1 #One-hot vector labels
    print tm.shape
    print rl[2]
    print tm[2]


if __name__ == '__main__':
    main()