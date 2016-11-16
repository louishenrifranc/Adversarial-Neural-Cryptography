import theano.tensor as T
import lasagne
from lasagne.layers import DenseLayer, InputLayer, get_output, get_all_params
import theano
import numpy as np


def get_init(name):
    if name == "glorotU":
        return lasagne.init.GlorotUniform


def get_nonlinearity(name):
    if name == "relu":
        return lasagne.nonlinearities.rectify
    elif name == "tanh":
        return lasagne.nonlinearities.tanh


def gen_data(n, msg_len, key_len):
    return (np.random.randint(0, 2, size=(n, msg_len)) * 2 - 1). \
               astype(theano.config.floatX), \
           (np.random.randint(0, 2, size=(n, key_len)) * 2 - 1). \
               astype(theano.config.floatX)


class HiddenLayer(object):
    def __init__(self, inputs, size_batch, input_size, n_hidden, n_out, name, nonlinearity="relu", depth=1):
        self.depth = depth
        self.l_in = InputLayer((size_batch, input()))
        self.l_hid = DenseLayer(self.l_in,
                                num_units=n_hidden,
                                nonlinearity=get_nonlinearity(nonlinearity),
                                W=get_init("glorotU"))
        if depth == 3:
            self.l_hid = DenseLayer(self.l_hid,
                                    num_units=n_hidden,
                                    nonlinearity=get_nonlinearity(nonlinearity),
                                    W=get_init("glorotU"))
        self.l_hid = DenseLayer(self.l_hid,
                                num_units=n_out,
                                nonlinearity=get_nonlinearity("tanh"),
                                W=get_init("glorotU"))

    def get_output(self):
        return get_output(self.l_hid)

    def get_params(self):
        params = get_all_params(self.l_hid)


class AdversarialNeuralCryptoNet(object):
    def __init__(self):
        self.SIZE_BATCH = 512
        self.SIZE_KEY = 16
        self.SIZE_MESSAGE = 16
        self.N_HIDDEN = 16

        ### BUILD THE MODELS
        X_msg = T.matrix('msg')
        X_key = T.matrix('key')
        input = T.concatenate([X_msg, X_key], axis=1)

        # Create Alice neural network
        alice_MLP = HiddenLayer(input,
                                self.SIZE_BATCH,
                                self.SIZE_KEY + self.SIZE_MESSAGE,
                                self.N_HIDDEN,
                                self.SIZE_MESSAGE,
                                "alice")

        # Create Bob neural network
        bob_MLP = HiddenLayer(alice_MLP.get_output(),
                              self.SIZE_BATCH,
                              self.SIZE_KEY + self.SIZE_MESSAGE,
                              self.N_HIDDEN,
                              self.SIZE_MESSAGE,
                              "bob")

        # Create Eve neural network
        eve_MLP = HiddenLayer(input,
                              self.SIZE_BATCH,
                              self.SIZE_KEY + self.SIZE_MESSAGE,
                              self.N_HIDDEN,
                              self.SIZE_MESSAGE,
                              "eve", depth=3)

        # LOSS FUNCTIONS
        eve_loss = T.mean(T.abs_(X_msg - eve_MLP.get_output()))

        bob_loss = T.mean(T.abs_(X_msg - bob_MLP.get_output())) \
                   + (self.SIZE_MESSAGE / 2 - eve_err) ** 2 / (self.SIZE_MESSAGE / 2)
        loss = {'bob': lasagne.updates.adagrad(bob_err)}
        loss = {'eve': lasagne.updates.adagrad(eve_err)}

        # PARAMS
        # update bob-alice network
        params = {'bob': [bob_MLP.get_params(), alice_MLP.get_params()]}
        self.train_fn = {'bob': theano.function([X_msg, X_key], bob_loss, updates=loss['bob'])}
        self.test_fn = {'bob': theano.function([X_msg, X_key], bob_loss)}

        params = {'eve': [eve_MLP.get_params()]}
        self.train_fn = {'eve': theano.function([X_msg], eve_loss, updates=params['eve'])}
        self.testfn = {'eve': theano.function([X_key], eve_loss)}

    def train(self, bob_or_eve, results, max_iters, print_every, es, es_limit=100):
        count = 0
        for i in range(max_iters):
            msg_in_val, key_val = gen_data(self.SIZE_BATCH, self.SIZE_MESSAGE, self.SIZE_KEY)

            loss = self.train_fn[bob_or_eve](msg_in_val, key_val)
            if not i % print_every:
                print('Iter ', i, ', Loss:', loss)
                if es and loss < es:
                    count += 1
                    if es_limit < count:
                        break


if __name__ == '__main__':
    adversarial_iterations = 60
    adv = AdversarialNeuralCryptoNet()
    print('LA')
    for i in range(adversarial_iterations):
        n = 2000
        print_every = 100
        print('Training bob and alice')
        adv.train('bob', None, n, print_every, es=0.01)
        print('Training eve')
        adv.train('eve', None, n, print_every, es=0.01)
