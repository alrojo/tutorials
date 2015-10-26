from __future__ import print_function


import numpy as np
import theano
import theano.tensor as T
import lasagne


# Number of units in the hidden (recurrent) layer
N_HIDDEN = 20
# Number of training sequences in each batch
# Optimization learning rate
LEARNING_RATE = .001
# All gradients above this will be clipped
GRAD_CLIP = 100





def main():
    sym_y = T.imatrix('target_output')
    sym_mask = T.matrix('mask')
    sym_x = T.tensor3()

    num_epochs = 20
    batch_size = 10
    num_classes = 3

    n_samples = 1000
    n_inputs = 10
    seq_len = 50

    print("Building network ...")
    l_in = lasagne.layers.InputLayer(shape=(batch_size, seq_len, n_inputs))
    l_forward = lasagne.layers.LSTMLayer(l_in, num_units=N_HIDDEN)
    l_reshape = lasagne.layers.ReshapeLayer(
        l_forward, (batch_size*seq_len, N_HIDDEN))
    l_recurrent_out = lasagne.layers.DenseLayer(
        l_reshape, num_units=num_classes, nonlinearity=lasagne.nonlinearities.softmax)
    l_out = lasagne.layers.ReshapeLayer(
        l_recurrent_out, (batch_size, seq_len, num_classes))

    print("Building cost function ...")    
    out_train = lasagne.layers.get_output(
        l_out, sym_x, mask=sym_mask, deterministic=False)
    out_eval = lasagne.layers.get_output(
        l_out, sym_x, mask=sym_mask, deterministic=True)
    probs_flat = out_train.reshape((-1, num_classes))
    cost = T.nnet.categorical_crossentropy(probs_flat, sym_y.flatten())
    cost = T.sum(cost*sym_mask.flatten()) / T.sum(sym_mask)

    print("Computing updates ...")
    all_params = lasagne.layers.get_all_params(l_out)
    all_grads = T.grad(cost, all_params)
    updates, norm_calc = lasagne.updates.total_norm_constraint(all_grads, max_norm=GRAD_CLIP, return_norm=True)
    updates = lasagne.updates.rmsprop(updates, all_params, LEARNING_RATE)
    print("Compiling functions ...")
    train = theano.function(
        [sym_x, sym_y, sym_mask], [cost, out_train, norm_calc], updates=updates)
        
    eval = theano.function([sym_x, sym_mask], out_eval)



    num_batches = n_samples // batch_size

    for epoch in range(num_epochs):
        curcost = 0
	for i in range(num_batches):
            idx = range(i*batch_size, (i+1)*batch_size)
            x_batch = X[idx]
            y_batch = y[idx]
            mask_batch = np.ones_like(y_batch).astype('float32')  # dont do this in your code!!!!!
            train_cost, out, norm = train(x_batch, y_batch, mask_batch)
            print(norm)
            curcost = curcost+train_cost
        curcost = curcost/num_batches    
	print(curcost)
            

if __name__ == '__main__':
    main()