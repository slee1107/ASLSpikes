import nengo
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import nengo_dl
import os
from PIL import Image

# Directory names holding input data
root_dir_train = './mydata/training_set'
root_dir_test = './mydata/test_set'
root_dir_ran = './mydata/random_set'

train_data = [[] for x in range(2)]
test_data = [[] for x in range(2)]

# Process training data
for i in range(1750):
    for letterEntry in os.scandir(root_dir_train):
        if not letterEntry.name.startswith('.') and letterEntry.is_dir():
            # for img in os.scandir(letterEntry.path):
            #     if img.name == str(i+1) + '.png':
            # Add label array to training_data[1]
            label_arr = np.zeros(26)
            label_arr[ord(letterEntry.name)-65] = 1
            # np.append(train_data[1],label_arr)
            train_data[1].append(label_arr)

            # lc_img = Image.open(img.path).convert('L')
            lc_img = Image.open('./mydata/training_set/'+letterEntry.name+'/'+str(i+1)+'.png').convert('L')
            
            img_arr = np.asarray(lc_img)
            # np.append(train_data[0],img_arr)
            train_data[0].append(img_arr.flatten())

# print(train_data)
# USE FOR ACTUAL TEST
# Process test data CHANGE BACK TO TEST
expected_arr = []
for i in range(250):
    for letterEntry in os.scandir(root_dir_test):
        if not letterEntry.name.startswith('.') and letterEntry.is_dir():

            # for img in os.scandir(letterEntry.path):
            # Add label array to training_data[1]
            label_arr = np.zeros(26)
            
            # For test_set
            label_arr[ord(letterEntry.name)-65] = 1
            expected_arr.append(np.argmax(label_arr))
            # print(expected_arr)

            test_data[1].append(label_arr)
            # np.append(test_data[1],label_arr)

            # lc_img = Image.open(img.path).convert('L')
            lc_img = Image.open('./mydata/test_set/'+letterEntry.name+'/'+str(i+1)+'.png').convert('L')

            img_arr = np.asarray(lc_img)
            test_data[0].append(img_arr.flatten())
            # np.append(test_data[0],img_arr.flatten())

# print(test_data)
# print(expected_arr)
# print(len(expected_arr))
# exit()

# # Process test data with RANDOM TEST
# for letterEntry in os.scandir(root_dir_ran):
#     if not letterEntry.name.startswith('.'):
#         # Add label array to training_data[1]
#         label_arr = np.zeros(26)
        
#         # For test_set
#         # label_arr[ord(letterEntry.name)-65] = 1
#         print(letterEntry.name)
#         # For random set
#         label_arr[ord(letterEntry.name[1:2])-97] = 1

#         test_data[1].append(label_arr)
#         # np.append(test_data[1],label_arr)

#         lc_img = Image.open(letterEntry.path).convert('L')
#         img_arr = np.asarray(lc_img)
#         test_data[0].append(img_arr.flatten())


# print(input_data)

# for i in range(3):
#     plt.figure()
#     plt.imshow(np.reshape(train_data[0][i], (64, 64)),
#                cmap="gray")
#     plt.axis('off')
#     plt.title(str(np.argmax(train_data[1][i])))
#     plt.show()

with nengo.Network() as net:
    # set some default parameters for the neurons that will make
    # the training progress more smoothly
    net.config[nengo.Ensemble].max_rates = nengo.dists.Choice([100])
    net.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
    neuron_type = nengo.LIF(amplitude=0.01)

    # we'll make all the nengo objects in the network
    # non-trainable. we could train them if we wanted, but they don't
    # add any representational power. note that this doesn't affect
    # the internal components of tensornodes, which will always be
    # trainable or non-trainable depending on the code written in
    # the tensornode.
    nengo_dl.configure_settings(trainable=False)

    # the input node that will be used to feed in input images
    inp = nengo.Node([0] * 64 * 64)

    # add the first convolutional layer
    x = nengo_dl.tensor_layer(
        inp, tf.layers.conv2d, shape_in=(64, 64, 1), filters=32,
        kernel_size=3)

    # apply the neural nonlinearity
    x = nengo_dl.tensor_layer(x, neuron_type)

    # add another convolutional layer
    x = nengo_dl.tensor_layer(
        x, tf.layers.conv2d, shape_in=(62, 62, 32),
        filters=64, kernel_size=3)
    x = nengo_dl.tensor_layer(x, neuron_type)

    # add a pooling layer
    x = nengo_dl.tensor_layer(
        x, tf.layers.average_pooling2d, shape_in=(60, 60, 64),
        pool_size=2, strides=2)

    # another convolutional layer
    x = nengo_dl.tensor_layer(
        x, tf.layers.conv2d, shape_in=(30, 30, 64),
        filters=128, kernel_size=3)
    x = nengo_dl.tensor_layer(x, neuron_type)

    # another pooling layer
    x = nengo_dl.tensor_layer(
        x, tf.layers.average_pooling2d, shape_in=(28, 28, 128),
        pool_size=2, strides=2)

    # linear readout
    x = nengo_dl.tensor_layer(x, tf.layers.dense, units=26)

    # we'll create two different output probes, one with a filter
    # (for when we're simulating the network over time and
    # accumulating spikes), and one without (for when we're
    # training the network using a rate-based approximation)
    out_p = nengo.Probe(x)
    out_p_filt = nengo.Probe(x, synapse=0.1)
    
    print(x.probeable)

    # Adding time steps
    minibatch_size = 200
    # minibatch_size = 50

    sim = nengo_dl.Simulator(net, minibatch_size=minibatch_size)

    train_data = {inp: np.asarray(train_data[0])[:, None,:],
                out_p: np.asarray(train_data[1])[:, None,:]}
    
    n_steps = 30
    # n_steps = 10
    # print(test_data)
    test_data = {
        inp: np.tile(np.asarray(test_data[0])[:minibatch_size*2, None, :],
                    (1, n_steps, 1)),
        out_p_filt: np.tile(np.asarray(test_data[1])[:minibatch_size*2, None, :],
                            (1, n_steps, 1))}
    # print(test_data)
    # Define objective and classification error functions
    def objective(outputs, targets):
        return tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=outputs, labels=targets)
    
    opt = tf.train.RMSPropOptimizer(learning_rate=0.001)

    def classification_error(outputs, targets):
        return 100 * tf.reduce_mean(
            tf.cast(tf.not_equal(tf.argmax(outputs[:, -1], axis=-1),
                                tf.argmax(targets[:, -1], axis=-1)),
                    tf.float32))
    
    for key in test_data:
        test_data[key] = np.asarray(test_data[key])
    print("error before training: %.8f%%" % sim.loss(
        test_data, {out_p_filt: classification_error}))
    accuracy_before = 100 - sim.loss(test_data, {out_p_filt: classification_error})
    print("accuracy before training: %.8f%%" % accuracy_before)

    do_training = False
    if do_training:
        # run training
        sim.train(train_data, opt, objective={out_p: objective}, n_epochs=10)

        # save the parameters to file
        sim.save_params("./asl_params")
    else:
        sim.load_params("./asl_params")

    print("error after training: %.8f%%" % sim.loss(
        test_data, {out_p_filt: classification_error}))
    accuracy_after = 100 - sim.loss(test_data, {out_p_filt: classification_error})

    print("accuracy after training: %.8f%%" % accuracy_after)

    # Print out probe attributes after training
    print(out_p_filt)
    output_probe = nengo.Probe(x, 'output',synapse=0.1)
    # Initialize confusion matrix
    confusion = np.zeros((26, 26))

    # Plotting, change to test_data[inp][:] and in range(larger size) to get more of the
    # Output to print out
    sim.run_steps(n_steps, data={inp: test_data[inp][:minibatch_size]})

    for i in range(200):
        expected = expected_arr[i]
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(np.reshape(test_data[inp][i, 0], (64, 64)),
                cmap="gray")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.plot(sim.trange(), sim.data[out_p_filt][i])
        print('Sim Data:\n')
        print(sim.data[out_p_filt][i])

        print('Before creating confusion matrix')
        print(sim.data[out_p_filt])
        print(sim.data[output_probe])
        # Confusion matrix creation
        # expected = np.argmax(test_data['Probe'].attributes)
        output_data = np.sum(sim.data[out_p_filt][i], axis=0) # output for the specific image we're testing 
        actual = np.argmax(output_data)
        confusion[expected_arr[i]][actual] = confusion[expected_arr[i]][actual] + 1

        print(confusion)
        
        plt.legend([str(i) for i in range(26)], loc="upper left")
        plt.xlabel("time")
        # plt.show()
    print(out_p_filt)
    print('end')

    # def reduce_maxLetter(data):
    #     maxLetter = np.zeros(26)
    #     for i in range(26):
    #         maxLetter[i] = np.sum(data[i])
        
        
    

