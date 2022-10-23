import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tqdm
import copy
import sys

def funckin_graph(y):
    plt.plot(np.arange(len(y)), y)
    plt.show()


def softmax(u):
    arr = np.exp(u)/np.sum(np.exp(u))
    return arr

def identify_funcion(u):
    return u

def read_image(data):
    return np.ravel(data)

def choice_random_by_data(data, label, bathces):
    #dataとlabelは同じ長さで
    length_of_data = len(data)
    indexes = np.random.randint(0, length_of_data, (bathces, ))
    n_data, n_label = data[indexes], label[indexes]
    return (n_data, n_label)

def combine_weight_and_output(output, weight):
    #weightは2次元，outputは1次元
    arr = np.array([np.sum(output * j) for j in weight])
    return arr

def get_delta_from_predelta_weight_output(predelta, weight, output):
    #predelta, weightは同層，outputは前層
    arr = np.array([e * predelta[i] for i, e in enumerate(weight)])
    arr = np.array([np.sum(output[i] * e) for i, e in enumerate(weight.T)])
    return arr

def revise_weight(delta, pre_weight, output):
    #deltaは2次元，pre_Weightも2次元。
    new_arr = np.zeros((len(delta), len(pre_weight), len(pre_weight[0])))
    new_arr = np.array([pre_weight for i in new_arr])
    print(np.array(output).shape, np.array(delta).shape, pre_weight.shape)

    print((np.array(output) * np.array(delta)).shape)

    dush_new_arr = np.array([np.array(output) * np.array(delta) * i for i in np.transpose(new_arr, (1, 0, 2))])
    nyannyann = np.sum(dush_new_arr, axis=1)
    print(new_arr.shape, dush_new_arr.shape, nyannyann.shape)

def get_two_dim_elements(arr, index):
    new_arr = [i[index] for i in arr]
    return np.array(new_arr)

def make_mesh_matrix(arr1, arr2):
    new_arr = np.zeros((len(arr1), len(arr2)))
    #print(arr1.shape, arr2.shape)
    for i, e in enumerate(arr1):
        new_arr[i] = e * arr2
    return np.array(new_arr)

def make_mesh_matrix_test(arr1, arr2):
    new_arr = np.zeros((len(arr1), len(arr2)))
    for i, e in enumerate(arr1):
        new_arr[i] = (e + arr2) / 2
    return np.array(new_arr)

def main(data_num):
    data_num = data_num

    weight = []

    test_weight = []

    output = []

    test_output = []

    error = []

    test_error = []

    #重み：28^2 -> 49 -> 49 -> 10

    weight.append(np.random.random_sample((784,2))) # input layer
    weight.append(np.random.random_sample((49, 784+1))) # hidden layer 1
    weight.append(np.random.random_sample((49, 49+1))) # hidden layer 2
    weight.append(np.random.random_sample((10, 49+1))) # output layer

    #test 重み：784 -> 14 -> 14 -> 10

    test_weight.append(np.random.random_sample((784,2))) # input layer
    test_weight.append(np.random.random_sample((49+784))) # hidden layer 1
    test_weight.append(np.random.random_sample((49*2))) # hidden layer 2
    test_weight.append(np.random.random_sample((10, 49+1))) # output layer
    test_weight.append(np.random.random_sample(49))
    test_weight.append(np.random.random_sample(49))

    #重みのindex=0はバイアス

    initial_matrix = []

    initial_matrix.append(np.zeros(784))
    initial_matrix.append(np.zeros(49))
    initial_matrix.append(np.zeros(49))
    initial_matrix.append(np.zeros(10))

    mnist = tf.keras.datasets.mnist
    (train_image_data, train_label_data), (test_image_data, test_label_data) = mnist.load_data()

    train_image_data = train_image_data / 255.0
    test_image_data = test_image_data / 255.0

    epochs = 50
    batches = 15
    learning_rate = 0.01

    sleep_time = 10e-5

    print('---END STARTUP---')

    loss = []
    test_loss = []

    for i in range(epochs):

        output = [copy.deepcopy(initial_matrix) for nyannnyan in range(batches)]

        test_output = [copy.deepcopy(initial_matrix) for meow in range(batches)]

        delta = [copy.deepcopy(initial_matrix) for nyago in range(batches)]

        test_delta = [copy.deepcopy(initial_matrix) for nyan in range(batches)]

        data, labels = choice_random_by_data(train_image_data, train_label_data, batches)

        images = []

        #error = []

        loss_disposable = []

        test_loss_disposable = []

        for j in tqdm.tqdm(range(batches)):
            image = read_image(data[j])
            images.append(image)
            output[j][0] = softmax( combine_weight_and_output(image, weight[0][:, 1:]) + weight[0][:, 0] )
            #print((combine_weight_and_output(image, weight[0][:, 1:])).shape)
            output[j][1] = softmax( combine_weight_and_output(output[j][0], weight[1][:, 1:]) + weight[1][:, 0] )
            output[j][2] = softmax( combine_weight_and_output(output[j][1], weight[2][:, 1:]) + weight[2][:, 0] )
            #print(combine_weight_and_output(output[j][2], weight[3][:, 1:]))
            output[j][3] = identify_funcion( combine_weight_and_output(output[j][2], weight[3][:, 1:]) + weight[3][:, 0] )

            test_output[j][0] = softmax( combine_weight_and_output(image, test_weight[0][:, 1:]) + test_weight[0][:, 0] )
            test_output[j][1] = softmax( combine_weight_and_output(test_output[j][0], make_mesh_matrix_test(test_weight[1][:49], test_weight[1][49:])) + test_weight[4] )
            test_output[j][2] = softmax( combine_weight_and_output(test_output[j][1], make_mesh_matrix_test(test_weight[2][:49], test_weight[2][49:])) + test_weight[5] )
            test_output[j][3] = identify_funcion( combine_weight_and_output(test_output[j][2], test_weight[3][:, 1:]) + test_weight[3][:, 0] )

            delta[j][3] = output[j][3] - labels[j]
            #print(delta[j][3])
            loss_disposable.append(np.mean(delta[j][3]))
            delta[j][2] = get_delta_from_predelta_weight_output(delta[j][3], weight[3][:, 1:], output[j][2])
            delta[j][1] = get_delta_from_predelta_weight_output(delta[j][2], weight[2][:, 1:], output[j][1])
            delta[j][0] = get_delta_from_predelta_weight_output(delta[j][1], weight[1][:, 1:], read_image(data[j]))

            test_delta[j][3] = test_output[j][3] - labels[j]
            test_loss_disposable.append(np.mean(test_delta[j][3]))
            test_delta[j][2] = get_delta_from_predelta_weight_output(test_delta[j][3], test_weight[3][:, 1:], test_output[j][2])
            test_delta[j][1] = get_delta_from_predelta_weight_output(test_delta[j][2], make_mesh_matrix_test(test_weight[2][:49], test_weight[2][49:]), test_output[j][1])
            test_delta[j][0] = get_delta_from_predelta_weight_output(test_delta[j][1], make_mesh_matrix_test(test_weight[1][:49], test_weight[1][49:]), read_image(data[j]))

        loss.append(sum(loss_disposable) / batches)
        test_loss.append(sum(test_loss_disposable) / batches)

        #bias の更新
        weight[0][:, 0] = weight[0][:, 0] - learning_rate * (1 / batches) * np.sum(np.array([delta[k][0] for k in range(batches)]), axis=0)
        weight[1][:, 0] = weight[1][:, 0] - learning_rate * (1 / batches) * np.sum(np.array([delta[k][1] for k in range(batches)]), axis=0)
        weight[2][:, 0] = weight[2][:, 0] - learning_rate * (1 / batches) * np.sum(np.array([delta[k][2] for k in range(batches)]), axis=0)
        weight[3][:, 0] = weight[3][:, 0] - learning_rate * (1 / batches) * np.sum(np.array([delta[k][3] for k in range(batches)]), axis=0)

        test_weight[0][:, 0] = test_weight[0][:, 0] - learning_rate * (1 / batches) * np.sum(np.array([test_delta[k][0] for k in range(batches)]), axis=0)
        test_weight[4] = test_weight[4] - learning_rate * (1 / batches) * np.sum(np.array([test_delta[k][1] for k in range(batches)]), axis=0)
        test_weight[5] = test_weight[5] - learning_rate * (1 / batches) * np.sum(np.array([test_delta[k][2] for k in range(batches)]), axis=0)
        test_weight[3][:, 0] = test_weight[3][:, 0] - learning_rate * (1 / batches) * np.sum(np.array([test_delta[k][3] for k in range(batches)]), axis=0)

        #weightの更新
        #print(delta[0][0].shape, images[0].shape, (delta[0][0] * images[0]).shape, weight[0][:, 1:].shape)
        weight[0][:, 1] = weight[0][:, 1] - learning_rate * (1 / batches) * np.sum(np.array([delta[k][0] * images[k] for k in range(batches)]), axis=0)
        weight[1][:, 1:] = weight[1][:, 1:] - learning_rate * (1 / batches) * np.sum(np.array([make_mesh_matrix(delta[k][1], output[k][0]) for k in range(batches)]), axis=0)
        weight[2][:, 1:] = weight[2][:, 1:] - learning_rate * (1 / batches) * np.sum(np.array([make_mesh_matrix(delta[k][2], output[k][1]) for k in range(batches)]), axis=0)
        weight[3][:, 1:] = weight[3][:, 1:] - learning_rate * (1 / batches) * np.sum(np.array([make_mesh_matrix(delta[k][3], output[k][2]) for k in range(batches)]), axis=0)

        test_weight[0][:, 1:] = test_weight[0][:, 1:] - learning_rate * (1 / batches) * np.sum(np.array([test_delta[k][0] * images[k] for k in range(batches)]), axis=0)[0]
        test_weight[3][:, 1:] = test_weight[3][:, 1:] - learning_rate * (1 / batches) * np.sum(np.array([make_mesh_matrix(test_delta[k][3], test_output[k][2]) for k in range(batches)]), axis=0)[0]
        multiplied_delta_and_input_hidden1 = np.sum(np.array([make_mesh_matrix(test_delta[k][1], test_output[k][0]) for k in range(batches)]), axis=0)
        multiplied_delta_and_input_hidden2 = np.sum(np.array([make_mesh_matrix(test_delta[k][2], test_output[k][1]) for k in range(batches)]), axis=0)
        #print(multiplied_delta_and_input_hidden1.shape)
        test_weight[1][:49] = test_weight[1][:49] - learning_rate * (1 / batches) * np.sum(multiplied_delta_and_input_hidden1, axis=1)
        test_weight[1][49:] = test_weight[1][49:] - learning_rate * (1 / batches) * np.sum(multiplied_delta_and_input_hidden1, axis=0)
        test_weight[2][:49] = test_weight[2][:49] - learning_rate * (1 / batches) * np.sum(multiplied_delta_and_input_hidden2, axis=1)
        test_weight[2][49:] = test_weight[2][49:] - learning_rate * (1 / batches) * np.sum(multiplied_delta_and_input_hidden2, axis=0)




        print('epochs:' + str(i) + '/' + str(epochs) + 'loss average:' + str(sum(loss_disposable) / batches))
        
        plt.plot(np.arange(len(loss)), loss)
        plt.plot(np.arange(len(test_loss)), test_loss)
        plt.draw()
        plt.pause(sleep_time)
        plt.cla()

    plt.savefig('experience_data/graph_fig_{}_learning_rate={}_epochs={}_bathces={}.png'.format(data_num, learning_rate, epochs, batches))

    np.save('experience_data/data_loss_{}_learning_rate={}_epochs={}_bathces={}'.format(data_num, learning_rate, epochs, batches), np.array(loss))
    np.save('experience_data/data_test_loss_{}_learning_rate={}_epochs={}_bathces={}'.format(data_num, learning_rate, epochs, batches), np.array(test_loss))