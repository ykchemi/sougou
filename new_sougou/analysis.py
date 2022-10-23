import numpy as np

loss_sum_001 = np.zeros((50, 50))
test_loss_sum_001 = np.zeros((50, 50))
loss_sum_005 = np.zeros((50, 50))
test_loss_sum_005 = np.zeros((50, 50))


for i in range(50):
    a = np.load('experience_data/data_loss_{}_learning_rate=0.01_epochs=50_bathces=15.npy'.format(i))
    b = np.load('experience_data/data_test_loss_{}_learning_rate=0.01_epochs=50_bathces=15.npy'.format(i))
    c = np.load('experience_data/data_loss_{}_learning_rate=0.05_epochs=50_bathces=15.npy'.format(i))
    d = np.load('experience_data/data_test_loss_{}_learning_rate=0.05_epochs=50_bathces=15.npy'.format(i))
    loss_sum_001[i] = a
    test_loss_sum_001[i] = b
    loss_sum_005[i] = c
    test_loss_sum_005[i] = d

np.savetxt('loss_001.txt', loss_sum_001)
np.savetxt('loss_test_001.txt', test_loss_sum_001)
np.savetxt('loss_005.txt', loss_sum_005)
np.savetxt('loss_test_005.txt', test_loss_sum_005)
    
