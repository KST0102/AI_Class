# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import gzip
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from simple_convnet import SimpleConvNet
from common.trainer import Trainer


def load_fashion_mnist_images(file_path):
    """이미지 데이터를 불러옵니다."""
    with gzip.open(file_path, 'rb') as f:
        # 첫 16바이트는 헤더
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 1, 28, 28)

def load_fashion_mnist_labels(file_path):
    """라벨 데이터를 불러옵니다."""
    with gzip.open(file_path, 'rb') as f:
        # 첫 8바이트는 헤더
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

# 데이터셋 경로 설정
dataset_dir = 'fashion'
train_images_path = os.path.join(dataset_dir, 'train-images-idx3-ubyte.gz')
train_labels_path = os.path.join(dataset_dir, 'train-labels-idx1-ubyte.gz')
test_images_path = os.path.join(dataset_dir, 't10k-images-idx3-ubyte.gz')
test_labels_path = os.path.join(dataset_dir, 't10k-labels-idx1-ubyte.gz')

# 데이터 불러오기
x_train = load_fashion_mnist_images(train_images_path)
t_train = load_fashion_mnist_labels(train_labels_path)
x_test = load_fashion_mnist_images(test_images_path)
t_test = load_fashion_mnist_labels(test_labels_path)

# mnist 데이터 읽기
#(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

# 시간이 오래 걸릴 경우 데이터를 줄인다.
x_train, t_train = x_train[:5000], t_train[:5000]
x_test, t_test = x_test[:1000], t_test[:1000]

max_epochs = 20

network = SimpleConvNet(input_dim=(1,28,28), 
                        conv_param = {'filter_num': 50, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=50, output_size=10, weight_init_std=0.01, use_dropout=True, dropout_ration=0.4)
                        
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

# 매개변수 보존
network.save_params("params.pkl")
print("Saved Network Parameters!")

# 그래프 그리기
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
