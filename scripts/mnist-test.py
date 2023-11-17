import struct
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms


class Net(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # self.dropout1 = nn.Dropout(0.25)
        # self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def inference():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]
    )

    device = torch.device('cpu')
    model = Net().to(device)
    model.load_state_dict(torch.load('data/mnist_cnn.pt'))
    model.eval()

    image = cv2.imread('data/7.png', cv2.IMREAD_GRAYSCALE)
    # image = cv2.resize(image, [28, 28])

    tensor = transform(image)
    tensor = torch.unsqueeze(tensor, 0)
    tensor = tensor.to(device)

    print(tensor.max())
    print(tensor.min())

    with torch.no_grad():
        output = model.forward(tensor)
        result = output.argmax(dim=1, keepdim=True)
        print('inference result = {}'.format(result.item()))


def tofile(f, t_name, tensor, t_type=np.float32):
    tensor = tensor.astype(t_type)
    
    t_shape = tensor.shape
    t_n_dims = len(t_shape)
    t_dtype = tensor.dtype

    print('processing: {} with shape {}, {}.'.format(t_name, t_shape, t_dtype))

    f.write(struct.pack('i', t_n_dims))
    for i in range(t_n_dims):
        f.write(struct.pack('i', t_shape[t_n_dims - 1 - i]))  # 0 1 2 3 -> 3 2 1 0
    tensor.tofile(f)


def convert():
    device = torch.device('cpu')
    model = Net().to(device)
    model.load_state_dict(torch.load('data/mnist_cnn.pt'))
    model.eval()

    with open('data/mnist_ggml.bin', 'wb') as f:
        state_dict = model.state_dict()

        # magic: ggml in hex
        f.write(struct.pack('i', 0x67676D6C))

        # conv1.weight
        t_name = "conv1.weight"
        tensor = state_dict[t_name].numpy()
        tofile(f, t_name, tensor, np.float16)
        # conv1.bias
        t_name = "conv1.bias"
        tensor = state_dict[t_name]
        tensor = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(tensor, 0), -1), -1) # [32] -> [1, 32, 1, 1]
        tensor = tensor.expand(-1, -1, 26, 26) # [1, 32, 1, 1] -> [1, 32, 26, 26]
        tensor = tensor.numpy()
        tofile(f, t_name, tensor)
        # conv2.weight
        t_name = "conv2.weight"
        tensor = state_dict[t_name].numpy()
        tofile(f, t_name, tensor, np.float16)
        # conv2.bias
        t_name = "conv2.bias"
        tensor = state_dict[t_name]
        tensor = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(tensor, 0), -1), -1) # [64] -> [1, 64, 1, 1]
        tensor = tensor.expand(-1, -1, 24, 24) # [1, 64, 1, 1] -> [1, 64, 24, 24]
        tensor = tensor.numpy()
        tofile(f, t_name, tensor)
        # fc1.weight
        t_name = "fc1.weight"
        tensor = state_dict[t_name].numpy()
        tofile(f, t_name, tensor)
        # fc1.bias
        t_name = "fc1.bias"
        tensor = state_dict[t_name].numpy()
        tofile(f, t_name, tensor)
        # fc2.weight
        t_name = "fc2.weight"
        tensor = state_dict[t_name].numpy()
        tofile(f, t_name, tensor)
        # fc2.bias
        t_name = "fc2.bias"
        tensor = state_dict[t_name].numpy()
        tofile(f, t_name, tensor)

        print('done!')


if __name__ == '__main__':
    inference()
    convert()
