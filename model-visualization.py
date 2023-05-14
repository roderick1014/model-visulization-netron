from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--FRAMEWORK', type=str, default='pytorch')

    args = parser.parse_args()

    if args.FRAMEWORK == 'tensorflow':
        from tensorflow import keras
        import netron

        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])
        model.save('tensorflow_model.h5')
        netron.start('tensorflow_model.h5')

    elif args.FRAMEWORK == 'pytorch':
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torch.onnx
        import netron

        class model(nn.Module):

            def __init__(self):
                super(model, self).__init__()
                self.block1 = nn.Sequential(
                    nn.Conv2d(64,64,3,padding=1,bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),  # inplace saves memory.
                    nn.Conv2d(64,32,1,bias=False),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32,64,3,padding=1,bias=False),
                    nn.BatchNorm2d(64),
                )
                self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias = False)
                self.output = nn.Sequential(
                    nn.Conv2d(64,1,3,padding=1, bias = True),
                    nn.Sigmoid()
                )

            def forward(self, x):
                x = self.conv1(x)
                identity = x
                x = F.relu(self.block1(x) + identity)  # Residual mechanism
                x = self.output(x)
                return x

        pytorch_model = model()
        vector = torch.rand(1,3,416,416)
        torch.onnx.export(pytorch_model, vector, 'pytorch_model.onnx')
        netron.start('pytorch_model.onnx')