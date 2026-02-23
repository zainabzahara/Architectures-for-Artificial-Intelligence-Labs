import torch

# Parameters
IN_CH  = 16
OUT_CH = 32
KER_HW = 3
IN_H   = 28
IN_W   = 28

class SingleConv(torch.nn.Module):

    def __init__(self):
        super(SingleConv, self).__init__()
        self.conv0 = torch.nn.Conv2d(in_channels=1, out_channels=IN_CH, kernel_size=1)
        self.conv1 = torch.nn.Conv2d(
            in_channels  = IN_CH,
            out_channels = OUT_CH,
            kernel_size  = KER_HW,
            stride   = 1,
            padding  = 1,
            dilation = 1
        )

    def forward(self, x):
        y = self.conv0(x)
        out = self.conv1(y)
        return out 


input_data = torch.ones(1, 1, IN_H, IN_W)
convnet = SingleConv()

for co in range(OUT_CH):
    for ci in range(IN_CH):
        for hk in range(KER_HW):
            for wk in range(KER_HW):
                convnet.conv1.weight.data[co, ci, hk, wk] = 1.0

output_data = convnet(input_data)

print(convnet)

convnet.eval()
torch.onnx.export(convnet, input_data, 'single_cnn.onnx')
