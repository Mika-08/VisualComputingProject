import torch
from torchvision import transforms
from PIL import Image
from torch import nn

# Define the provided classes
class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=True)
        self.relu = nn.ReLU() if relu else None
        self.bn = nn.BatchNorm2d(out_dim) if bn else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim/2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim/2))
        self.conv2 = Conv(int(out_dim/2), int(out_dim/2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim/2))
        self.conv3 = Conv(int(out_dim/2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        self.need_skip = inp_dim != out_dim

    def forward(self, x):
        residual = self.skip_layer(x) if self.need_skip else x
        out = self.relu(self.bn1(x))
        out = self.relu(self.bn2(self.conv1(out)))
        out = self.relu(self.bn3(self.conv2(out)))
        out = self.conv3(out)
        return out + residual

class Hourglass(nn.Module):
    def __init__(self, n, f, bn=None, increase=0):
        super(Hourglass, self).__init__()
        nf = f + increase
        self.up1 = Residual(f, f)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.low1 = Residual(f, nf)
        self.n = n
        self.low2 = Hourglass(n-1, nf, bn=bn) if self.n > 1 else Residual(nf, nf)
        self.low3 = Residual(nf, f)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        return self.up1(x) + self.up2(self.low3(self.low2(self.low1(self.pool1(x)))))

# Create model instance
model = Hourglass(n=4, f=3)

# Load the trained weights
state_dict = torch.load("Model/Hourglass 2HG/checkpoint.pt", map_location="cpu")
model.load_state_dict(state_dict['state_dict'], strict=False)

model.eval()


# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load and process an image
image_path = "Images/099548361.jpg"
image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Run inference
with torch.no_grad():
    output = model(input_tensor)

print(output)