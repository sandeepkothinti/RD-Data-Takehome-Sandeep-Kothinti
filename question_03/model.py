from torch import nn

class simpleModel(nn.Module):
    def __init__(self, in_size=32):
        super(simpleModel, self).__init__()
        self.in_size = in_size
        self.cnn_layers = nn.Sequential(
                            nn.Conv2d(1,6, 5,1,2),
                            nn.ReLU(),
                            nn.MaxPool2d(2, 2), 
                            nn.Conv2d(6,10, 5,1,2),
                            nn.ReLU(), 
                            nn.MaxPool2d(2, 2),
                            nn.Conv2d(10,16, 5,1,2),
                            nn.ReLU(), 
                            nn.MaxPool2d(2, 2)
                            )
        self.out_size = 16*int(in_size/8)*int(in_size/8)
        self.fully_connected = nn.Linear(self.out_size,1)
        self.activation = nn.Sigmoid()
    
    def forward(self, images):
        conv_out = self.cnn_layers(images)
        conv_flat = conv_out.view(-1,self.out_size)
        return self.activation(self.fully_connected(conv_flat))
        