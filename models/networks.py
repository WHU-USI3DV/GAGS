import torch
from torch import nn
import torch.nn.functional as F
import warnings
import math

class MLP_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        #self.fc1 = nn.Linear(512, 128).cuda()
        #self.fc2 = nn.Linear(128, 16).cuda()

        ####self.fc = nn.Linear(512, 32).cuda()
        self.fc = nn.Linear(512, 512).cuda()
        
        # Set the layer to be non-learnable
        # for param in self.fc.parameters():
        #     param.requires_grad = False
        
        seed = 42
        torch.manual_seed(seed)
        ####self.fixed_weights = torch.randn(32, 512).cuda()
        ####self.fixed_bias = torch.randn(32).cuda()
        self.fixed_weights = (torch.randn(512, 512)*0.001).cuda()
        self.fixed_bias = (torch.randn(512)*0.001).cuda()

        #print('max###', self.fixed_weights.max())
        #print('min###', self.fixed_weights.min())

        self.fc.weight = nn.Parameter(self.fixed_weights, requires_grad=False)
        self.fc.bias = nn.Parameter(self.fixed_bias, requires_grad=False)

    def forward(self, x):
        #x = torch.relu(self.fc1(x))    
        
        #x = self.fc1(x)
        #x = self.fc2(x)

        x = self.fc(x)

        return x
    

# class MLP_encoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc = nn.Linear(512, 512).cuda()

#         # Set the weight and bias such that it performs an identity operation
#         with torch.no_grad():
#             self.fc.weight.fill_(0.01)  # Adjust the weight to control the scaling
#             self.fc.bias.fill_(0.0)  # No bias is added for simplicity

#     def forward(self, x):

#         #x = self.fc(x)

#         return x



def sf_linear_transformation(input_tensor, output_dim=512):
    seed = 42
    torch.manual_seed(seed)
    transformation_matrix = torch.randn(input_tensor.shape[0], input_tensor.shape[-1], output_dim).requires_grad_(False).cuda()
    output_tensor = torch.bmm(input_tensor, transformation_matrix)
    
    return output_tensor 

def fmap_linear_transformation(input_tensor, output_dim):
    seed = 42
    torch.manual_seed(seed)
    transformation_matrix = torch.randn(input_tensor.shape[0], output_dim).requires_grad_(False).cuda() #(16, 512)
    reshaped_tensor = input_tensor.view(input_tensor.shape[0], -1) #(16,360,480) -> (16, 360*480)
    transformed_tensor = torch.matmul(transformation_matrix.T, reshaped_tensor) # (512, 16) (16, 360*480) -> (512, 360*480)
    output_tensor = transformed_tensor.view(-1, input_tensor.shape[1], input_tensor.shape[2])
    
    return output_tensor 

########################################################################################

class MLP_decoder(nn.Module):
    def __init__(self, feature_out_dim):
        super().__init__()
        self.output_dim = feature_out_dim
        #self.fc1 = nn.Linear(16, 128).cuda()
        #self.fc1 = nn.Linear(128, 128).cuda()
        #self.fc2 = nn.Linear(128, self.output_dim).cuda()

        #self.fc0 = nn.Linear(16, 16).cuda()
        #self.fc1 = nn.Linear(16, 32).cuda()
        #self.fc2 = nn.Linear(32, 64).cuda()
        #self.fc3 = nn.Linear(128, 128).cuda()
        self.fc4 = nn.Linear(128, 256).cuda()

    def forward(self, x):
        input_dim, h, w = x.shape
        x = x.permute(1,2,0).contiguous().view(-1, input_dim) #(16,48,64)->(48,64,16)->(48*64,16)
        #x = torch.relu(self.fc0(x))
        #x = torch.relu(self.fc1(x))
        #x = self.fc2(x)
        #x = torch.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = x.view(h, w, self.output_dim).permute(2, 0, 1).contiguous()
        return x


class CNN_decoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        # decoder_hidden_dims=[32, 64, 128, 256, output_dim] # default
        # decoder_hidden_dims=[256, 256, 256, 256, 256, 256, 256, 256, output_dim] 
        decoder_layers = []
        # for i in range(len(decoder_hidden_dims)):
        #     if i == 0:
        #         decoder_layers.append(nn.Conv2d(input_dim, decoder_hidden_dims[i], kernel_size=1).cuda()) # c, h, w
        #     else:
        #         # decoder_layers.append(torch.nn.BatchNorm2d(decoder_hidden_dims[i-1]))
        #         decoder_layers.append(nn.ReLU().cuda())
        #         decoder_layers.append(nn.Conv2d(decoder_hidden_dims[i-1], decoder_hidden_dims[i], kernel_size=1).cuda())
        
        # 256 * 8 
        # cat
        # decoder_hidden_dims_in=[input_dim, 256, 256, 512, 256, 256, 512, 256, 256]
        # + 
        decoder_hidden_dims_in=[input_dim, 256, 256, 256, 256, 256, 256, 256, 256]
        decoder_hidden_dims_out=[256,      256, 256, 256, 256, 256, 256, 256, output_dim]
        #                                             |              |
        # 2 output 512 768
        # decoder_hidden_dims_in=[input_dim, 256, 256, 256, 256, 256, 256, 768, 256]
        # decoder_hidden_dims_out=[256,      256, 256, 256, 256, 256, 768, 256, output_dim]
        # shallow mlp
        # decoder_hidden_dims_in=[input_dim, 32, 64, 128, 256, 256]
        # decoder_hidden_dims_out=[32,      64, 128, 256, 256, output_dim]
        
        # 64*3  -> 128*3 -> 256*3 -> 512
        # decoder_hidden_dims_in=[input_dim, 64, 64, 64, 128, 128, 128, 256, 256, 256, 512]
        # decoder_hidden_dims_out=[64,       64, 64, 128, 128, 128, 256, 256, 256, 512, output_dim]
        # decoder_hidden_dims_in=[input_dim, 32, 32, 32, 64, 64, 64, 128, 128, 128, 256]
        # decoder_hidden_dims_out=[32,       32, 32, 64, 64, 64, 128, 128, 128, 256, output_dim]       
        for i in range(len(decoder_hidden_dims_in)):
            if i == 0:
                decoder_layers.append(nn.Conv2d(decoder_hidden_dims_in[i], decoder_hidden_dims_out[i], kernel_size=1).cuda()) # c, h, w
            else:
                # decoder_layers.append(torch.nn.BatchNorm2d(decoder_hidden_dims[i-1]))
                decoder_layers.append(nn.ReLU().cuda())
                decoder_layers.append(nn.Conv2d(decoder_hidden_dims_in[i], decoder_hidden_dims_out[i], kernel_size=1).cuda())
        
        self.decoder = nn.ModuleList(decoder_layers)
        print(self.decoder)
        
        # original
        # self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=1).cuda()


    def forward(self, x):
        
        # original
        # x = self.conv(x)
        
        # for m in self.decoder:
        #     x = m(x)
        # x = F.normalize(x, dim=0) # new added
        
        # 256*8
        x1 = self.decoder[0](x) # 16 -> 256
        x1 = self.decoder[1](x1) # relu
        
        x2 = self.decoder[2](x1) # 256 -> 256
        x2 = self.decoder[3](x2) # relu
        x2 = self.decoder[4](x2) # 256 -> 256
        x2 = self.decoder[5](x2) # relu
        
        # x3 = torch.concat((x1, x2), dim=0) # 256+256 -> 512
        x3 = x1+x2
        x3 = self.decoder[6](x3) # 256 -> 256
        x3 = self.decoder[7](x3) # relu
        
        x4 = self.decoder[8](x3) # 256 -> 256
        x4 = self.decoder[9](x4) # relu
        x4 = self.decoder[10](x4) # 256 -> 256
        x4 = self.decoder[11](x4) # relu
        
        # x5 = torch.concat((x3, x4), dim=0) # 256+256 -> 512
        x5 = x3+x4
        x5 = self.decoder[12](x5) # 256 -> 256
        x5 = self.decoder[13](x5) # relu
        x5 = self.decoder[14](x5) # 256 -> 256
        x5 = self.decoder[15](x5) # relu
        x5 = self.decoder[16](x5) # 256 -> outdim
        
        x_out = F.normalize(x5, dim=0)
        
        # x1 = self.decoder[0](x) # 16 -> 256
        # x1 = self.decoder[1](x1) # relu
        # x1 = self.decoder[2](x1) # 256 -> 256
        # x1 = self.decoder[3](x1) # relu
        # x1 = self.decoder[4](x1) # 256 -> 256
        # x1 = self.decoder[5](x1) # relu
        # x1 = self.decoder[6](x1) # 256 -> 256
        # x1 = self.decoder[7](x1) # relu
        # x1 = self.decoder[8](x1) # 256 -> 256
        # x1 = self.decoder[9](x1) # relu
        # x1 = self.decoder[10](x1) # 256 -> 256
        # x1 = self.decoder[11](x1) # relu
        # x1 = self.decoder[12](x1) # 256 -> 768
        
        # x2 = self.decoder[13](x1) # relu
        # x2 = self.decoder[14](x2) # 768 -> 256
        # x2 = self.decoder[15](x2) # relu
        # x2 = self.decoder[16](x2) # 256 -> 512

        # x1_out = F.normalize(x1, dim=0)
        # x2_out = F.normalize(x2, dim=0)

        return x_out
    
class CNN_scale_decoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        # decoder_hidden_dims=[32, 32, output_dim] # before 08/02/2024
        decoder_hidden_dims=[64, 128, 64, 32, 16, output_dim] 
        decoder_layers = []
        for i in range(len(decoder_hidden_dims)):
            if i == 0:
                decoder_layers.append(nn.Conv2d(input_dim, decoder_hidden_dims[i], kernel_size=1).cuda()) # c, h, w
            else:
                # decoder_layers.append(torch.nn.BatchNorm2d(decoder_hidden_dims[i-1]))
                decoder_layers.append(nn.ReLU().cuda())
                decoder_layers.append(nn.Conv2d(decoder_hidden_dims[i-1], decoder_hidden_dims[i], kernel_size=1).cuda())
        self.decoder = nn.ModuleList(decoder_layers)
        print(self.decoder)

    def forward(self, x):

        for m in self.decoder:
            x = m(x)
        x = F.softmax(x,dim=0)
        
        # x = F.normalize(x, dim=0) # new added
        
        # original
        # x = self.conv(x)
        
        return x

        