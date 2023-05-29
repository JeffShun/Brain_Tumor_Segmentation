import torch
import torch.nn as nn


class ConvBlock(nn.Module):
	def __init__(self,inchannel,outchannel):
		super(ConvBlock,self).__init__()
		self.left = nn.Sequential(
				nn.BatchNorm3d(inchannel),
				nn.ReLU(inplace=True),
				nn.Conv3d(inchannel,outchannel,kernel_size=3,stride=1,padding=1),
				nn.BatchNorm3d(outchannel),
				nn.ReLU(inplace=True),
				nn.Conv3d(outchannel,outchannel,kernel_size=3,stride=1,padding=1),
				)
		self.right = nn.Conv3d(inchannel,outchannel,kernel_size=1)
	def forward(self,x):
		left = self.left(x)
		right = self.right(x)
		return left+right


def maxpool(x):
	f = torch.nn.MaxPool3d(2,stride=2,return_indices=False,ceil_mode=False)
	return f(x)



class Unet3D(nn.Module):
	def __init__(self,indim,base_channel):
		super(Unet3D,self).__init__() 
		self.layer1 = ConvBlock(indim,base_channel)
		self.layer2 = ConvBlock(base_channel,base_channel*2)
		self.layer3 = ConvBlock(base_channel*2,base_channel*4)
		self.layer4 = ConvBlock(base_channel*4,base_channel*8)

		self.upconv1= nn.ConvTranspose3d(base_channel*8,base_channel*4,kernel_size=2,stride=2)
		self.layer3_= ConvBlock(base_channel*4,base_channel*4)
		self.upconv2= nn.ConvTranspose3d(base_channel*4,base_channel*2,kernel_size=2,stride=2)
		self.layer2_= ConvBlock(base_channel*2,base_channel*2)
		self.upconv3= nn.ConvTranspose3d(base_channel*2,base_channel,kernel_size=2,stride=2)
		self.layer1_= ConvBlock(base_channel,base_channel)
		
	def forward(self,x):
		x1 = self.layer1(x)
		x2 = self.layer2(maxpool(x1))
		x3 = self.layer3(maxpool(x2))
		x4 = self.layer4(maxpool(x3))
		x3_= self.layer3_(x3+self.upconv1(x4))
		x2_= self.layer2_(x2+self.upconv2(x3_))
		x1_= self.layer1_(x1+self.upconv3(x2_))
		return x1_,x2_,x3_,x4


class Cascaded_Unet(nn.Module):
	def __init__(self,indim, base_channel):
		super(Cascaded_Unet,self).__init__()
		self.unet = Unet3D(indim,base_channel)
		
		self.layer1 = ConvBlock(base_channel,base_channel*2)
		
		self.layer2 = ConvBlock(base_channel*2,base_channel*4)

		self.layer3 = ConvBlock(base_channel*4,base_channel*8)

		self.layer4 = ConvBlock(base_channel*8,base_channel*16)

		self.upconv1= nn.ConvTranspose3d(base_channel*16,base_channel*8,kernel_size=2,stride=2)
		self.layer3_= ConvBlock(base_channel*8,base_channel*8)

		self.upconv2= nn.ConvTranspose3d(base_channel*8,base_channel*4,kernel_size=2,stride=2)
		self.layer2_= ConvBlock(base_channel*4,base_channel*4)

		self.upconv3= nn.ConvTranspose3d(base_channel*4,base_channel*2,kernel_size=2,stride=2)
		self.layer1_= ConvBlock(base_channel*2,base_channel)
		

	def forward(self,x):
		y11,y12,y13,y14 = self.unet(x)
		y1 = self.layer1(y11)
		y2 = self.layer2(maxpool(y1)+y12)
		y3 = self.layer3(maxpool(y2)+y13)
		y4 = self.layer4(maxpool(y3)+y14)

		y3_= self.layer3_(y3+self.upconv1(y4))
		y2_= self.layer2_(y2+self.upconv2(y3_))
		y1_= self.layer1_(y1+self.upconv3(y2_))
		
		return [y11, y1_]