# -基于简易神经网络的CIFAR10分类即超参数影响的探究
姓名：李秦峰    学号：22351115
### **1. 研究目的**
本研究的目标是通过 PyTorch 实现一个简易的卷积神经网络模型，并使用该模型对 CIFAR-10 数据集进行图像分类。我们旨在探索学习率、网络架构等超参数对网络分类性能的影响。代码在https://github.com/Toruforx/Data-classification-based-on-pytorch。


### **2. 方法**


**数据集：** 

CIFAR-10是一个更接近普适物体的彩色图像数据集。CIFAR-10 是由Hinton 的学生Alex Krizhevsky 和Ilya Sutskever 整理的一个用于识别普适物体的小型数据集。一共包含 10 个类别的 RGB 彩色图 片：飞机（airplane）、汽车（ automobile ）、鸟类（ bird ）、猫（ cat ）、鹿（ deer ）、狗（ dog ）、蛙类（ frog ）、马（ horse ）、船（ ship ）和卡车（ truck ）。图片的尺寸为 32×32 ，数据集中一共有 50000 张训练圄片和 10000 张测试图片。 CIFAR-10 的图片样例如图所示。
下面这幅图就是列举了10各类，每一类展示了随机的10张图片：
![image](https://github.com/feng-Li-zjut/AI_Security_Homework/assets/74243537/817fd100-3593-4aa8-9ff2-ed742ae29975)

我们使用了数据增广技术，通过填充、随机水平翻转和随机裁剪，对图像进行处理。这有助于扩充数据集，增加模型的泛化能力。
```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
```


**模型结构：** 我们设计了一个包含两个卷积层和一个全连接层的卷积神经网络，用于图像分类任务。让我们逐步介绍模型的结构：

1. **卷积层 (`conv1` 和 `conv2`):**
   - `conv1`: 接受一个3通道的输入（假设是RGB图像），应用一个2D卷积，使用16个大小为5x5的滤波器（卷积核），然后进行ReLU激活、批量归一化（Batch Normalization），最后进行2x2的最大池化。
   - `conv2`: 接受`conv1`的输出，再次应用一个2D卷积，使用32个大小为5x5的滤波器，然后进行ReLU激活、批量归一化，最后进行2x2的最大池化。
2. **全连接层 (`fc`):**
   - 将卷积层的输出展平成一维向量（通过`out.reshape`），然后通过一个全连接层进行分类。这个全连接层(`fc`)的输入大小是`8 * 8 * 32`，输出大小是`num_classes`，默认为10，表示模型要输出10个类别的概率分布。
3. **前向传播 (`forward` 方法):**
   - 在前向传播中，输入数据经过卷积层`conv1`，再经过`conv2`，最后被展平成一维向量。然后，该向量通过全连接层`fc`进行分类。整个模型的输出是对输入图像的类别预测。

```
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3,6,5,padding=2)  
        self.pool = nn.MaxPool2d(2,2)
        self.avpool = nn.AvgPool2d(2,2)# 池化层，采用最大池化结构，模板为2*2，下同
        self.conv2 = nn.Conv2d(6, 12, 5,padding=2)
        self.conv3 = nn.Conv2d(12, 16, 5,padding=2)
        self.fc1 = nn.Linear(16*4*4,120)#,nn.Softmax()
        self.fc2 = nn.Linear(120,84)#,nn.Softmax()
        self.fc3 = nn.Linear(84, 10)
        # nn.Softmax(dim=1)
    def forward (self,x):
        x = F.relu(self.pool((self.conv1(x))))
        #print(x.shape)# 采用relu为激活函数
        x = self.avpool(F.relu(self.conv2(x)))
        #print(x.shape)
        x = self.avpool(F.relu(self.conv3(x)))
        #print(x.shape)
        x = x.view(-1,16*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```









