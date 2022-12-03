import cv2
import torch
from matplotlib import pyplot as plt
from torchvision import transforms
import torch.nn.functional as F



if __name__ == "__main__":
    #加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load("./LeNet/models/model-minist.pth")
    model = model.to(device)
    model.eval()
    
    img = cv2.imread("./LeNet/images/test5.png")
    img = cv2.resize(img,dsize=(32,32),interpolation=cv2.INTER_NEAREST)
    
    plt.imshow(img,cmap="gray")
    plt.axis("off")
    plt.show()
    
    #导入图片后，扩展为[1,1,32,32]
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))
    ])
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = trans(img)
    img = img.to(device)
    img = img.unsqueeze(0)
    
    #预测
    output = model(img)
    prob = F.softmax(output,dim = 1)
    print("识别概率为：",prob)
    value,predict = torch.max(output.data,1)
    predict = output.argmax(dim = 1)
    print("预测类别是",predict.item())
    
    
 

