from PIL import Image
from torchvision import transforms
img_path='./myimg/test1.jpg'
img = Image.open(img_path)
img= img.convert('RGB')

trans_1=transforms.Resize(40)
trans_2=transforms.RandomResizedCrop(32, scale=(0.64, 1),ratio=(1.0, 1.0))
trans_3=transforms.ToTensor()


img_1=trans_1(img)
img_2=trans_2(img_1)

img_1.show()
img_2.show()