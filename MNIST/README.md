# MNIST with CNN
Udemy Pytorch sect 8 

## Create convolutional layers

### Layer 1

Following will create a 2d convolutional layer with:
- 1 input challen (since this dataset work with grey scale)
- 6 output challens for feature extraction  (feature maps) (The filters that the convolutional network will figure out for us)
- A filer by 3x3 
- A step size/strid of 1
There are more features to apply as well eg padding

```Python3
conv1 = nn.Conv2d(1,6,3,1)
```

### Layer 2
This will create second layer with 
- 6 input filters (must be same number as above)
- 16 filers (second filter expand to (can be arbitrary))
- A kernal size of 3
- A Stride of 1



```Python3
conv2 = nn.Conv2d(6,16,3,1)
```