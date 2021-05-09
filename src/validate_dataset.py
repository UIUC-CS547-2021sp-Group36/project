import data.ImageLoader as ImageLoader

if __name__ == "__main__":
    
    import torchvision.datasets
    #print("Atempt to load offending image")
    #torchvision.datasets.folder.default_loader("/workspace/datasets/tiny-imagenet-200/train/n03980874/images/n03980874_480.JPEG")

    print("load data")
    all_train = ImageLoader.load_imagefolder("/workspace/datasets/tiny-imagenet-200/")
    
    for i,j in enumerate(all_train):
        if i % 1000 == 0:
            print(i)
