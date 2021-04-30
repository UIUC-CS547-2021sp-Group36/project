import ImageLoader

if __name__ == "__main__":
    print("load data")
    all_train = ImageLoader.load_imagefolder("/workspace/datasets/tiny-imagenet-200/train")
    
    for i,j in enumerate(all_train):
        if i % 1000 == 0:
            print(i)
