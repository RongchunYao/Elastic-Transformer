



normalize = torchvision.transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
test_dataset = torchvision.datasets.ImageFolder(file_dir+'/ILSVRC2012',
                torchvision.transforms.Compose([
                    torchvision.transforms.Resize(int(img_size*8/7)),
                    torchvision.transforms.CenterCrop(img_size),
                    torchvision.transforms.ToTensor(),
                    normalize,
                ]))        
        
small_dataset = Small_Dataset(test_dataset,-1,100,1234)
data_loader = torch.utils.data.DataLoader(small_dataset, batch_size=1, shuffle=False, num_workers=1)