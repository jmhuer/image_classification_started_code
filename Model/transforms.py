from torchvision import transforms

# Image transformations
image_transforms = {
    # Train uses data augmentation
    'train':
        transforms.Compose([
            transforms.Resize(size=(200, 256)),
            # transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(0.7, 0.7, 0.7, 0.2),
            transforms.RandomHorizontalFlip(),
            # transforms.CenterCrop(size=224),  # Image net standards
            transforms.ToTensor(),
            transforms.RandomErasing(),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])  # Imagenet standards
        ]),
    # Validation does not use augmentation
    'valid':
        transforms.Compose([
            transforms.Resize(size=(200, 256)),
            # transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
        ]),
}

##only tensor transforms
tensor_transform = {
    'train':
        transforms.Compose([
            transforms.Resize(size=(200, 256)),
            transforms.ToTensor(),  # Imagenet standards
        ]),
    'valid':
        transforms.Compose([
            transforms.Resize(size=(200, 256)),
            transforms.ToTensor(),
        ]),
}
