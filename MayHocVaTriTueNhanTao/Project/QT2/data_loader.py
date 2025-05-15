# data_loader.py
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

'''
* Args:
    - batch_size (int): Kích thước batch.
    - data_dir (str): Thư mục để lưu/tải dữ liệu CIFAR-100.
    - img_size (int): Kích thước ảnh đầu ra sau khi resize. 
        + Mặc định là 32 cho CIFAR-100 gốc.
        + Thay đổi thành 224 (hoặc kích thước khác) nếu dùng model pre-trained.
* Returns:
    - (train_loader, test_loader, num_classes)
        + train_loader: DataLoader cho tập huấn luyện.
        + test_loader: DataLoader cho tập kiểm tra.
        + num_classes: Số lượng lớp (100 cho CIFAR-100).
'''
def get_cifar100_loaders(batch_size, data_dir='./data', img_size=32, use_augmentation=True, num_workers=4):
    num_classes = 100
    # Giá trị Mean và Std chuẩn cho CIFAR-100
    cifar100_mean = (0.5071, 0.4867, 0.4408)
    cifar100_std = (0.2675, 0.2565, 0.2761)

    # Transform cho tập Test/Validation (chỉ resize, ToTensor, Normalize)
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)), # Resize ảnh về kích thước mong muốn
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std)
    ])

    # Transform cho tập Train
    if use_augmentation:
        train_transform = transforms.Compose([
            # Augmentation nên được thực hiện TRƯỚC Normalize và ToTensor
            # Một số kỹ thuật augmentation phổ biến cho CIFAR:
            transforms.RandomCrop(32, padding=4), # Crop ngẫu nhiên sau khi padding
            transforms.RandomHorizontalFlip(),    # Lật ngang ngẫu nhiên
            transforms.RandomRotation(15),      # Xoay ngẫu nhiên
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), # Thay đổi màu sắc
            # AutoAugment cung cấp chính sách augmentation tốt đã được tìm kiếm tự động
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10), # Dùng policy cho CIFAR10 cũng khá hiệu quả
            transforms.Resize((img_size, img_size)), # Resize ảnh về kích thước mong muốn (sau augmentation)
            transforms.ToTensor(),
            transforms.Normalize(cifar100_mean, cifar100_std),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False) # Cutout/Random Erasing
        ])
    else:
        # Nếu không dùng augmentation, transform giống tập test
        train_transform = test_transform

    # --- Tải Dataset ---
    try:
        train_dataset = torchvision.datasets.CIFAR100(
            root=data_dir,
            train=True,
            download=True,
            transform=train_transform
        )

        test_dataset = torchvision.datasets.CIFAR100(
            root=data_dir,
            train=False,
            download=True,
            transform=test_transform
        )
    except Exception as e:
        print(f"Lỗi khi tải CIFAR-100: {e}")
        return None, None, num_classes

    # --- Tạo DataLoader ---
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Xáo trộn dữ liệu huấn luyện
        num_workers=num_workers,
        pin_memory=True # Tăng tốc độ chuyển dữ liệu sang GPU (nếu có)
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False, # Không cần xáo trộn dữ liệu kiểm tra
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Đã tải xong CIFAR-100.")
    print(f"Số lượng ảnh Train: {len(train_dataset)}")
    print(f"Số lượng ảnh Test: {len(test_dataset)}")
    print(f"Kích thước ảnh: {img_size}x{img_size}")
    print(f"Sử dụng Data Augmentation: {use_augmentation}")

    return train_loader, test_loader, num_classes

if __name__ == '__main__':
    BATCH_SIZE = 64
    IMG_SIZE = 32 # Giữ nguyên 32x32 cho Custom CNN ban đầu

    print("Kiểm tra data loader với augmentation:")
    train_loader_aug, test_loader_aug, num_classes_aug = get_cifar100_loaders(
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        use_augmentation=True
    )

    if train_loader_aug:
        dataiter = iter(train_loader_aug)
        images, labels = next(dataiter)
        print("Kích thước batch huấn luyện (có augmentation):", images.shape) # Output: torch.Size([64, 3, 32, 32])
        print("Kiểu dữ liệu ảnh:", images.dtype)
        print("Giá trị min/max của ảnh (sau normalize):", images.min(), images.max())
        print("Số lớp:", num_classes_aug) # Output: 100

    print("\nKiểm tra data loader không có augmentation:")
    train_loader_noaug, test_loader_noaug, num_classes_noaug = get_cifar100_loaders(
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        use_augmentation=False
    )
    
    if test_loader_noaug:
        dataiter = iter(test_loader_noaug)
        images, labels = next(dataiter)
        print("Kích thước batch kiểm tra (không augmentation):", images.shape)