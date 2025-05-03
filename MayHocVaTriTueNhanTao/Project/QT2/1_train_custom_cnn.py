# train_custom_cnn.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
from data_loader import get_cifar100_loaders # Import hàm từ file data_loader.py

class CustomCNN(nn.Module):
    def __init__(self, num_classes=100, input_size=32):
        super(CustomCNN, self).__init__()
        # Thiết kế một CNN đơn giản
        # Block 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) # Batch Norm giúp ổn định huấn luyện
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 32x32 -> 16x16

        # Block 2
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 16x16 -> 8x8

        # Block 3
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 8x8 -> 4x4 (nếu input_size=32)

        # Tính toán kích thước đầu vào cho lớp Fully Connected
        # Kích thước sau 3 lớp pooling: input_size / (2^3)
        # Ví dụ: 32 / 8 = 4. Vậy kích thước là 4x4
        # Số kênh đầu ra của lớp conv cuối cùng là 256
        # Kích thước phẳng = 256 * (input_size // 8) * (input_size // 8)
        fc_input_features = 256 * (input_size // 8) * (input_size // 8)

        # Fully Connected Layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(fc_input_features, 512)
        self.dropout = nn.Dropout(0.5) # Dropout để giảm overfitting
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(F.relu(self.bn2(self.conv2(x))))
        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool2(F.relu(self.bn4(self.conv4(x))))
        # Block 3
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool3(F.relu(self.bn6(self.conv6(x))))

        # Fully Connected
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x) # Output là logits (chưa qua softmax)
        return x

def train_model(model, train_loader, test_loader, num_epochs, learning_rate, device):
    """Hàm huấn luyện và đánh giá model."""
    criterion = nn.CrossEntropyLoss() # Loss function cho multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4) # Adam optimizer với weight decay nhẹ
    # Cân nhắc thêm Learning Rate Scheduler
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    model.to(device) # Chuyển model lên GPU/CPU

    best_accuracy = 0.0 # Lưu lại accuracy tốt nhất trên tập test

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train() # Chuyển sang chế độ huấn luyện
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Thống kê loss và accuracy trong epoch
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc_train = 100.0 * correct_train / total_train

        # Đánh giá trên tập test sau mỗi epoch
        model.eval() # Chuyển sang chế độ đánh giá
        correct_test = 0
        total_test = 0
        with torch.no_grad(): # Không cần tính gradient khi đánh giá
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        epoch_acc_test = 100.0 * correct_test / total_test
        end_time = time.time()
        epoch_duration = end_time - start_time

        print(f'Epoch {epoch+1}/{num_epochs} | '
              f'Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc_train:.2f}% | '
              f'Test Acc: {epoch_acc_test:.2f}% | Duration: {epoch_duration:.2f}s')

        # Cập nhật scheduler (nếu dùng)
        # scheduler.step()

        # Lưu lại model có accuracy tốt nhất
        if epoch_acc_test > best_accuracy:
            best_accuracy = epoch_acc_test
            try:
                torch.save(model.state_dict(), 'custom_cnn_cifar100_best.pth')
                print(f'>>> Best model saved with Test Accuracy: {best_accuracy:.2f}%')
            except Exception as e:
                 print(f"Lỗi khi lưu model: {e}")


    print('Finished Training')
    print(f'Best Test Accuracy: {best_accuracy:.2f}%')

if __name__ == '__main__':
    # --- Hyperparameters ---
    NUM_EPOCHS = 50        # Số lượng epochs (có thể cần nhiều hơn cho CNN từ đầu)
    BATCH_SIZE = 128       # Batch size lớn hơn có thể giúp hội tụ nhanh hơn (tùy bộ nhớ GPU)
    LEARNING_RATE = 0.001  # Learning rate cho Adam
    IMG_SIZE = 32          # Kích thước ảnh đầu vào cho Custom CNN
    USE_AUGMENTATION = True # Nên sử dụng augmentation
    DATA_DIR = './data_cifar100' # Thư mục lưu trữ dataset
    NUM_WORKERS = 4        # Số worker tải data (điều chỉnh tùy hệ thống)

    # --- Thiết bị ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    # --- Tải dữ liệu ---
    print("Đang tải dữ liệu...")
    train_loader, test_loader, num_classes = get_cifar100_loaders(
        batch_size=BATCH_SIZE,
        data_dir=DATA_DIR,
        img_size=IMG_SIZE,
        use_augmentation=USE_AUGMENTATION,
        num_workers=NUM_WORKERS
    )

    if train_loader is None:
        print("Không thể tải dữ liệu. Kết thúc chương trình.")
        exit()

    # --- Khởi tạo model ---
    print("Khởi tạo Custom CNN model...")
    model = CustomCNN(num_classes=num_classes, input_size=IMG_SIZE)
    # print(model) # In kiến trúc model nếu muốn

    # --- Huấn luyện ---
    print("Bắt đầu huấn luyện...")
    train_model(model, train_loader, test_loader, NUM_EPOCHS, LEARNING_RATE, device)