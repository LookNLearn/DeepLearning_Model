import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
from PIL import ImageFile
from facenet_pytorch import MTCNN  # MTCNN 라이브러리 추가

# 경고 무시 설정
warnings.filterwarnings("ignore", message="Corrupt EXIF data")
warnings.filterwarnings("ignore", message="Truncated File Read")
ImageFile.LOAD_TRUNCATED_IMAGES = True

print(torch.version.cuda)
print(torch.backends.cudnn.version())


# CUDA 디버그용 환경 설정
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
# 체크포인트 경로 설정
checkpoint_path = './checkpoints/best_checkpoint_mtcnn_kdef.pth'

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 이미지와 예측 결과 시각화를 위한 함수
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # 잠시 대기 후 이미지 표시

# MTCNN으로 얼굴을 감지하는 함수
mtcnn = MTCNN(keep_all=True, device=device)

def extract_faces(inputs):
    faces = []
    for img in inputs:
        face, _ = mtcnn.detect(img)
        if face is not None:
            faces.append(face)
    return faces

# 얼굴 감지 후, 모델에 전달하여 예측하는 함수
def visualize_model(model, dataloaders, class_names, num_images=6):
    model.eval()  # 평가 모드로 전환
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # MTCNN을 사용해 얼굴 추출
            faces = extract_faces(inputs)

            if len(faces) == 0:
                continue

            outputs = model(faces)
            _, preds = torch.max(outputs, 1)

            # 이미지 표시
            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'Pred: {class_names[preds[j]]} (Actual: {class_names[labels[j]]})')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    return

# 학습 및 검증 결과 시각화를 위한 함수 (정확도 및 손실)
def plot_training_results(train_acc, val_acc, train_loss, val_loss):
    epochs = range(1, len(train_acc) + 1)

    plt.figure(figsize=(12, 5))

    # 정확도 그래프
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # 손실 그래프
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

def main():
    # config 파일 불러오기
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    # config에서 불러온 값 사용
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    learning_rate = config['learning_rate']
    momentum = config['momentum']
    train_dir = config['train_dir']
    val_dir = config['val_dir']
    image_size = tuple(config['image_size'])
    dropout_rate = config['dropout_rate']

    # TensorBoard SummaryWriter 설정 (기록할 디렉토리 지정)
    writer = SummaryWriter(log_dir="runs")

    # 이미지 전처리 작업 (데이터 증강 추가)
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(image_size),   # 랜덤 크기 조정 및 자르기
            transforms.RandomHorizontalFlip(),          # 좌우 뒤집기
            transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.2, hue=0.1),  # 밝기, 대비, 채도 및 색조 변환
            transforms.Grayscale(num_output_channels=3), 
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'val': transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
    }

    # 데이터셋 불러오기
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, data_transforms['train']),
        'val': datasets.ImageFolder(val_dir, data_transforms['val'])
    }

    # DataLoader 설정
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=4),
        'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=4)
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    # EfficientNet-B0 모델 불러오기
    model_ft = models.efficientnet_b0(pretrained=True)

    # EfficientNet의 마지막 fully connected layer 수정 (7개의 표정 분류에 맞게)
    num_ftrs = model_ft.classifier[1].in_features  # EfficientNet의 마지막 레이어 입력 특징 수
    model_ft.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate),  # 드롭아웃
        nn.Linear(num_ftrs, 7)  # 7개의 표정 분류를 위한 출력층으로 변경
    )

    model_ft = model_ft.to(device)

    # 손실 함수 및 옵티마이저 설정
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=learning_rate)

    # 체크포인트 불러오기 (기존 학습된 모델이 있는 경우)
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # strict=False 옵션을 사용하여 모델 가중치만 로드
        model_ft.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        # 옵티마이저의 상태는 불러오지 않고 새로운 옵티마이저를 사용
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        print(f"Checkpoint loaded. Starting from epoch {start_epoch} with best accuracy {best_acc:.4f}")
    else:
        print("No checkpoint found. Starting training from scratch.")
        start_epoch = 0
        best_acc = 0.0
    

    # 학습 함수 정의
    def train_model(model, criterion, optimizer, num_epochs=25, start_epoch=0, best_acc=0.0):
        best_model_wts = model.state_dict()

        train_acc_history = []
        val_acc_history = []
        train_loss_history = []
        val_loss_history = []

        for epoch in range(start_epoch, num_epochs):
            print(f'Epoch {epoch + 1}/{num_epochs}')
            print('-' * 10)

            # 각 epoch마다 훈련 및 검증 진행
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # 모델을 훈련 모드로 설정
                else:
                    model.eval()   # 모델을 평가 모드로 설정

                running_loss = 0.0
                running_corrects = 0

                # tqdm을 사용한 진행바 추가
                with tqdm(dataloaders[phase], unit="batch") as tepoch:
                    for inputs, labels in tepoch:
                        tepoch.set_description(f"{phase.capitalize()} Epoch {epoch + 1}")

                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        # 파라미터 gradients를 0으로 설정
                        optimizer.zero_grad()

                        # forward
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)

                            # backward + optimize (only in training phase)
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()

                        # 통계 계산
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)

                        # tqdm에 진행 상황 업데이트
                        tepoch.set_postfix(loss=loss.item())

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # TensorBoard에 기록
                if phase == 'train':
                    train_acc_history.append(epoch_acc.item())
                    train_loss_history.append(epoch_loss)
                    writer.add_scalar('Loss/train', epoch_loss, epoch)
                    writer.add_scalar('Accuracy/train', epoch_acc, epoch)
                else:
                    val_acc_history.append(epoch_acc.item())
                    val_loss_history.append(epoch_loss)
                    writer.add_scalar('Loss/val', epoch_loss, epoch)
                    writer.add_scalar('Accuracy/val', epoch_acc, epoch)

                # best model 저장
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict()
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_acc': best_acc,
                    }, checkpoint_path)  # best checkpoint 저장

            print()

        print(f'Best val Acc: {best_acc:4f}')

        # best model weights 로드
        model.load_state_dict(best_model_wts)

        # 학습 및 검증 결과 시각화
        plot_training_results(train_acc_history, val_acc_history, train_loss_history, val_loss_history)

        return model

    # 학습 실행
    model_ft = train_model(model_ft, criterion, optimizer_ft, num_epochs=num_epochs)

    # 시각화 함수 사용 (이미지 예측 결과 시각화)
    visualize_model(model_ft, dataloaders, class_names)

    # TensorBoard 종료
    writer.close()

if __name__ == '__main__':
    main()
