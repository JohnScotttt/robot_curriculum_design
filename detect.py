import cv2
import torch
import torchvision.transforms as transforms
import torchvision.models as models

# 加载已经训练好的模型
model_path = 'model.pth'
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 10)
model.load_state_dict(torch.load(model_path))
model.eval()


# 摄像头初始化
cap = cv2.VideoCapture(0)

# 图像预处理函数


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((200, 150)),
        transforms.ToTensor()
    ])
    image = transform(image)
    return image


while True:
    # 读取摄像头图像
    ret, frame = cap.read()

    # 预处理图像
    image = preprocess_image(frame)

    # 图像转换为模型输入的格式
    image = image.unsqueeze(0)  # 添加批处理维度

    # 使用模型进行预测
    outputs = model(image)
    _, predicted_labels = torch.max(outputs, 1)

    # 解析预测结果
    window_switch_label = predicted_labels.item() // 5  # 获取窗户开关标签
    blinds_label = predicted_labels.item() % 5  # 获取百叶窗开合标签

    # 执行相应的控制动作
    if window_switch_label == 0:
        # 执行窗户关闭操作
        # 控制电机关闭窗户
        pass
    elif window_switch_label == 1:
        # 执行窗户打开操作
        # 控制电机打开窗户
        pass

    # 根据百叶窗开合标签控制百叶窗角度
    # 控制电机调整百叶窗开合角度
    angle = blinds_label * 0.25  # 将标签映射到实际角度值
    # 控制电机使百叶窗角度达到设定值

    # 显示预测结果和控制状态
    cv2.putText(frame, f'Window Switch: {window_switch_label}',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f'Blinds Angle: {angle}', (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Window Control', frame)

    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头和关闭窗口
cap.release()
cv2.destroyAllWindows()
