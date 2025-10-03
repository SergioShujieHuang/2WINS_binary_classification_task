First, I need to check the dataset by myself.

It’s like a toyota logo, the background is all black.
According to the document, there are 1000 pictures of good and 350 pictures of bad. it's unbalance.

I think when there are many black, corrupted part or scratched part, it will be labeled as the bad class.

There is no need for data cleaning

PNG size 1024*1024

I try to do data augumentation since the product is a cirle-shaped item.
Also, because there is no significant influence for the RGB image, for process speed, it's better using the gray scale image.

for data augumentation, I am consider to rotate the image.

First, for the RGB image and gray scale image, I am consider to do the experiment.

validation/test set will not be augumented.
explainable AI, Grad-CAM / saliency

initialize anaconda

# RGB and grayscale image experiment with data augumentation
baseline A: ResNet50, RGB, rotation
baseline B: ResNet50, grayscale, roration

resize to 224 * 224

the result shows that 
the RGB image will be influence by the background sometime(red part)

![RGB image](./RGB_grayscale_experiment_with%20data%20augumentation/RBG_influenced_by_bg.png "RBG image")

![RGB image](./RGB_grayscale_experiment_with%20data%20augumentation/RGB.png "RBG image")

at the same time, the gray scale image concentrates on the "bad" part

![Grayscale image](./RGB_grayscale_experiment_with%20data%20augumentation/grayscale_bg.png "Grayscale image")

Also there is no significant difference in confusion_matrix and ROC

So based on this experiment, I choose to use grayscale image as input.

because the current model resnet50 is enough SOTA. I think I need to test the speed on GPU and CPU.

# experiment on the prediction speed of GPU and CPU
### Hardware Info
CPU: Intel64 Family 6 Model 165 Stepping 2, GenuineIntel

CPU Cores (logical): 12

CPU Frequency: 2592.00 MHz
RAM: 15.84 GB

GPU 0: NVIDIA GeForce RTX 2070 with Max-Q Design
  - Total Memory: 8.00 GB
  - CUDA Capability: 7.5



### Inference Speed
CUDA avg inference time: 0.002124 s/image

CPU avg inference time: 0.042809 s/image

![Inference Speed image](./GPU_CPU_speed_experiment/inference_speed_comparison_gray.png "Inference Speed image")

so the Inference Speed is fast enough

# visualize log

I need to add a visualize log when it's the real factory environment.

![Log image](./visualize_log/05_IMG_2E420008_2025-02-24_14-56-14_000197_GlossRatio_cropped.png_gradcam.png "Log image")

according to my experience, the real factory will use GPU or Edge AI.

# to try more lite model

for resnet50, it need 360MB or more, so we need to try more lite model when we consider edge AI.

after the experiment, I decide to use MobileNetV3-Large model.

# Change to Docker


# after feedback meeting (Thks for MR.Teshima !!!)
- 1. set the priority of evaluation metrics
False Negative Rate / Miss Rate >> False Positive Rate >> Inference Latency >> Resource Cost (model size, deployment cost)
- 2. Change to Docker
- 3. finish the ppt

# Change evaluation metrics
 the result shows that the FNR and FPR is very large, means it's not good.
 so I am consider to change to Focal Loss, and do more date Augumentation.

 (1) 损失函数

把 Focal Loss 改成 加权 CrossEntropyLoss，直接拉高坏品的权重。

use_focal_loss = False
use_class_weight_loss = True


这样：

w_good = 1.0
w_bad = float(good_count / (bad_count + 1e-8)) * 2.0  # 再提高一点坏品权重
class_weights = torch.tensor([w_good, w_bad], dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

(2) Early Stopping 改用 FNR

把早停条件从 val_loss 换成 FNR，让模型更偏向降低漏检。

(3) 调整数据增强

坏品样本不要过度旋转、裁剪，否则可能把缺陷信息裁掉。
建议坏品图像只做轻微增强，比如：

transforms.RandomHorizontalFlip(),
transforms.RandomVerticalFlip(),

(4) 阈值调节

你现在是 argmax(outputs)，其实可以用 概率阈值调节。
比如：

preds = (probs > 0.3).long()   # 把阈值从 0.5 降低到 0.3

这样能降低 FNR（但 FPR 可能稍微升高）。

Using class weights for loss: good=1.0, bad=5.738

考虑使用更好的模型

手动拆分过程中导致可能训练集中没有任何一个bad，改成分层拆分版本

