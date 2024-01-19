Project: MNSIT data-classification-by CNN
# Name: Bhavesh Subhash Sandbhor                    Roll No: 22B2446


This report presents the implementation and evaluation of two modified convolutional neural network (CNN) architectures, a modified AlexNet (ConvNet-Alex) and a modified LeNet (ConvNet-Le), for the task of handwritten digit classification using the MNIST dataset. The report outlines the architectures, training processes, and results, and includes a comparative analysis of their effectiveness

The MNIST dataset is a benchmark in the field of machine learning, comprising 28x28 grayscale images of handwritten digits (0-9). Handwritten digit classification is a fundamental problem with applications in optical character recognition and digitizing documents. The objective of this project is to investigate the efficacy of CNNs, particularly AlexNet and LeNet, in accurately classifying these digits.


# Code Implementation:
The provided PyTorch code begins by setting up the necessary parameters such as the number of epochs, batch size, and learning rate.

![](Aspose.Words.16761cbf-c6b9-400c-8629-bc9a5cce7b0e.001.png)
# Dataset Handling:

In the preprocessing phase, MNIST images are transformed into PyTorch tensors and normalized to have a mean of 0.5 and a standard deviation of 0.5. The training dataset is shuffled, and both training and testing datasets are loaded using PyTorch's DataLoader for efficient batch processing. These steps ensure proper formatting and optimization of the MNIST data for training the Convolutional Neural Network.

![](Aspose.Words.16761cbf-c6b9-400c-8629-bc9a5cce7b0e.002.png) 


# AlexNet-inspired Convolutional Neural Network (ConvNet)
Architecture Overview:

The ConvNet implemented in this project is inspired by the architecture of AlexNet, a pioneering convolutional neural network designed for image classification tasks. While it is not an exact replica, it follows a similar structure with convolutional and fully connected layers.
# Convolutional Layer:
#
# Convolutional Layer 1:
Input Channels: 1 (grayscale)

Output Channels: 5

Kernel Size: 5x5

Activation Function: ReLU

Max Pooling: 2x2 with a stride of 1
# Convolutional Layer 2:
Input Channels: 5

Output Channels: 20

Kernel Size: 5x5

Activation Function: ReLU

Max Pooling: 2x2 with a stride of 1
# Convolutional Layer 3:
Input Channels: 20

Output Channels: 20

Kernel Size: 5x5

Activation Function: ReLU

Max Pooling: 2x2 with a stride of 1
# Convolutional Layer 4:
Input Channels: 20

Output Channels: 20

Kernel Size: 5x5

Activation Function: ReLU

Max Pooling: 2x2 with a stride of 1
# Fully Connected Layer 1:
Input Size: 1620 (resulting from flattening the output of the last convolutional layer)

Output Size: 100

Activation Function: ReLU
# Fully Connected Layer 2:
Input Size: 100

Output Size: 100

Activation Function: ReLU
# Fully Connected Layer 3:
Input Size: 100

Output Size: 10 (corresponding to the number of classes in the MNIST dataset)

Activation Function: None (output layer)
# Activation Function:
The Rectified Linear Unit (ReLU) activation function is used after each convolutional and fully connected layer, introducing non-linearity to the model.

Code :

![](Aspose.Words.16761cbf-c6b9-400c-8629-bc9a5cce7b0e.003.png)![](Aspose.Words.16761cbf-c6b9-400c-8629-bc9a5cce7b0e.004.png)

The ConvNet-Alex achieved an accuracy of 97% on the MNIST test set, with a training time of 55-60 seconds per epoch.   
# ` `LeNet-inspired Convolutional Neural Network (CNN)
# Architecture  Overview :
The LeNet-inspired Convolutional Neural Network (CNN) is designed for the classification of handwritten digits using the MNIST dataset. This neural network architecture draws inspiration from the classic LeNet-5 architecture, with modifications tailored to the specific requirements of the task.
## Convolutional Layers:
#
# First Convolutional Layer (Conv1):
Input: Grayscale images (1 channel)

Output: 5 feature maps

Kernel Size: 5x5

Max Pooling Layer (Pool):

Max pooling with a 2x2 window and stride 1

Activation Function (Tanh):

Hyperbolic Tangent (tanh) activation applied after each convolutional layer
# Second Convolutional Layer (Conv2):
Input: 5 feature maps from the previous layer

Output: 5 feature maps

Kernel Size: 5x5

Max pooling and tanh activation similar to the first convolutional layer
# Third Convolutional Layer (Conv3):
Input: 5 feature maps from the second convolutional layer

Output: 5 feature maps

Kernel Size: 5x5

No pooling or activation applied after this layer

Flatten Layer:

Flattens the output from the last convolutional layer into a vector for input to fully connected layers

Fully Connected Layers:
# First Fully Connected Layer (FC1):
Input: Flattened vector with 980 features

Output: 100 features

Tanh activation function applied
# Second Fully Connected Layer (FC2):
Input: 100 features from the previous layer

Output: 10 features (corresponding to the 10 digit classes)
# Activation Function:
The Tanh activation function is used after each convolutional and fully connected layer, introducing non-linearity to the model.

![](Aspose.Words.16761cbf-c6b9-400c-8629-bc9a5cce7b0e.005.png)![](Aspose.Words.16761cbf-c6b9-400c-8629-bc9a5cce7b0e.006.png)Code:

The ConvNet-LeNet achieved an accuracy of 96% on the MNIST test set, with a training time of 35-40 seconds per epoch.   

# Comparison of Architectures:
1\.**Architecture Design**:

•ConvNet-Alex explores deeper architectures with additional convolutional layers, allowing for more intricate feature extraction.

•ConvNet-Le follows a classic design, emphasizing simplicity with fewer convolutional layers.

2\.**Accuracy:**

ConvNet-Alex achieves a slightly higher accuracy of 97%, indicating the effectiveness of deeper architectures for the MNIST task.

•ConvNet-Le achieves a commendable accuracy of 96%, showcasing the reliability of classic designs.

3\.**Computational Efficiency:**

•ConvNet-Alex requires a longer training time of 55-60 seconds per epoch, reflecting the computational cost of deeper architectures.

•ConvNet-Le exhibits better computational efficiency with a training time of 35-40 seconds per epoch, providing a balance between accuracy and speed.
# Discussion:
The comparative analysis highlights the trade-offs between architecture depth and computational efficiency. ConvNet-Alex, with its deeper architecture, achieves marginally higher accuracy but at the expense of increased computational time. ConvNet-Le, while slightly less accurate, demonstrates better computational efficiency, making it a favourable choice for scenarios with resource constraints.
# Conclusion:
Convolutional Neural Networks (CNNs) prove highly advantageous for MNIST digit classification due to their inherent capabilities in localized feature extraction, parameter sharing, and translation invariance. With pooling layers facilitating down-sampling and hierarchical learning, CNNs automatically discern intricate patterns in handwritten digits. Their effectiveness in image classification, robustness to local variations, scalability, and adaptability to model complexity make them a go-to choice. CNNs achieve state-of-the-art performance, and their automatic feature learning eliminates the need for manual engineering. Utilizing CNNs for MNIST data classification optimally combines computational efficiency with exceptional recognition capabilities
# Future Work:
Future work may involve further fine-tuning of both architectures, exploring hybrid designs, and assessing their generalization across diverse datasets



