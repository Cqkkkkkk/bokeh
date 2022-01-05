# Bokeh Effect Rendering

## 下载预训练模型
如果想要测试其他的（非本项目提供的，位于`data/input`之外的）图片，需要额外下载用于深度预测的模型[LeResModel](https://disk.pku.edu.cn:443/link/DBA221915F86D88B0F262708F4F2D4D0)以及用于调整深度信息的模型[U2NetModel](https://disk.pku.edu.cn:443/link/91757D41C92EBFD1A7042E8682E73CC7)。
将LeResModel放入`LeRes/ckpts/`下，将U2NetModel放入`U2Net/ckpts/`即可。

否则，可以直接跳过本节。

## 基本使用
运行GUI程序：
```bash
python3 main.py
```
#### 输入图片

通过菜单`File-Open`按钮，可以打开一个系统对话框，用于选择输入的原始图片。项目提供的一些例子图片在`data/input/`中。

注意，在`data/input/`中的图片都已经预先生成了相应的深度图片，它们位于`data/depth/`中。如果需要自己测试其他的图片，务必将首先图片放到`data/input/`文件夹下，并将相应的深度图片放入`data/depth/`文件下。如果程序没有检测到对应的深度文件，将由模型自动生成，但是这会比较耗费时间。例子：原始图片`data/input/1002.jpg`，深度图片`data/depth/1002-depth.png`。此外，可以直接通过`LeRes/run.sh`来批量生成深度图像：

```bash
./LeRes/run.sh
```
#### 切换Blur模式
通过菜单`Mode`栏，可以在三种不同的算法中切换，其中`Simple`模式对应简单模糊算法，`Depth`模式对应Depth-aware的模糊算法，而`DNN`模式则代表基于深度神经网络的模糊算法。

#### 切换核函数

基于当前模式，算法适用的核函数也是可以调节的。通过菜单`Kernel`栏，在每个不同的模式下，有一些不同的核函数可供选择，包括`Simple`核、`Gaussian`核，`Radial`核以及`Tanh`核等。注意这个菜单只会显示适合当前模式的核函数，在`DNN`模式下将不会显示任何核函数。

#### 进行转换

点击控制区域的`Convert`按钮，即可进行对图片的模糊处理。处理后的结果将自动显示在GUI右上角。

#### 保存结果

通过菜单`File-Save`按钮，可以将模糊后的图片存储到任意位置。注意图片的名字须以一定的格式结尾，如`.png`。

## 参数调节

基于当前模式，调整当前算法的参数：在对应的模式被激活时，GUI界面右下角的控制区域会显示一定的可调节参数。

#### `Simple` 模式

- `KernelSize`: 用于控制对应核函数的大小
- `Threshold`: 用于控制前后景分割的界限，深度低于界限的像素将被分类为前景，反之则为后景。


#### `Depth-Aware` 模式

- $a$：控制深度分层的区间，模型将利用深度信息将原图分为若干层，层与层之间的深度的差异界限为$\frac 1a$；此外，$a$还将影响基于深度信息，以及$d_f$计算出的当前层的Kernel Size。一般来说，$a$越接近1，算法的分层越细腻，效果越好，但是模型的运算速度就会越低。
- $d_f$：用于控制聚焦位置，模型将聚焦深度为$d_f$的对应区域。

## 其他

#### `Depth-Aware` 模式下参数$d_f$的手动调整

在`Depth-Aware` 模式下，Depth和Original窗口都可以通过点击来设定聚焦深度$d_f$，通过点击原图，或者深度图片的对应位置，程序会将聚焦深度$d_f$设定为对应的深度，这时候再次点击`Convert`可以以对应的深度进行Blur。

#### 使用Modified Depth Map

在`Simple`以及`Depth-Aware` 模式下，可以使用基于Disparity图像修正后的Depth图像来进一步优化算法，只需要选中GUI右下角的`DepthModify`按钮，即可激活该功能。

注意，使用该功能时需保证对应的Mask图像位于`data/mask/`之下，程序将自动检测有无对应的Mask图像。本程序预先提供的所有输入图片都有对应的Mask。如果需要自己增加新图片作为输入，程序会自动调用U2Net生成相应的图像，这个操作比较耗费时间。

此外，也可以直接用`U2Net/`下的`generate_all.sh`来批量生成对应的图片的Mask：

```bash
./U2Net/generate_all.sh
```
一个例子如下所示:（如果无法加载，下载[链接](https://pimags.oss-cn-beijing.aliyuncs.com/uPic/df-range.gif)）
![image](https://pimags.oss-cn-beijing.aliyuncs.com/uPic/df-range.gif)