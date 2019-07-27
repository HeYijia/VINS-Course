# Vins Course
**作者**：贺一家，高翔，崔华坤，赵松
**描述**：
这是一个用于深蓝学院教学的代码，她基于 VINS-Mono 框架，但不依赖 ROS, Ceres, G2o. 虽然这是个基础版本, 她应该能帮助你学习后端优化 LM 算法流程，滑动窗口先验如何维护，鲁棒核函数如何书写等等 SLAM 优化中的常见问题。

### Licence
The source code is released under GPLv3 license.

We are still working on improving the code reliability. For any technical issues, please contact Yijia He <	heyijia2016@gmail.com> , Xiang Gao <<https://github.com/gaoxiang12>> or Huakun Cui<<https://github.com/StevenCui>>.

For commercial inquiries, please contact Song Zhao <?>

### 安装依赖项：

2. pangolin: <https://github.com/stevenlovegrove/Pangolin>

   用于 GUI 显示

2. opencv

3. Eigen

### 编译代码

```c++
mkdir build 
cd build
cmake ..
make -j4
../bin/run_euroc /home/dataset/EuRoC/MH-05/mav0/ ../config/
```

### 感谢

我们使用了港科大沈老师组的 [VINS-Mono](https://github.com/HKUST-Aerial-Robotics/VINS-Mono) 作为基础代码，非常感谢该组的工作。

