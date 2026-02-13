#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
    // 读取照片
    Mat image = imread("/Users/yanbo/Desktop/xcode_Tipps.jpg");

    // 检测失误
    if (image.empty())
    {
        cout << "Could not open or find the image" << endl;
        cin.get(); // 等待键盘输入
        return -1;
    }

    String windowName = "OpenCV Test"; // 窗口名称
    namedWindow(windowName);           // 创建新窗口
    imshow(windowName, image);         // 使用新窗口显示照片
    waitKey(0);                        // 等待键盘输入
    destroyWindow(windowName);         // 关闭所有窗口
    return 0;
}