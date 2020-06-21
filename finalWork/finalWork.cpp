#include <iostream>
#include<opencv2/opencv.hpp>
#include<cmath>

using namespace std;
using namespace cv;


//检测火焰
void detect_fire(cv::Mat&srcMat,cv::Mat&dstMat)
{
	cv::Mat gryMat;
	cv::Mat binMat;
	cv::Mat hsvMat;


	//火焰HSV范围,参考https://www.cnblogs.com/wangyblzu/p/5710715.html
	double i_minH1 = 0;
	double i_maxH1 = 10;
	double i_minH2 = 125;
	double i_maxH2 = 180;
	double i_minS = 43;
	double i_maxS = 255;
	double i_minV = 46;
	double i_maxV = 255;


	//获得灰度图
	cv::cvtColor(srcMat, gryMat, COLOR_BGR2GRAY);
	cv::Mat detectMat = Mat::zeros(gryMat.size(),gryMat.type());
	//转为HSV格式
	cv::cvtColor(srcMat, hsvMat, COLOR_BGR2HSV);
	std::vector<Mat>channels;
	split(hsvMat, channels);
	//分别获得三通道值
	cv::Mat H_Mat;
	cv::Mat S_Mat;
	cv::Mat V_Mat;
	channels.at(0).copyTo(H_Mat);
	channels.at(1).copyTo(S_Mat);
	channels.at(2).copyTo(V_Mat);

	//遍历像素，捕捉火焰
	int rows = srcMat.rows;
	int cols = srcMat.cols;
	for (int i = 200; i < rows; i++)
	{
		for (int j = 400; j < cols; j++)
		{
			if ((((H_Mat.at<uchar>(i, j) >= i_minH1) && (H_Mat.at<uchar>(i, j) <= i_maxH1)) || ((H_Mat.at<uchar>(i, j) >= i_minH2) && (H_Mat.at<uchar>(i, j) <= i_maxH2))) && ((S_Mat.at<uchar>(i, j) >= i_minS) && (S_Mat.at<uchar>(i, j) <= i_maxS)) && ((V_Mat.at<uchar>(i, j) >= i_minV) && (V_Mat.at<uchar>(i, j) <= i_maxV)))
			{
				detectMat.at<uchar>(i, j) = 255;
			}
		}
	}

	//膨胀处理
	cv::Mat kernel = getStructuringElement(MORPH_RECT, Size(15, 15));
	cv::morphologyEx(detectMat, detectMat, MORPH_DILATE, kernel);
	//寻找连通域
	vector<vector<Point>>contours;
	findContours(detectMat, contours, RETR_LIST, CHAIN_APPROX_NONE);
	//绘制轮廓
	for (int i = 0; i < contours.size(); i++)
	{
		RotatedRect rbox = minAreaRect(contours[i]);
		drawContours(srcMat, contours, i, Scalar(0, 255, 255), 1, 8);
		Point2f points[4];
		rbox.points(points);
		for (int j = 0; j < 4; j++)
		{
			cv::line(dstMat, points[j], points[j < 3 ? j + 1 : 0], Scalar(255, 255, 255), 2, LINE_AA);
		}
	}


}


//计算拟合曲线多项式的系数矩阵,参考https ://blog.csdn.net/guduruyu/java/article/details/72866144 
bool polynomial_curve_fit(std::vector<cv::Point>& key_point, int n, cv::Mat& A)
{
	//Number of key points
	int N = key_point.size();

	//构造矩阵X
	cv::Mat X = cv::Mat::zeros(n + 1, n + 1, CV_64FC1);
	for (int i = 0; i < n + 1; i++)
	{
		for (int j = 0; j < n + 1; j++)
		{
			for (int k = 0; k < N; k++)
			{
				X.at<double>(i, j) = X.at<double>(i, j) +
					std::pow(key_point[k].x, i + j);
			}
		}
	}

	//构造矩阵Y
	cv::Mat Y = cv::Mat::zeros(n + 1, 1, CV_64FC1);
	for (int i = 0; i < n + 1; i++)
	{
		for (int k = 0; k < N; k++)
		{
			Y.at<double>(i, 0) = Y.at<double>(i, 0) +
				std::pow(key_point[k].x, i) * key_point[k].y;
		}
	}

	A = cv::Mat::zeros(n + 1, 1, CV_64FC1);
	//求解矩阵A
	cv::solve(X, Y, A, cv::DECOMP_LU);
	return true;
}


//寻找水柱上的点
void find_points(cv::Mat& srcMat, std::vector<cv::Point>& key_points)
{
	for (int i = 150; i < 460; i++)
	{
		for (int j = 35; j < 240; j++)
		{
			if (srcMat.at<uchar>(j, i) == 255)
			{
				key_points.push_back(cv::Point(i, j));
				break;
			}
		}
	}
}

//主函数
int main()
{
	//读取视频，并检测读取成功性
	VideoCapture cap("D:\\Files\\DIP\\Final.mp4");
	if (!cap.isOpened())
	{
		std::cout << "Fail to open Video!" << endl;
		return -1;
	}

	//定义参数及变量
	int cnt = 0;
	cv::Mat frame;
	cv::Mat bgMat;
	cv::Mat subMat;
	cv::Mat bin_subMat;
	cv::Mat gryMat;
	cv::Mat binMat;
	cv::Mat rgbMat1;
	cv::Mat rgbMat2;

	//进入循环
	while (0)
	{
		cap >> frame;

		//保存彩色图
		rgbMat1 = frame.clone();
		rgbMat2 = frame.clone();


		//转为灰度图
		cv::cvtColor(frame, frame, COLOR_BGR2GRAY);
		if (cnt == 0)
		{
			//第一帧，获取背景图像
			frame.copyTo(bgMat);
			//检测火焰
			detect_fire(rgbMat1, rgbMat1);
			cv::imshow("dst", rgbMat1);
			waitKey(30);
		}
		else if (cnt <= 100)//还未出现水柱
		{
			detect_fire(rgbMat1, rgbMat1);
			cv::imshow("dst", rgbMat1);
			waitKey(30);
		}
		else//出现水柱
		{
			//开始背景差分
			//当前图像与背景图像相减
			absdiff(frame, bgMat, subMat);
			//二值化
			threshold(subMat, binMat, 100, 255, THRESH_BINARY);
			//寻找水柱上的点
			std::vector<cv::Point>points;
			find_points(binMat, points);


			//拟合曲线，参考https://blog.csdn.net/guduruyu/article/details/72866144
			cv::Mat coefficients;
			polynomial_curve_fit(points, 3, coefficients);
			//计算拟合点
			std::vector<cv::Point>points_fitted;
			for (int x = 150; x < 460; x++)
			{
				double y = coefficients.at<double>(0, 0) + coefficients.at<double>(1, 0)*x + coefficients.at<double>(2, 0)*std::pow(x, 2) + coefficients.at<double>(3, 0)*std::pow(x, 3);
				points_fitted.push_back(cv::Point(x, y));
			}
			//绘制拟合曲线
			cv::polylines(rgbMat1, points_fitted, false, cv::Scalar(0, 255, 255), 1, 8, 0);
			//检测火焰
			detect_fire(rgbMat2, rgbMat1);
			//显示最终结果，描绘出水柱的轨迹与检测火焰
			cv::imshow("dst", rgbMat1);
			waitKey(30);
		}
		cnt++;
	}
	waitKey(0);
	return 0;
}

