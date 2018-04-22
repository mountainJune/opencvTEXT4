#include<iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<opencv2\calib3d\calib3d.hpp>
#include<math.h>

using namespace std;
using namespace cv;
//访问像素的3种方法
void firstWay(Mat I, uchar table[])
{

	//获得图像某个位置的rgb值
	int channels = I.channels();
	int nRows = I.rows;
	int nCols = I.cols * channels;
	if (I.isContinuous())
	{
		nCols *= nRows;
		nRows = 1;
	}
	int i, j;
	uchar* p;
	for (i = 0; i < nRows; ++i)
	{
		p = I.ptr<uchar>(i);
		for (j = 0; j < nCols; ++j)
		{
			p[j] = table[p[j]];
		}
	}
}
void secondWay(Mat I, uchar table[])
{

	const int channels = I.channels();
	switch (channels)
	{
	case 1:
	{
		MatIterator_<uchar> it, end;
		for (it = I.begin<uchar>(), end = I.end<uchar>(); it != end; ++it)
			*it = table[*it];
		break;
	}
	case 3:
	{
		MatIterator_<Vec3b> it, end;
		for (it = I.begin<Vec3b>(), end = I.end<Vec3b>(); it != end; ++it)
		{
			(*it)[0] = table[(*it)[0]];
			(*it)[1] = table[(*it)[1]];
			(*it)[2] = table[(*it)[2]];
		}
	}
	}
}
void thirdWay(Mat I, uchar table[])
{

	const int channels = I.channels();
	switch (channels)
	{
	case 1:
	{
		for (int i = 0; i < I.rows; ++i)
			for (int j = 0; j < I.cols; ++j)
				I.at<uchar>(i, j) = table[I.at<uchar>(i, j)];
		break;
	}
	case 3:
	{
		Mat_<Vec3b> _I = I;
		for (int i = 0; i < I.rows; ++i)
			for (int j = 0; j < I.cols; ++j)
			{
				_I(i, j)[0] = table[_I(i, j)[0]];
				_I(i, j)[1] = table[_I(i, j)[1]];
				_I(i, j)[2] = table[_I(i, j)[2]];
			}
		I = _I;
	}
	}
}

void show(Mat image,string name)//输入图像
{
	const int channels = image.channels();

	switch (channels)
	{
	case 1:
	{
		int nimages = 1;//待处理的图像张数
		int channels = 0;//待处理的通道序列
		Mat outputHist;
		int dims = 1;//直方图维度
		int histSize = 256;//直方图每个维度的大小
		float hranges[2] = { 0, 255 };
		const float *ranges[1] = { hranges };//直方图每个维度的范围
		bool uni = true;//是否均匀计算
		bool accum = false;//是否累加计算
						   //计算图像的直方图
		calcHist(&image, nimages, &channels, Mat(), outputHist, dims, &histSize, ranges, true, false);
		//找到最大值和最小值
		double maxValue = 0;
		minMaxLoc(outputHist, NULL, &maxValue);
		int scale = 1;
		Mat histPic(histSize * scale, histSize, CV_8UC1, Scalar(0));
		//纵坐标缩放比例
		double rate = (histSize * scale / maxValue) * 0.9;
		

		for (int i = 0; i < histSize; i++)
		{
			//得到每个i和箱子的值
			float value = outputHist.at<float>(i);
			
			line(histPic, cv::Point(i, histSize * scale), cv::Point(i, histSize * scale - value*rate), Scalar(255));
		}
		namedWindow(name, WINDOW_NORMAL);

		imshow(name, histPic);
	}
	case 3:
	{
		//图片数量nimages
		int nimages = 1;
		//通道数量,我们总是习惯用数组来表示
		int channels[3] = { 0,1,2 };
		//输出直方图
		Mat outputHist_red, outputHist_green, outputHist_blue;
		//维数
		int dims = 1;
		//存放每个维度直方图尺寸（bin数量）的数组histSize
		int histSize[3] = { 256,256,256 };
		//每一维数值的取值范围ranges
		float hranges[2] = { 0, 255 };
		//值范围的指针
		const float *ranges[3] = { hranges,hranges,hranges };
		//是否均匀
		bool uni = true;
		//是否累积
		bool accum = false;
		//计算图像的直方图(红色通道部分)
		calcHist(&image, nimages, &channels[0], cv::Mat(), outputHist_red, dims, &histSize[0], &ranges[0], uni, accum);
		//计算图像的直方图(绿色通道部分)
		calcHist(&image, nimages, &channels[1], cv::Mat(), outputHist_green, dims, &histSize[1], &ranges[1], uni, accum);
		//计算图像的直方图(蓝色通道部分)
		calcHist(&image, nimages, &channels[2], cv::Mat(), outputHist_blue, dims, &histSize[2], &ranges[2], uni, accum);
		//画出直方图
		int scale = 2;
		//直方图的图片,因为尺寸是一样大的,所以就以histSize[0]来表示全部了.
		Mat histPic(histSize[0], histSize[0] * scale * 3, CV_8UC3, cv::Scalar(0, 0, 0));
		//找到最大值和最小值,索引从0到2分别是红,绿,蓝
		double maxValue[3] = { 0, 0, 0 };
		double minValue[3] = { 0, 0, 0 };
		minMaxLoc(outputHist_red, &minValue[0], &maxValue[0], NULL, NULL);
		minMaxLoc(outputHist_green, &minValue[1], &maxValue[1], NULL, NULL);
		minMaxLoc(outputHist_blue, &minValue[2], &maxValue[2], NULL, NULL);
		//纵坐标缩放比例
		double rate_red = (histSize[0] / maxValue[0])*0.9;
		double rate_green = (histSize[0] / maxValue[1])*0.9;
		double rate_blue = (histSize[0] / maxValue[2])*0.9;
		for (int i = 0; i < histSize[0]; i++)
		{
			float value_red = outputHist_red.at<float>(i);
			float value_green = outputHist_green.at<float>(i);
			float value_blue = outputHist_blue.at<float>(i);
			//分别画出直线
			//line(histPic, cv::Point(i*scale, histSize[0]), cv::Point(i*scale, histSize[0] - value_red*rate_red), cv::Scalar(0, 0, 255));
			//line(histPic, cv::Point((i + 256)*scale, histSize[0]), cv::Point((i + 256)*scale, histSize[0] - value_green*rate_green), cv::Scalar(0, 255, 0));
			//line(histPic, cv::Point((i + 512)*scale, histSize[0]), cv::Point((i + 512)*scale, histSize[0] - value_blue*rate_blue), cv::Scalar(255, 0, 0));
			//分别画出矩形
			rectangle(histPic, Point(i*scale, histSize[0]), Point(i*scale + 2, histSize[0] - value_red*rate_red), Scalar(0, 0, 255));
			rectangle(histPic, Point((i + 256)*scale, histSize[0]), Point((i + 256)*scale + 2, histSize[0] - value_green*rate_green), Scalar(0, 255, 0));
			rectangle(histPic, Point((i + 512)*scale, histSize[0]), Point((i + 512)*scale + 2, histSize[0] - value_blue*rate_blue), Scalar(255, 0, 0));

		}
		imshow(name, histPic);
		
	}
	}

}
void bianhuan1(Mat I,float value)
{

}

int main()
{
	const string filename = "input2.jpg";
	Mat srcImg = imread(filename, CV_LOAD_IMAGE_COLOR);
	if (srcImg.empty())
		return -1;
	/*灰度图像的直方图均衡化
	Mat dstImg(srcImg.size(), CV_8UC1);
	cvtColor(srcImg, dstImg, CV_RGB2GRAY);//图像灰度化
	imshow("dstImg",dstImg);
	Mat dst2Img;
	equalizeHist(dstImg, dst2Img);
	//dstImg = equalizeChannelHist(srcImg);
	imshow("处理前", dstImg);
	show(dstImg);
	imshow("处理后", dst2Img);
	show(dst2Img);
	waitKey(0);
	*/
	imshow("原图", srcImg);
	show(srcImg,"原图的直方图");
	Mat mergeImg;
	vector<Mat> splitBGR(srcImg.channels());//创建splitBGR对象存储分割通道
	split(srcImg, splitBGR);
	for (int i = 0; i < srcImg.channels(); i++) {
		equalizeHist(splitBGR[i], splitBGR[i]);
		merge(splitBGR, mergeImg); //融合通道
	}
	imshow("均衡化的图", mergeImg);
	show(mergeImg,"均衡化的直方图");
	waitKey(0);
}
