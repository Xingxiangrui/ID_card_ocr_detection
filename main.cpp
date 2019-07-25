#include<iostream>
#include<fstream>
#include<string>
#include<math.h>
#include<algorithm>
#include<opencv.hpp>
#include"highgui/highgui.hpp"
#include<features2d.hpp>
#include"imgproc/imgproc.hpp"
#include<Windows.h>

#include<allheaders.h>
#include<capi.h>
#include <locale.h>

#define PI 3.141593

using namespace cv;
using namespace std;

bool cmp2(Point2f a, Point2f b){
	if (a.x != b.x) return a.x < b.x;
}

bool cmp(RotatedRect a, RotatedRect b){

	Point2f vertices1[4], vertices2[4];
	a.points(vertices1);
	sort(vertices1, vertices1 + 3, cmp2);
	b.points(vertices2);
	sort(vertices2, vertices2 + 3, cmp2);
	if ((vertices1[2].x - vertices1[0].x) != (vertices2[2].x - vertices2[0].x))
		return (vertices1[2].x - vertices1[0].x) > (vertices2[2].x - vertices2[0].x);
}

Mat ConnectChar(Mat img_gray){
	//文本区域（ＭＳＥＲ算法）
	Mat img_gray1 = 255 - img_gray;
	vector<vector<Point>>regContours;
	
	MSER(6,25,150)(img_gray1, regContours);

	Mat mserMapMat = Mat::zeros(img_gray1.size(), CV_8UC1);
	for (int i = (int)regContours.size() - 1; i >= 0; i--)
	{
		Rect bboxes1 = boundingRect(regContours[i]);
		Mat roi(mserMapMat, Rect(bboxes1.x, bboxes1.y, bboxes1.width, bboxes1.height));
		roi = 255;
		
	}
	Mat img_bw;
	morphologyEx(mserMapMat, img_bw, MORPH_CLOSE, Mat::ones(1, 20, CV_8UC1));
	return img_bw;
}
void die(const char *errstr){
	fputs(errstr, stderr);
	system("pause");
	exit(1);
}

int main(){
	
	// 图片缩放
	IplImage* pImg;
    pImg = cvLoadImage("picture/s6.jpg");
	float ratio = pImg->width / 1.0 / pImg->height;
	long square = 400000;
	int  Pixel_Y = sqrt(square / ratio);
	int  Pixel_X = square / Pixel_Y;
	Mat img(pImg, 0);
	Mat img_gray;
	cvtColor(img, img_gray, CV_RGB2GRAY);
	resize(img_gray, img_gray, { Pixel_X, Pixel_Y }, 0, 0, INTER_AREA);
	imshow("mserclose", img_gray);
	waitKey();

	
	Mat img_bw = ConnectChar(img_gray);//MSER文本区域检测,闭操作
	imshow("mserclose", img_bw);
	waitKey();
	// 查找轮廓，对应连通域 
	vector<vector<Point>> contours, contours1;
	vector<Vec4i> hierarchy;
	findContours(img_bw, contours, RETR_LIST, CHAIN_APPROX_NONE);

	if (!contours.empty())
	{
		vector<vector<Point>>::iterator it;
		vector<RotatedRect> Bw_info;
		int i = 0;

		for (it = contours.begin(); it != contours.end(); it++)
		{
			RotatedRect item = minAreaRect(*it);//查找连通域的最小外接矩形
			Bw_info.push_back(item);
		}

		sort(Bw_info.begin(), Bw_info.end(), cmp);//对连通域长度排序
	
		RotatedRect IdRect = Bw_info[0];

		float angle1 = 0.0, width = 0.0, height = 0.0;
		if (IdRect.size.width > IdRect.size.height){
			angle1 = IdRect.angle;
			width =floor( IdRect.size.width*1.05);
			height =floor( IdRect.size.height*1.3);
		}

		else{
			angle1 = 90.0 + IdRect.angle;
			height =floor( IdRect.size.width*1.3);
			width = floor(IdRect.size.height*1.05);
		}

		int midx, midy;
		Mat img_c,img_c2;
		if (fabs(angle1) > 1){
			float angle = fabs(angle1*PI / 180);
			float angle0 = -angle1*PI / 180;
			int  w = floor(Pixel_Y*sin(angle) + Pixel_X*cos(angle));
			int  h = floor(Pixel_Y*cos(angle) + Pixel_X*sin(angle));
		    midx =floor( (IdRect.center.x - Pixel_X / 2)*cos(angle0) + ( Pixel_Y/2-IdRect.center.y)*sin(angle0) + w/2);
		    midy =floor((IdRect.center.x - Pixel_X / 2)*sin(angle0) - ( Pixel_Y/2-IdRect.center.y )*cos(angle0) + h/2);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
			Point2f center(Pixel_X / 2, Pixel_Y / 2);
			Mat rotMat = getRotationMatrix2D(center, angle1, 1);  //旋转映射矩阵
			rotMat.at<double>(0, 2) += (w - Pixel_X) / 2;     // 修改坐标偏移
			rotMat.at<double>(1, 2) += (h - Pixel_Y) / 2;   // 修改坐标偏移
			Mat img_r(h, w, CV_8UC1);
			warpAffine(img_gray, img_r, rotMat, img_r.size()); //仿射变换
			imshow("Image rotate", img_r);
			waitKey();
			Rect rect(midx - width / 2, midy - height / 2, width, height);
			img_r(rect).copyTo(img_c);
			Rect rect2(floor(midx - width / 4.5*3.5), floor(midy - width / 4.5 * 4), floor(width / 4.5*3.5), floor(width / 4.5 * 3.6));
			img_r(rect2).copyTo(img_c2);

       }
		else{
			
			midx =floor( IdRect.center.x);
			midy = floor(IdRect.center.y);
			Rect rect(midx - width / 2, midy - height / 2, width, height);
			img_gray(rect).copyTo(img_c);
			Rect rect2(floor(midx - width / 4.5*3.5), floor(midy - width / 4.5 * 4), floor(width / 4.5*3.5), floor(width / 4.5 * 3.6));
			img_gray(rect2).copyTo(img_c2);
        }

		square = 18000;
		ratio = width / 1.0 / height;
		int Py = sqrt(square / ratio);
		int Px = square / Py;
		resize(img_c, img_c, { Px, Py });
		imshow("Identity cut", img_c);
		waitKey();
		imshow("Identity cut", img_c2);
		waitKey();
		Mat img_cb,img_cb2;
		int blocksize = floor(Py/4)*2+1;
		CvScalar Pix_mean;
		Pix_mean = mean(img_c);
		adaptiveThreshold(img_c, img_cb, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, blocksize, 20);//局部二值化
		adaptiveThreshold(img_c2, img_cb2, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, blocksize, 50);
		imwrite("cut.jpg", img_cb);
		imwrite("cut2.jpg", img_cb2);
		imshow("BW image", img_cb);
		waitKey();
		imshow("BW image", img_cb2);
		waitKey();

		TessBaseAPI *handle,*handle2;
		PIX *img1,*img2;
		char *text,*text2;
		if ((img1 = pixRead("cut.jpg")) == NULL)
			die("Error reading image\n");
		handle  = TessBaseAPICreate();
		if (TessBaseAPIInit3(handle, NULL, "eng") != 0)//英文库识别身份证号码
			die("Error initialising tesseract\n");
		TessBaseAPISetImage2(handle, img1);
		TessBaseAPISetVariable(handle, "tessedit_char_whitelist", "0123456789Xx");//设置白名单
		TessBaseAPISetVariable(handle, "tessedit_char_blacklist", "ABCDEFGHIJKLMNPQRSTUVWYZabcdefghijklmnopqrstuvwyz");//设置黑名单
		if (TessBaseAPIRecognize(handle, NULL) != 0)
			die("Error in Tesseract recognition\n");
		if ((text  = TessBaseAPIGetUTF8Text(handle)) == NULL)// tesseract 识别
			die("Error getting text\n");
		fputs(text, stdout);
		ofstream f1("me1.txt");
		f1 <<text << endl;
		f1.close();
		TessDeleteText(text);
		TessBaseAPIEnd(handle);
		TessBaseAPIDelete(handle);
		pixDestroy(&img1);

		if ((img2 = pixRead("cut2.jpg")) == NULL)
			die("Error reading image\n");
		handle2 = TessBaseAPICreate();
		if (TessBaseAPIInit3(handle2, NULL, "chi_sim") != 0)//调用中文简体库识别汉字
			die("Error initialising tesseract\n");
		TessBaseAPISetImage2(handle2, img2);
    	if (TessBaseAPIRecognize(handle2, NULL) != 0)
			die("Error in Tesseract recognition\n");
		if ((text2 = TessBaseAPIGetUTF8Text(handle2)) == NULL)
			die("Error getting text\n");
		ofstream f2("me2.txt");
		f2 << text2 << endl;
		f2.close();
		TessDeleteText(text2);
		TessBaseAPIEnd(handle2);
		TessBaseAPIDelete(handle2);
		pixDestroy(&img2);

}
	system("pause");
   return 0;
}
