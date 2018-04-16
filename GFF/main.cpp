// C++ code for S. Li, X. Kang, and J. Hu, “Image fusion with guided filtering,” IEEE
// Trans.Image Process., vol. 22, no. 7, pp. 2864–2875, Jul. 2013.
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/ximgproc.hpp"
#include <vector>
#include <iostream>

void rgb2gray(const std::vector<cv::Mat>& vecI, std::vector<cv::Mat>& vecI_gray)
{
	vecI_gray.reserve(vecI.size());

	for (std::vector<cv::Mat>::const_iterator it = vecI.begin(); it != vecI.end(); ++it)
	{
		cv::Mat gray;
		cv::cvtColor(*it, gray, cv::COLOR_BGR2GRAY);
		vecI_gray.push_back(gray);
	}
}

void laplacianFilter(const std::vector<cv::Mat>& vecI_gray, std::vector<cv::Mat>& vecH)
{
	vecH.reserve(vecI_gray.size());

	cv::Mat L = (cv::Mat_<float>(3, 3) << 0, 1, 0, 1, -4, 1, 0, 1, 0);
	for (std::vector<cv::Mat>::const_iterator it = vecI_gray.begin(); it != vecI_gray.end(); ++it)
	{
		cv::Mat l;
		cv::filter2D(*it, l, -1, L);
		vecH.push_back(l);
	}
}

void gaussianSaliency(const std::vector<cv::Mat>& vecH, std::vector<cv::Mat>& vecS)
{
	vecS.reserve(vecH.size());

	for (std::vector<cv::Mat>::const_iterator it = vecH.begin(); it != vecH.end(); ++it)
	{
		cv::Mat absH = cv::abs(*it);
		cv::Mat s;
		cv::GaussianBlur(absH, s, cv::Size(11, 11), 5.0);
		vecS.push_back(s);
	}
}

void initWightMaps(const std::vector<cv::Mat>& vecS, std::vector<cv::Mat>& vecP)
{
	vecP.reserve(vecS.size());

	for (size_t i = 0; i < vecS.size(); ++i)
	{
		vecP.push_back(cv::Mat::zeros(vecS.front().size(), CV_32FC1));
	}

	const int w = vecS.front().cols;
	const int h = vecS.front().rows;
	for (int y = 0; y < h; ++y)
	{
		for (int x = 0; x < w; ++x)
		{
			int n = 0;
			float max = 0.0f;
			for (size_t i = 0; i < vecS.size(); ++i)
			{
				if (max < vecS[i].at<float>(y, x))
				{
					max = vecS[i].at<float>(y, x);
					n = (int)i;
				}
			}
			vecP[n].at<float>(y, x) = 1.0f;
		}
	}
}

void guidedOptimize(const std::vector<cv::Mat>& vecP, const std::vector<cv::Mat>& vecI_gray,
	std::vector<cv::Mat>& vecWB, std::vector<cv::Mat>& vecWD)
{
	vecWB.reserve(vecP.size());
	vecWD.reserve(vecP.size());

	for (size_t i = 0; i < vecP.size(); ++i)
	{
		cv::Mat wb;
		cv::ximgproc::guidedFilter(vecI_gray[i], vecP[i], wb, 45, 0.3 * 255 * 255); // 因为导引图未归一化，所以参数需要乘以255*255
		vecWB.push_back(wb);

		cv::Mat wd;
		cv::ximgproc::guidedFilter(vecI_gray[i], vecP[i], wd, 7, 1e-6 * 255 * 255);
		vecWD.push_back(wd);
	}

	const int w = vecP.begin()->cols;
	const int h = vecP.begin()->rows;
	for (int y = 0; y < h; ++y)
	{
		for (int x = 0; x < w; ++x)
		{
			float sumB = 0.0f;
			float sumD = 0.0f;
			for (size_t i = 0; i < vecWB.size(); ++i)
			{
				float fB = vecWB[i].at<float>(y, x);
				if (fB > 1.0f) // 过虑大于1.0的值
				{
					vecWB[i].at<float>(y, x) = 1.0f;
					fB = 1.0f;
				}
				sumB += fB;

				float fD = vecWD[i].at<float>(y, x);
				if (fD > 1.0f) // 过虑大于1.0的值
				{
					vecWD[i].at<float>(y, x) = 1.0f;
					fD = 1.0f;
				}
				sumD += fD;
			}
			for (size_t i = 0; i < vecWB.size(); ++i)
			{
				vecWB[i].at<float>(y, x) /= sumB;
				vecWD[i].at<float>(y, x) /= sumD;
			}
		}
	}
}

void decompose(const std::vector<cv::Mat>& vecI, std::vector<cv::Mat>& vecB, std::vector<cv::Mat>& vecD)
{
	vecB.reserve(vecI.size());
	vecD.reserve(vecI.size());

	for (std::vector<cv::Mat>::const_iterator it = vecI.begin(); it != vecI.end(); ++it)
	{
		cv::Mat b;
		cv::boxFilter(*it, b, -1, cv::Size(31, 31));
		cv::Mat d = *it - b;
		vecB.push_back(b);
		vecD.push_back(d);
	}
}

void fuse(const std::vector<cv::Mat>& vecI, std::vector<cv::Mat>& vecWB, std::vector<cv::Mat>& vecWD,
	cv::Mat& F)
{
	F = cv::Mat::zeros(vecI.begin()->size(), vecI.begin()->type());

	std::vector<cv::Mat> vecB;
	std::vector<cv::Mat> vecD;
	decompose(vecI, vecB, vecD);

	if (vecI.begin()->channels() == 3)
	{
		std::vector<cv::Mat> vec;
		for (size_t i = 0; i < vecWB.size(); ++i)
		{
			vec.clear();
			vec.push_back(vecWB[i]);
			vec.push_back(vecWB[i]);
			vec.push_back(vecWB[i]);
			cv::merge(vec, vecWB[i]);

			vec.clear();
			vec.push_back(vecWD[i]);
			vec.push_back(vecWD[i]);
			vec.push_back(vecWD[i]);
			cv::merge(vec, vecWD[i]);
		}
	}

	for (size_t i = 0; i < vecI.size(); ++i)
	{
		cv::Mat temp1 = vecWB[i].mul(vecB[i]);
		cv::Mat temp2 = vecWD[i].mul(vecD[i]);
		F = F + temp1 + temp2;
	}
}

int main()
{
	std::vector<cv::Mat> vecI;
 	vecI.push_back(cv::imread("../data/colourset/garage1.jpg"));
 	vecI.push_back(cv::imread("../data/colourset/garage2.jpg"));
 	vecI.push_back(cv::imread("../data/colourset/garage3.jpg"));
 	vecI.push_back(cv::imread("../data/colourset/garage4.jpg"));
 	vecI.push_back(cv::imread("../data/colourset/garage5.jpg"));
 	vecI.push_back(cv::imread("../data/colourset/garage6.jpg"));
	for (std::vector<cv::Mat>::iterator it = vecI.begin(); it != vecI.end(); ++it)
	{
		if (it != vecI.begin())
		{
			if (it->size() != vecI.begin()->size())
			{
				std::cout << "All images should have the same size." << std::endl;
				return -1;
			}
		}
		(*it).convertTo(*it, CV_32FC3);
	}

	std::vector<cv::Mat> vecI_gray;
	rgb2gray(vecI, vecI_gray);

	std::vector<cv::Mat> vecH;
	laplacianFilter(vecI_gray, vecH);

	std::vector<cv::Mat> vecS;
	gaussianSaliency(vecH, vecS);

	std::vector<cv::Mat> vecP;
	initWightMaps(vecS, vecP);

	std::vector<cv::Mat> vecWB;
	std::vector<cv::Mat> vecWD;
	guidedOptimize(vecP, vecI_gray, vecWB, vecWD);

	cv::Mat F;
	fuse(vecI, vecWB, vecWD, F);
	F.convertTo(F, CV_8U);

	cv::imshow("F", F);

	cv::waitKey();
	return 0;
}
