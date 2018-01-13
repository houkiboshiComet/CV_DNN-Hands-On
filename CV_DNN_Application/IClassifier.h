#pragma once

#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace OpenCVApp {

	class  IClassifier
	{
	public:
		IClassifier();
		virtual ~IClassifier();
		virtual void createNeuralNet() = 0;
		virtual void applyNeuralNet(const cv::Mat* image, cv::Mat* output) = 0;
		static void getMaxClass(const cv::Mat* probBlob, int *classId, double *classProb);
		static std::vector<std::string> readClassNames(const char *filename);

	protected:
		cv::dnn::Net net;
	};
}