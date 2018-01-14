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
		virtual void setImage(const cv::Mat* image) = 0;
		virtual void classify(cv::Mat* probabilities) = 0;
		
		void outLayerAsImage(const std::string& layer, const std::string& nameHead, const std::string& extension, int number = 0);
		
		static void getMaxClass(const cv::Mat* probabilities, int *classId, double *classProb);
		static std::vector<std::string> readClassNames(const std::string& filename);

	protected:
		cv::dnn::Net net;
	};
}