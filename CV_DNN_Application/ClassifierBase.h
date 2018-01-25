#pragma once

#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace OpenCVApp {

	class ClassifierBase
	{
	public:
		ClassifierBase();
		virtual ~ClassifierBase();
		
		virtual void createNeuralNet() = 0;
		virtual void setImage(const cv::Mat* image) = 0;
		virtual void classify(cv::Mat* probabilities) = 0;
		
		void outLayerAsImage(const std::string& layer, const std::string& nameHead, const std::string& extension, int number = 0);
		void outLayerAsCsv(const std::string& layer, const std::string& nameHead, int number = 0);

		static void getMaxClass(const cv::Mat* probabilities, int *topClass, double *topProb);
		static void getRankHigherClasses(const cv::Mat* probabilities, std::vector<int> *higherClass, std::vector<float> *higherProbs, int rankCount);
		static std::vector<std::string> readClassNames(const std::string& filename);

	protected:
		cv::dnn::Net net;
		static const int IMAGENET_MEAN_R = 123; 
		static const int IMAGENET_MEAN_G = 117; 
		static const int IMAGENET_MEAN_B = 104;

	};
}