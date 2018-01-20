#pragma once

#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "ClassifierBase.h"

namespace OpenCVApp {

	class GoogLeNetClassifier : public ClassifierBase {

	public:
		GoogLeNetClassifier();
		~GoogLeNetClassifier();
		void createNeuralNet();
		void setImage(const cv::Mat* image);
		void classify(cv::Mat* probabilities);
	};

}