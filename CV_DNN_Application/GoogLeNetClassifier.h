#pragma once

#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "IClassifier.h"

namespace OpenCVApp {

	class GoogLeNetClassifier : public IClassifier {
	public:
		GoogLeNetClassifier();
		~GoogLeNetClassifier();
		void createNeuralNet();
		void applyNeuralNet(const cv::Mat* image, cv::Mat* output);
	};

}