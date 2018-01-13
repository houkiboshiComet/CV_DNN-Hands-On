#pragma once
#include "IClassifier.h"

namespace OpenCVApp {
	class AlexNetClassifier : public IClassifier {
	public:
		AlexNetClassifier();
		~AlexNetClassifier();
		void createNeuralNet();
		void applyNeuralNet(const cv::Mat* image, cv::Mat* output);
	};
}