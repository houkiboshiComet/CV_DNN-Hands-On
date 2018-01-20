#pragma once
#include "ClassifierBase.h"

namespace OpenCVApp {
	class AlexNetClassifier : public ClassifierBase {
	public:
		AlexNetClassifier();
		~AlexNetClassifier();
		void createNeuralNet();
		void setImage(const cv::Mat* image);
		void classify(cv::Mat* probabilities);
	};
}