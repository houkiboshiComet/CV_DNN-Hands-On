#pragma once
#include "IClassifier.h"

namespace OpenCVApp {
	class AlexNetClassifier : public IClassifier {
	public:
		AlexNetClassifier();
		~AlexNetClassifier();
		void createNeuralNet();
		void setImage(const cv::Mat* image);
		void classify(cv::Mat* probabilities);
	};
}