#pragma unmanaged
#include "AlexNetClassifier.h"
#include "Paths.h"
using std::string;

namespace OpenCVApp {
	
	AlexNetClassifier::AlexNetClassifier() {
		/* do nothing */
	}
	AlexNetClassifier::~AlexNetClassifier() {
		/* do nothing */
	}
	
	void AlexNetClassifier::createNeuralNet() {
		static const string prototxt 
			= Paths::ALEXNET_DIR + "\\deploy.prototxt";
		static const string model
			= Paths::ALEXNET_DIR + "\\bvlc_alexnet.caffemodel";
		
		net = cv::dnn::readNetFromCaffe(prototxt, model);
	}
	
	void AlexNetClassifier::setImage(const cv::Mat* image) {
		static const cv::Size cropSize(227, 227);
		static const cv::Scalar averageColor(104, 117, 123);
		cv::Mat blob = cv::dnn::blobFromImage(*image, 1, cropSize, averageColor);
		net.setInput(blob);
	}

	void AlexNetClassifier::classify(cv::Mat* probabilities) {
		*probabilities = net.forward("prob");
	}

}