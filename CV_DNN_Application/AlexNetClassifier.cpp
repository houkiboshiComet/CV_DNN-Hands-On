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

	/* 1. ニューラルネットワークを形成する。 */
	void AlexNetClassifier::createNeuralNet() {
		static const string prototxt 
			= Paths::ALEXNET_DIR + "\\deploy.prototxt";
		static const string model
			= Paths::ALEXNET_DIR + "\\bvlc_alexnet.caffemodel";
		
		net = cv::dnn::readNetFromCaffe(prototxt, model);
	}
	
	/* 2. 入力層に画像をセットする。 */
	void AlexNetClassifier::setImage(const cv::Mat* image) {
		static const cv::Size cropSize(227, 227);
		static const cv::Scalar averageColor(0, 0, 0);
		cv::Mat blob = cv::dnn::blobFromImage(*image, 1, cropSize, averageColor, false, false);
		net.setInput(blob);
	}

	/* 3. 画像分類を行う。( = 出力層まで順伝播させる。) */
	void AlexNetClassifier::classify(cv::Mat* probabilities) {
		*probabilities = net.forward("prob");
	}

}