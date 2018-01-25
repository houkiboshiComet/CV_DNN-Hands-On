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
		/* IMPLEMENTS ME */
	}
	
	/* 2. 入力層に画像をセットする。 */
	void AlexNetClassifier::setImage(const cv::Mat* image) {
		/* IMPLEMENTS ME */
	}

	/* 3. 出力層まで順伝播させる。( = 画像分類を行う。) */
	void AlexNetClassifier::classify(cv::Mat* probabilities) {
		/* IMPLEMENTS ME */
	}

}