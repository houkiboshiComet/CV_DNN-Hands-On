#pragma unmanaged
#include "GoogLeNetClassifier.h"


namespace OpenCVApp {
	
	GoogLeNetClassifier::GoogLeNetClassifier()
	{
		/* do nothing */
	}
	GoogLeNetClassifier::~GoogLeNetClassifier()
	{
		/* do nothing */
	}

	/* 1. ニューラルネットワークを形成する。 */
	void GoogLeNetClassifier::createNeuralNet() {
		/* IMPLEMENTS ME */
	}

	/* 2. 入力層に画像をセットする。 */
	void GoogLeNetClassifier::setImage(const cv::Mat* image) {
		/* IMPLEMENTS ME */
	}
	/* 3. 出力層まで順伝播させる。( = 画像分類を行う。) */
	void GoogLeNetClassifier::classify(cv::Mat* probabilities) {
		/* IMPLEMENTS ME */
	}
}