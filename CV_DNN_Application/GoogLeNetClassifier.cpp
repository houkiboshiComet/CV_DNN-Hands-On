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
		/* FIX ME */
	}

	/* 2. 入力層に画像をセットする。 */
	void GoogLeNetClassifier::setImage(const cv::Mat* image) {
		/* FIX ME */
	}
	/* 3. 画像分類を行う。( = 出力層まで順伝播させる。) */
	void GoogLeNetClassifier::classify(cv::Mat* probabilities) {
		/* FIX ME */
	}
}