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

	/* 1. �j���[�����l�b�g���[�N���`������B */
	void GoogLeNetClassifier::createNeuralNet() {
		/* IMPLEMENTS ME */
	}

	/* 2. ���͑w�ɉ摜���Z�b�g����B */
	void GoogLeNetClassifier::setImage(const cv::Mat* image) {
		/* IMPLEMENTS ME */
	}
	/* 3. �o�͑w�܂ŏ��`�d������B( = �摜���ނ��s���B) */
	void GoogLeNetClassifier::classify(cv::Mat* probabilities) {
		/* IMPLEMENTS ME */
	}
}