#pragma unmanaged
#include "GoogLeNetClassifier.h"
#include "Paths.h"
using std::string;

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
		/* IMPLEMENT ME */
	}

	/* 2. ���͑w�ɉ摜���Z�b�g����B */
	void GoogLeNetClassifier::setImage(const cv::Mat* image) {
		/* IMPLEMENT ME */
	}
	/* 3. �o�͑w�܂ŏ��`�d������B( = �摜���ނ��s���B) */
	void GoogLeNetClassifier::classify(cv::Mat* probabilities) {
		/* IMPLEMENT ME */
	}
}