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

	/* 1. �j���[�����l�b�g���[�N���`������B */
	void AlexNetClassifier::createNeuralNet() {
		/* IMPLEMENTS ME */
	}
	
	/* 2. ���͑w�ɉ摜���Z�b�g����B */
	void AlexNetClassifier::setImage(const cv::Mat* image) {
		/* IMPLEMENTS ME */
	}

	/* 3. �o�͑w�܂ŏ��`�d������B( = �摜���ނ��s���B) */
	void AlexNetClassifier::classify(cv::Mat* probabilities) {
		/* IMPLEMENTS ME */
	}

}