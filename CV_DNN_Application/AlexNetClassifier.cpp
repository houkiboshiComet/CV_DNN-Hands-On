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
		static const string prototxt 
			= Paths::ALEXNET_DIR + "\\deploy.prototxt";
		static const string model
			= Paths::ALEXNET_DIR + "\\bvlc_alexnet.caffemodel";
		
		net = cv::dnn::readNetFromCaffe(prototxt, model);
	}
	
	/* 2. ���͑w�ɉ摜���Z�b�g����B */
	void AlexNetClassifier::setImage(const cv::Mat* image) {
		static const cv::Size cropSize(227, 227);
		static const cv::Scalar averageColor( IMAGENET_MEAN_B, IMAGENET_MEAN_G, IMAGENET_MEAN_R);
		cv::Mat blob = cv::dnn::blobFromImage(*image, 1, cropSize, averageColor, false, true);
		net.setInput(blob, "data");
	}

	/* 3. �o�͑w�܂ŏ��`�d������B( = �摜���ނ��s���B) */
	void AlexNetClassifier::classify(cv::Mat* probabilities) {
		cv::Mat layersOutput = net.forward("prob");
		/* �|�C���^�̎w���s��̒l���A���`�d�ɂ���ĕς�邽��Deep Copy */
		*probabilities = layersOutput.clone();
	}

}