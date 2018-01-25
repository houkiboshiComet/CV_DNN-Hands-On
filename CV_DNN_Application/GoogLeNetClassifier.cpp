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
		static const string prototxt
			= Paths::GOOGLENET_DIR + "\\deploy.prototxt";
		static const string model
			= Paths::GOOGLENET_DIR + "\\bvlc_googlenet.caffemodel";

		net = cv::dnn::readNetFromCaffe(prototxt, model);
	}

	/* 2. ���͑w�ɉ摜���Z�b�g����B */
	void GoogLeNetClassifier::setImage(const cv::Mat* image) {
		static const cv::Size cropSize(224, 224);
		static const cv::Scalar averageColor(IMAGENET_MEAN_B, IMAGENET_MEAN_G, IMAGENET_MEAN_R);
		cv::Mat blob = cv::dnn::blobFromImage(*image, 1, cropSize, averageColor, false, true);
		net.setInput(blob, "data");
	}
	/* 3. �o�͑w�܂ŏ��`�d������B( = �摜���ނ��s���B) */
	void GoogLeNetClassifier::classify(cv::Mat* probabilities) {
		cv::Mat layersOutput = net.forward("prob");
		/* �|�C���^�̎w���s��̒l���A���`�d�ɂ���ĕς�邽��Deep Copy */
		*probabilities = layersOutput.clone();
	}
}