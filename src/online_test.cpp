#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include "online_test.h"
#include "conv_layer.h"
#include "pool.h"
#include "softmax_class.h"
using namespace std;
using namespace Eigen;
using namespace libcnn;
namespace libcnn{
void online_test(const vector<vector<MatrixXd> >& vec_data, const VectorXi& vec_label, const vector<MatrixXd>& weight_0, const MatrixXd& bias_0, const vector<MatrixXd>& weight_1, const MatrixXd& bias_1, const vector<MatrixXd>& weight_2, const MatrixXd& bias_2, const MatrixXd& weight_class, const MatrixXd& bias_class, const int& num_kerns1, const int& num_kerns2, const int& num_kerns3, const int& kern_size, const int& pool_size)
{
	vector<vector<MatrixXd> > test_data;
	copy(vec_data.begin(), vec_data.begin() + 1, back_inserter(test_data));
	VectorXi test_label;
	test_label = vec_label.segment(0, 1200);
	ConvPoolLayer *p_conv0 = new ConvPoolLayer(test_data, num_kerns1, kern_size, "same", "tanh", weight_0, bias_0);
	Pool *p_pool0 = new Pool(p_conv0->batch_maps_activated, pool_size, pool_size);
	ConvPoolLayer *p_conv1 = new ConvPoolLayer(p_pool0->output_batch_pooled, num_kerns2, kern_size, "same", "tanh", weight_1, bias_1);
	Pool *p_pool1 = new Pool(p_conv1->batch_maps_activated, pool_size, pool_size);
	ConvPoolLayer *p_conv2 = new ConvPoolLayer(p_pool1->output_batch_pooled, num_kerns3, kern_size, "same", "tanh", weight_2, bias_2);
	Pool *p_pool2 = new Pool(p_conv2->batch_maps_activated, pool_size, pool_size);
	MatrixXd feature_vectors;
	get_feature_vector(p_pool2->output_batch_pooled, feature_vectors);
	
//	classifier.input = feature_vectors;
	Softmax classifier(feature_vectors, 9, test_label, 1, 1, weight_class, bias_class);
	classifier.calculation_output();
	cout << classifier.m.transpose() << endl;	
	int accuracy = 0;
	for(int i = 0; i < test_label.size(); i++)
	{
		if(classifier.m(i) == test_label(i))
		{
			accuracy ++;
		}
	}
	cout << "accuracy : " << accuracy << endl;
}

}
