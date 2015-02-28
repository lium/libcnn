#include <iostream>
#include <Eigen/Dense>
#include <string>
#include <vector>
#include "save_load.h"
#include "read_write.h"
using namespace std;
using namespace Eigen;
using namespace libcnn;

namespace libcnn{

void save(const vector<vector<MatrixXd> >& weights, const vector<MatrixXd>& biases, const MatrixXd& class_weight, const MatrixXd& class_bias)
{
	typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> Matrix_MxN;
	assert(weights.size() == biases.size());
	for(int i = 0; i < weights.size(); i++)
	{
		for(int j = 0; j < weights[i].size(); j++)
		{
			string filename_w = "weight" + to_string(i) + to_string(j) + ".dat";
			const char* w_ptr = filename_w.c_str();
			write_binary(w_ptr, weights[i][j]);
		}
		string filename_b = "bias" + to_string(i) + ".dat";
		const char* b_ptr = filename_b.c_str();
		write_binary(b_ptr, biases[i]);
	}
	string filename_cw = "classifier_weight.dat"; const char* cw_ptr = filename_cw.c_str();
	string filename_cb = "classifier_bias.dat"; const char* cb_ptr = filename_cb.c_str();
	write_binary(cw_ptr, class_weight);
	write_binary(cb_ptr, class_bias);
}

void load(const vector<int>& num_kerns, vector<vector<MatrixXd> >& weights, vector<MatrixXd>& biases, MatrixXd& class_weight, MatrixXd& class_bias)
{
	weights.clear();
	weights.reserve(3);
	biases.clear();
	biases.reserve(3);
	
	/* load convolutional kernels and biases*/
	for(int i = 0; i < num_kerns.size(); i++)
	{
		vector<MatrixXd> vec_weight;
		vec_weight.reserve(num_kerns[i]);
		for(int j = 0; j < num_kerns[i]; j++)
		{
			string filename_w = "weight" + to_string(i) + to_string(j) + ".dat";
			const char* w_ptr = filename_w.c_str();
			MatrixXd weight_temp;
			read_binary(w_ptr, weight_temp);
			vec_weight.push_back(weight_temp);
		}
		string filename_b = "bias" + to_string(i) + ".dat";
		const char* b_ptr = filename_b.c_str();
		MatrixXd bias_temp;
		read_binary(b_ptr, bias_temp);
		biases.push_back(bias_temp);
		weights.push_back(vec_weight);
	}

	/* load classifier weights and biases*/
	string filename_cw = "classifier_weight.dat"; const char* cw_ptr = filename_cw.c_str();
	string filename_cb = "classifier_bias.dat"; const char* cb_ptr = filename_cb.c_str();
	read_binary(cw_ptr, class_weight);
	read_binary(cb_ptr, class_bias);
}

}
