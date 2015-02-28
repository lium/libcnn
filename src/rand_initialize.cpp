#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <ctime>
#include <cstdlib>
#include "rand_initialize.h"
using namespace std;
using namespace Eigen;
using namespace libcnn;
namespace libcnn{

void rand_classifier(MatrixXd& weight_class, MatrixXd& bias_class, const vector<int>& num_kernels, const int& num_labels, const int& num_input_maps, const double& wb, const double& bb)
{
	int num_cols = num_input_maps;
	for(int i = 0; i < num_kernels.size(); i++)
	{
		num_cols = num_cols * num_kernels[i];
	}
//	srand((unsigned)time(NULL));
	weight_class = wb * MatrixXd::Random(num_labels, num_cols);
	bias_class = bb * MatrixXd::Random(num_labels, 1);
}

void rand_conv(vector<vector<MatrixXd> >& weights, vector<MatrixXd>& biases, const vector<int>& num_kernels, const int& kernel_size, const int& num_labels, const double& weight_bound, const double& bias_bound)
{
	weights.clear();
	weights.reserve(num_kernels.size());
	biases.clear();
	biases.reserve(num_kernels.size());
	vector<MatrixXd> temp_weight;
	int prev = 0;
	for(int i = 0; i < num_kernels.size(); i++)
	{
		for(int p = 0; p < i; p++)
		{	
			prev = prev + num_kernels[p];	
		}
		temp_weight.clear();
		temp_weight.reserve(num_kernels[i]);
		srand(i + 1);
		MatrixXd temp_bias = bias_bound * MatrixXd::Random(num_kernels[i], 1);
		biases.push_back(temp_bias);
		for(int j = 0; j < num_kernels[i]; j++)
		{
			srand(prev + j + 1);
			MatrixXd temp = weight_bound * MatrixXd::Random(kernel_size, kernel_size);
			temp_weight.push_back(temp);
		}
		weights.push_back(temp_weight);
	}
}
void print_params(const vector<vector<MatrixXd> >& weights, const vector<MatrixXd>& biases, const MatrixXd& weight_class, const MatrixXd& bias_class)
{
	cout << "Initial Weights, Classifier:\n" << weight_class << endl;
	cout << "Initial Biases, Classifier:\n" << bias_class << endl;
	for(int i = 0; i < weights.size(); i++)
	{
		for(int j = 0; j < weights[i].size(); j++)
		{
			cout << "Initial Weights, Layer " << i << " No. " << j << ":"<< endl;
			cout << weights[i][j] << endl;
		}
		cout << "Initial Biases, Layer " << i << endl;
		cout << biases[i] << endl;
	}
}
}
