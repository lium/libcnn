#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include "save_params.h"
using namespace std;
using namespace Eigen;
using namespace libcnn;
namespace libcnn{
void print_params(vector<vector<MatrixXd> >& weights, vector<MatrixXd>& biases, vector<MatrixXd>& weight_0, vector<MatrixXd>& weight_1, vector<MatrixXd>& weight_2, MatrixXd& bias_0, MatrixXd& bias_1, MatrixXd& bias_2, MatrixXd weight_class, MatrixXd& bias_class)
{
	cout << "Final Weights, Classifier:\n" << weight_class << endl;
	cout << "Final Biases, Classifier:\n" << bias_class << endl;
	weights.clear();
	biases.clear();
	weights.push_back(weight_0);
	weights.push_back(weight_1);
	weights.push_back(weight_2);
	biases.push_back(bias_0);
	biases.push_back(bias_1);
	biases.push_back(bias_2);
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
