#ifndef RAND_INITIALIZA_H
#define RAND_INITIALIZA_H
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <ctime>
using namespace std;
using namespace Eigen;
namespace libcnn{

void rand_classifier(MatrixXd&, MatrixXd&, const vector<int>&, const int&, const int&, const double&, const double&);
void rand_conv(vector<vector<MatrixXd> >&, vector<MatrixXd>&, const vector<int>&, const int&, const int&, const double&, const double&);
void print_params(const vector<vector<MatrixXd> >&, const vector<MatrixXd>&, const MatrixXd&, const MatrixXd&);
}
#endif
