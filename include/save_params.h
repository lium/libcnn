#ifndef SAVE_PARAMS_H
#define SAVE_PARAMS_H
#include <iostream>
#include <Eigen/Dense>
#include <vector>
using namespace std;
using namespace Eigen;
namespace libcnn{
void print_params(vector<vector<MatrixXd> >&, vector<MatrixXd>&, vector<MatrixXd>&, vector<MatrixXd>&, vector<MatrixXd>&, MatrixXd&, MatrixXd&, MatrixXd&, MatrixXd&, MatrixXd&);
}
#endif
