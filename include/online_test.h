#ifndef ONLINE_TEST_H
#define ONLINE_TEST_H
#include <iostream>
#include <Eigen/Dense>
#include <vector>
using namespace std;
using namespace Eigen;
namespace libcnn{
void online_test(const vector<vector<MatrixXd> >&, const VectorXi&, const vector<MatrixXd>&, const MatrixXd&, const vector<MatrixXd>&, const MatrixXd&, const vector<MatrixXd>&, const MatrixXd&, const MatrixXd&, const MatrixXd&, const int&, const int&, const int&, const int&, const int&);
}
#endif
