#ifndef LOAD_DATA_H
#define LOAD_DATA_H
 
#include <cstring>
#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include <fstream>


using namespace Eigen;
using namespace std;

void getdir(const string & dir, vector<string> & files);

void read_image_files(const string & dir, std::vector<vector<MatrixXd> > & vvm, const int & image_row_dim, const int & image_col_dim, const int & start, const int & image_number);

void read_label_files(const string & dir, VectorXi & m, const int & image_row_dim, const int & image_col_dim, const int & start, const int & image_number, const string & data_mode);

void read_markov_files(const string & dir, MatrixXd & markov);

#endif
