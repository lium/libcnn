#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <sstream>
#include "init_params.h"
#define KERNEL_SIZE 7

using namespace std;
using namespace Eigen;
using namespace libcnn;

namespace libcnn{

void get_parameter(vector<MatrixXd>& feature_weights)
{
	feature_weights.clear();
	ifstream infile;
	infile.open("../matlab/kernels.txt");
	while(infile.peek() != EOF)
	{
		MatrixXd mat(KERNEL_SIZE,KERNEL_SIZE);
		string line;
		getline(infile, line);
		stringstream stream(line);
		for(int i = 0; i < KERNEL_SIZE; i++)
		{
			for(int j = 0; j < KERNEL_SIZE; j++)
			{
				stream >> mat(i, j);	
			}	
		}
		feature_weights.push_back(mat);
	}	
}

}
