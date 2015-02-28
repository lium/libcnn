#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include <string>
#include "conv_layer.h"
#include "softmax_class.h"
#include "pool.h"
#include "load_data.h"

using namespace std;
using namespace Eigen;
using namespace libcnn;

void evaluate()
{
	string file_names[4];
	file_names[0] = "../parameter/parameter_conv0.txt";
	file_names[1] = "../parameter/parameter_conv1.txt";
	file_names[2] = "../parameter/parameter_conv2.txt";
	file_names[3] = "../parameter/parameter_classifier.txt";

	ifstream files[4];
	files[0].open(file_names[0]);
	files[1].open(file_names[1]);
	files[2].open(file_names[2]);
	files[3].open(file_names[3]);
	
	vector<vector<MatrixXd> > weights;
	vector<MatrixXd> biases;	
	
	for(int i = 0; i < 3; i++)
	{
		vector<MatrixXd> weight_conv;
		while(files[i].peek() != EOF)
		{
			MatrixXd weight_temp(7,7);
			string line;
			getline(files[i], line);
			stringstream stream(line);
			for(int j = 0; j < 7; j++)
			{
				for(int k = 0; k < 7; k++)
				{
					stream >> weight_temp(j, k);
				}
			}
			weight_conv.push_back(weight_temp);	
		}
		weights.push_back(weight_conv);
	}
	
	for(int i = 0; i < 4; i++)
	{
		files[i].close();
	}

	for(int i = 0; i < 3; i++)
	{
		for(int j = 0; j < weights[i].size(); j++)
		{
			cout << "***********" << endl;
			cout << weights[i][j] << endl;
		}
	}
	
}

int main()
{
	evaluate();
	return 0;
}
