/*
 * =====================================================================================
 *
 *       Filename:  yaml_test.cpp
 *
 *    Description:  yaml_test
 *
 *        Version:  1.0
 *        Created:  Tuesday, October 21, 2014 11:48:28 HKT
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#include "yaml-cpp/yaml.h"
#include <Eigen/Dense>
#include <iostream>
#include <fstream>

using namespace Eigen;
using namespace std;
using namespace YAML;
int main()
{
	Emitter out;
	//out << Flow;
	//out << BeginSeq << 1 << 2 << 3 << 4 << EndSeq;
	//cout << out.c_str() << endl;
	out << BeginMap;
	out << Key << "cnn_output";
	out << Value << Flow << BeginSeq << 1 << 1 << 1 << 2 << 2 << 2 << 3 << 3 << 3 << EndSeq;
	out << Key << "Label_number";
	out << Value << 1;
	out << Key << "ground_truth";
	out << Value << Flow << BeginSeq << 1 << 2 << 3 << EndSeq;
	out << Key << "w_bound";
	out << Value << 0.3;
	out << Key << "b_bound";
	out << Value << 0.3;
	out << Key << "learning_rate";
	out << Value << 0.1;
	out << EndMap;
	ofstream fout("seq.yaml");
	fout << out.c_str();		
	return 0;
}	
