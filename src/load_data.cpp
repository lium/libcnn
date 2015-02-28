#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <vector>
#include <string.h>
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <Eigen/Dense>
#include <cassert>
#include <algorithm>
#include <string>

using namespace Eigen;
using namespace std;

void getdir(const string & dir, vector<string> & files){
   	DIR *dp;
	struct dirent *dirp;
	char *n;
	if((dp = opendir(dir.c_str())) == NULL){
		cout << "Error opening " << dir << endl;
	}
	
	while ((dirp = readdir(dp)) != NULL){
		n = dirp->d_name;
		if(strcmp(n,".")!=0 && strcmp(n,"..")!=0 && strcmp(n, ".svn")!=0){
	//		cout << dirp->d_name << endl;
			files.push_back(string(dirp->d_name));
		}
	}
	closedir(dp);
	std::sort(files.begin(), files.end());
//	for(int i = 0; i < files.size(); i++)
//	{
//		cout << files[i] << endl;	
//	}	
}

void read_image_files(const string & dir, std::vector<vector<MatrixXd> > & vvm, const int & image_row_dim, const int & image_col_dim, const int & start ,const int & image_number)
{
	
	vector<string> files = vector<string>();
	getdir(dir, files);
	vector<MatrixXd> vm = vector<MatrixXd>();	
	MatrixXd m(image_row_dim, image_col_dim);
	m = MatrixXd::Zero(image_row_dim, image_col_dim);

	for(int i = 3 * start; i < 3 * (start + image_number); i = i + 3){
		vm.clear();	
		string tmp_U = dir + files[i];
		ifstream fin (tmp_U.c_str());
		if(i == 3 * start)
			cout << "%" << files[i] << endl;		
		if(fin.is_open()){
				for(int j = 0; j < m.rows(); j++){
					for(int k = 0; k < m.cols(); k++){
						fin >> m(j,k);
				}
			}
		}
	
		fin.close();
		vm.push_back(m);

		string tmp_V = dir + files[i+1];
		ifstream fin2 (tmp_V.c_str());
		if(fin2.is_open()){
				for(int j = 0; j < m.rows(); j++){
					for(int k = 0; k < m.cols(); k++){
						fin2 >> m(j,k);
				}
			}
		}	
	
		fin2.close();
		vm.push_back(m);
	
		string tmp_Y = dir + files[i+2];
		ifstream fin3 (tmp_Y.c_str());
		if(fin3.is_open()){
				for(int j = 0; j < m.rows(); j++){
					for(int k = 0; k < m.cols(); k++){
						fin3 >> m(j,k);
				}
			}
		}	
	
		fin3.close();
		vm.push_back(m);
		
		vvm.push_back(vm);

	}
		
//		cout << vvm.size() << endl;		

}



void read_label_files(const string & dir, VectorXi & m, const int & image_row_dim, const int & image_col_dim, const int & start, const int & image_number, const string & data_mode){
	vector<string> files = vector<string>();
	getdir(dir, files);
	m = VectorXi::Zero(image_row_dim * image_col_dim * image_number);
	
	if(data_mode=="cvpr10"){
		int index;
		for(int i = start; i < start + image_number; i++){
//			cout << files[i] << endl;
			string tmp = dir + files[i];
			ifstream fin (tmp.c_str());
			if(fin.is_open()){
				for(int j = 0; j < image_row_dim * image_col_dim; j++){
					index = (i - start) * (image_row_dim * image_col_dim) + j;
					fin >> m(index);
					if(m(index) == -1){
						m(index) = 8;
					}	
				}
			}
		}			
	}	
}

void read_markov_files(const string & dir_name, MatrixXd & markov ){
	ifstream fin (dir_name.c_str());
	if(fin.is_open()){
		for(int i = 0; i < markov.rows(); i++){
			for(int j = 0; j < markov.cols(); j++){
				fin >> markov(i,j);
			}
		}
	}
}

