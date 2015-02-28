#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <string>
#include "pool.h"

using namespace std;

using namespace Eigen;
using namespace libcnn;

namespace libcnn{

void Pool::_pool(const MatrixXd& map_activated, const int& scale_hight, const int& scale_width, vector<MatrixXd>& coordinate, MatrixXd& map_pooled)
{
	ptrdiff_t px, py;
	map_pooled.resize(map_activated.rows()/scale_hight, map_activated.cols()/scale_width);
	
    int i = 0, j = 0;
	coordinate.clear();
	coordinate.resize(map_pooled.rows() * map_pooled.cols());

   MatrixXd position(2,1); 
   //#pragma omp parallel for schedule(dynamic,1) collapse(2) private(i,j)
	for(i = 0; i < map_pooled.rows(); i++)
	{
		for(j = 0; j < map_pooled.cols(); j++)
		{
			map_pooled(i,j) = map_activated.block(i * scale_hight, j * scale_width, scale_hight, scale_width).maxCoeff(&px, &py);
			position(0,0) = px + i * scale_hight;
			position(1,0) = py + j * scale_width;
			coordinate[i*map_pooled.cols() + j]=position;
	//		coordinate[i*map_pooled.cols() + j] << (double)(px + i * scale_hight), (double)(py + j * scale_width);
		}
	}	
		
	#ifdef POOL
	cout << "Input map:\n" << map_activated << endl;
	for(int i = 0; i < coordinate.size(); i++)
	{
		cout << "Coordinate of the " << i << "-th patch: " << coordinate[i].transpose() << endl;
	}
	cout << map_pooled << endl;
	#endif
}		

void Pool::_pool_(const vector<MatrixXd>& maps_activated, const int& scale_hight, const int& scale_width, vector<vector<MatrixXd> >& coordinates, vector<MatrixXd>& maps_pooled)
{
	coordinates.clear();
	maps_pooled.clear();
	vector<MatrixXd> coordinate_temp;
	MatrixXd map_pooled_temp;
	for(int i = 0; i < maps_activated.size(); i++)
	{
		_pool(maps_activated[i], scale_hight, scale_width, coordinate_temp, map_pooled_temp);
		coordinates.push_back(coordinate_temp);
		maps_pooled.push_back(map_pooled_temp);
	}
}

void Pool::_normalize(MatrixXd& mat, double& std)
{
	double mean = mat.mean();
	std = sqrt((((mat.array() - mean).matrix().transpose()) * ((mat.array() - mean).matrix())).trace() / (mat.rows() * mat.cols() - 1));
	mat = ((mat.array() - mean) / std). matrix();
}

void Pool::_denormalize(MatrixXd& mat, const double& std)
{
	mat = (mat.array() / std).matrix();	
}

void Pool::_normalize_(vector<vector<MatrixXd> >& batch_maps, vector<vector<double> >& batch_std)
{
	batch_std.clear();
	batch_std.reserve(batch_maps.size());
	int i, j;
	double std_temp;
	vector<double> vec_std_temp;
	for(i = 0; i < batch_maps.size(); i++)
	{
		vec_std_temp.clear();	
		vec_std_temp.reserve(batch_maps[i].size());
		for(j = 0; j < batch_maps[i].size(); j++)
		{
			_normalize(batch_maps[i][j], std_temp);
			vec_std_temp.push_back(std_temp);
		}
		batch_std.push_back(vec_std_temp);
	}
}

void Pool::_denormalize_(vector<vector<MatrixXd> >& vec_vec_mat, const vector<vector<double> >& vec_vec_std)
{
	int i, j;
	for(i = 0; i < vec_vec_mat.size(); i++)
	{
		for(j = 0; j < vec_vec_mat[i].size(); j++)
		{
			_denormalize(vec_vec_mat[i][j], vec_vec_std[i][j]);
		}
	}
}

Pool::Pool(const vector<vector<MatrixXd> >& batch_input, const int& hight, const int& width)
{
	this->_scale_hight = hight;
	this->_scale_width = width;
	compute(batch_input);
	
}

void Pool::compute(const vector<vector<MatrixXd> >& batch_activated)
{
	this->output_batch_pooled.clear();
	this->_coordinates.clear();
	vector<MatrixXd> maps_pooled_temp;
	vector<vector<MatrixXd> > coordinate_temp;
	for(int i = 0; i < batch_activated.size(); i++)
	{
		_pool_(batch_activated[i], this->_scale_hight, this->_scale_width, coordinate_temp, maps_pooled_temp);
		this->output_batch_pooled.push_back(maps_pooled_temp);
		this->_coordinates.push_back(coordinate_temp);
	}	
	_normalize_(this->output_batch_pooled, this->vec_vec_std);	
}

void Pool::back_prop(vector<vector<MatrixXd> >& grad_batch_next)
{
	/* first, denormalize the inputs */
	_denormalize_(grad_batch_next, this->vec_vec_std);
	this->grad_batch_input.clear();

	int x;
	int y;
	
	MatrixXd grad_map_conved;
	vector<MatrixXd> grad_maps_conved;

	for(int i = 0; i < this->_coordinates.size(); i++)
	{
		assert(this->_coordinates.size() == grad_batch_next.size());
		grad_maps_conved.clear();
		for(int j = 0; j < this->_coordinates[i].size(); j++)
		{
			assert(this->_coordinates[i].size() == grad_batch_next[i].size());
			grad_map_conved.setZero(grad_batch_next[0][0].rows() * this->_scale_hight, grad_batch_next[0][0].cols() * this->_scale_width);
			for(int k = 0; k < this->_coordinates[i][j].size(); k++)
			{
				x = this->_coordinates[i][j][k](0,0);
				y = this->_coordinates[i][j][k](1,0);
				grad_map_conved(x, y) = grad_batch_next[i][j](k / grad_batch_next[i][j].cols(), k % grad_batch_next[i][j].cols());
			}
			grad_maps_conved.push_back(grad_map_conved);
		}
		this->grad_batch_input.push_back(grad_maps_conved);
	}
}


}

