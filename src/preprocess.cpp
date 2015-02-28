#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include "preprocess.h"
#include "load_data.h"
//#define DISPLAY
using namespace std;
using namespace Eigen;
using namespace libcnn;

namespace libcnn{
void normalize(MatrixXd& mat)
{
	double mean = mat.mean();
	double variance = ((((mat.array() - mean).matrix()).transpose() * (mat.array() - mean).matrix()).trace()) / (mat.rows() * mat.cols() - 1);
	mat = ((mat.array() - mean) / sqrt(variance)).matrix();
}
/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  data_generation
 *  Description:  This is a function used to load in image files and label file in specified location. 
 * =====================================================================================
 */
void data_generation(const int& height, const int& width, const int& beginning, const int& total_images, vector<vector<MatrixXd> >& all_data, VectorXi& all_label)
{
	all_data.clear();
	all_data.reserve(total_images);
	string path_dat = "../dat/cvpr10_image_txt/";
	string path_lab = "../dat/cvpr10_label_m_loss/";  // specify data folder

	read_image_files(path_dat, all_data, height, width, beginning, total_images);
	read_label_files(path_lab, all_label, height, width, beginning, total_images, "cvpr10"); // load in data and label
	#ifdef DISPLAY
	cout << "display original data" << endl;
	cout << "number of images: " << all_data.size() << endl;
	cout << "number of channels: " << all_data[0].size() << endl;
	cout << "image rows: " << all_data[0][0].rows() << endl << "image columns: " << all_data[0][0].cols() << endl;               // display original data
	#endif
}		/* -----  end of function data_generation  ----- */


/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  unit_scaling
 *  Description:  This function is used to scale an image from 0 - 255 to 0 - 1 
 * =====================================================================================
 */
void unit_scaling(vector<vector<MatrixXd> >& all_data)
{
	for(int i = 0; i < all_data.size(); i++)
	{
		for(int j = 0; j < all_data[i].size(); j++)
		{
			normalize(all_data[i][j]);
		}
	}


	#ifdef DISPLAY
	cout << "display scaled data" << endl;
	cout << all_data[0][0] << endl << endl;
	#endif
}		/* -----  end of function unit_scaling  ----- */

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  label_sampling
 *  Description:  This function is used for sample labels from initial resolution to pooled resolution
 * =====================================================================================
 */
void label_sampling(const int& height, const int& width, const int& pool, const int& images, VectorXi& all_label, VectorXi& true_label)
{
	int pooled_width = width / pool;
	for(int i = 0; i < height * images; i++)
	{
		for(int j = 0; j < width; j++)
		{
			if(i % pool == 0 && j % pool == 0)
			{
				true_label(i / pool * pooled_width + j / pool) = all_label(i * width + j);
			}
		}
	}
}		/* -----  end of function label_sampling  ----- */

}
