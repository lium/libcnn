/*
 * =====================================================================================
 *
 *       Filename:  read_dat_file.cpp
 *
 *    Description:  read data file
 *
 *        Version:  1.0
 *        Created:  Wednesday, October 08, 2014 03:41:29 HKT
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <stdlib.h>
#include <sstream>
#include <cassert>
#include <vector>
using namespace std;
using namespace Eigen;

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  string_to_double
 *  Description:  
 * =====================================================================================
 */

double string_to_double(string & s)
{
        std::istringstream i(s);
        double x;
        if(!(i>>x)){
            return 0;
        }
        return x;
}
/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  main
 *  Description:  
 * =====================================================================================
 */
        
int main ( int argc, char *argv[] )
{
        //VectorXd m(9);
        std::vector<double>myvector;
        MatrixXd m(2401,1);
        //char d[3000];//m << 1,2,3,4;
        //cout << m << endl;
        string line;
        ifstream myfile ("t.txt");
        double input;
    
        if (myfile.is_open()){
                while (getline(myfile,line)){
                    input = string_to_double(line);
                    //cout << input << '\n';
                    myvector.push_back(input);
                }
             //cout << m << endl;  
             myfile.close();
        }
        else cout << "unable to open file" << endl;
            
        
        for ( unsigned i=0; i < myvector.size(); i++ ) {
                m(i,0) = myvector[i];
               //cout << myvector[i] << endl;
        }
        m.resize(49,49);        
        cout << m << endl; 
        return EXIT_SUCCESS;
}				/* ----------  end of function main  ---------- */
