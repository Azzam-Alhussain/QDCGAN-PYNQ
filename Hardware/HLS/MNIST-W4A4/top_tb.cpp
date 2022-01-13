#include <iostream>
#include <string.h>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <random>
#include "testbench_utils.h"
#include "config.h"
#include <opencv2/core/core.hpp>
#include <hls_opencv.h>
using namespace std;

void load_parameters(const char* path)
{
    cout << "Setting network weights and thresholds in accelerator..." << endl;
    FoldedMVLoadLayerMem(path, 0, L0_PE, L0_WMEM, L0_TMEM, 0);
    FoldedMVLoadLayerMem(path, 1, L1_PE, L1_WMEM, L1_TMEM, 0);
    FoldedMVLoadLayerMem(path, 2, L2_PE, L2_WMEM, L2_TMEM, 0);
    FoldedMVLoadLayerMem(path, 3, L3_PE, L3_WMEM, L3_TMEM, 0);
}

void inference(const char* path, const char* path_out) 
{
    const unsigned int count = 1;
    float arr[IMG_CH];
    float usecPerImage;
    unsigned char image[L3_OFM_DIM][L3_OFM_DIM];
    ap_fixed<8, 2, AP_RND, AP_SAT> *class_result;

    // string test;
    // ifstream in_file;
    // in_file.open(path);

    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator(seed);
    normal_distribution<float> distribution(0,1);

    // ofstream ofs(path_out);
    cv::Mat imgCvOut(cv::Size(L3_OFM_DIM, L3_OFM_DIM), CV_8UC1, image, cv::Mat::AUTO_STEP);
    cv::Mat out_image;


    for(unsigned int i=0; i < IMG_CH; i++)
    {
       	// getline(in_file, test);
       	// arr[i] = atof(test.c_str());
        arr[i] = distribution(generator);
    }

    class_result=test_gan<8, 8, count, ap_fixed<8, 2, AP_RND, AP_SAT>>(arr, usecPerImage);

    unsigned char val=0;
   for(int i=0;i<L3_OFM_DIM;i++)
   {
       for(int j=0;j<L3_OFM_DIM;j++)
       {
    	   float temp = class_result[i*L3_OFM_DIM+j];
    	   temp = (temp+1)/2;
           image[i][j] = temp*255;
           // ofs << temp*255 << '\n';
       }
   }

  // ofs.close();

   cv::resize(imgCvOut, out_image, cv::Size(L3_OFM_DIM, L3_OFM_DIM));
   cv::imwrite(path_out, out_image);
   cout << "\nSample Image generated at: " << path_out << '\n' << endl;
}


int main(int argc, char** argv) {
    if (argc != 4) 
    {
        cout << "3 parameters are needed: " << endl;
        cout << "1 - folder for the weights - full path " << endl;
        cout << "2 - path to input" << endl;
        cout << "3 - output path" << endl;
        return 1;
    }
    FoldedMVInit("MNIST-W4A4");
    if (!BAKED_WEIGHTS)
    	{load_parameters(argv[1]);}
    else
    	{cout << "Using baked in weights..." << endl;}
    inference(argv[2], argv[3]);
    FoldedMVDeinit();
    return 0;
}