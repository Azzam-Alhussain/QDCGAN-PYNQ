
#include <hls_stream.h>
#include "ap_int.h"
#include <iostream>
#include <string>
#include "input_tswg.h"
#include "math.h"
using namespace hls;
using namespace std;

#define MAX_IMAGES 1

void Testbench(stream<ap_uint<IFM_Channels*INPUT_PRECISION> > & in, stream<ap_uint<IFM_Channels*INPUT_PRECISION> > & out, unsigned int numReps);

void print_img(ap_uint<INPUT_PRECISION> IMGS[MAX_IMAGES][IFMDim*IFMDim][IFM_Channels], unsigned int ch)
{
	for(int i=0; i<IFMDim; i++)
	{
		for(int j=0; j< IFMDim; j++)
		{
			ap_uint<INPUT_PRECISION> elem = IMGS[0][i*IFMDim+j][ch];
			cout << elem << "\t";
		}
		cout << endl;
	}
	cout << endl;
}

int main()
{
	static	ap_uint<INPUT_PRECISION> INPUT_IMAGES[MAX_IMAGES][IFMDim*IFMDim][IFM_Channels];
	stream<ap_uint<IFM_Channels*INPUT_PRECISION> > input_stream("input_stream");
	stream<ap_uint<IFM_Channels*INPUT_PRECISION> > output_stream("output_stream");
	unsigned int counter = 0;
	for (unsigned int n_image = 0; n_image < MAX_IMAGES; n_image++) {
		for (unsigned int oy = 0; oy < IFMDim; oy++) {
			for (unsigned int ox = 0; ox < IFMDim; ox++) {
				ap_uint<INPUT_PRECISION*IFM_Channels> input_channel = 0;
				for(unsigned int channel = 0; channel < IFM_Channels; channel++)
				{
					ap_uint<INPUT_PRECISION> input = (ap_uint<INPUT_PRECISION>)(counter);
					INPUT_IMAGES[n_image][oy*IFMDim+ox][channel]= input;
					input_channel = input_channel >> INPUT_PRECISION;
					input_channel(IFM_Channels*INPUT_PRECISION-1,(IFM_Channels-1)*INPUT_PRECISION)=input;
					counter++;
				}
				input_stream.write(input_channel);
			}
		}
	}

	unsigned int print_channel = 0;
	print_img(INPUT_IMAGES, print_channel);
	Testbench(input_stream, output_stream, MAX_IMAGES);
	int size = output_stream.size();
	cout << "Size = " << size << endl;
	for (unsigned int n_image = 0; n_image < MAX_IMAGES; n_image++) {
		for (unsigned int oy = 0; oy < OFMDim; oy++) {
			for (unsigned int ox = 0; ox < OFMDim; ox+=MMV)
			{
				cout << "Window No.: " << ox << endl;
				for (unsigned int ky = 0; ky < KERNEL_DIM; ky++)
				{
					for (unsigned int kx = 0; kx < KERNEL_DIM; kx++)
					{
						unsigned int input_base = (oy*STRIDE) * IFMDim + (ox*STRIDE);
						unsigned int input_ind = input_base + ky * IFMDim + kx;
						ap_uint<IFM_Channels*INPUT_PRECISION> outElem = output_stream.read();
						for(unsigned int channel = 0; channel < IFM_Channels; channel++)
						{
							ap_uint<INPUT_PRECISION> out_chan = 0;
							out_chan = outElem(INPUT_PRECISION-1,0);
//							if (((INPUT_IMAGES[n_image][input_ind][channel])) != out_chan){
//								std::cout << "ERROR: " <<  " Expected " << INPUT_IMAGES[n_image][input_ind][channel] << " actual " <<  out_chan << std::endl;
//								std::cout << "oy= " << oy << " ox= " << ox << " ky= " << ky << " kx= " << kx << std::endl;
//								return 1;
//							}
							if (channel == print_channel)
								cout << out_chan << "\t";
							outElem = outElem >> INPUT_PRECISION;
						}
					}
					cout << endl;
				}
				cout << endl;
			}
		}
		std::cout << "Image # " << n_image << std::endl;
	}
	return 0;

}
