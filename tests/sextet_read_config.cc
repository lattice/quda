#include <iostream>
#include <string>
#include "math.h"
#include <arpa/inet.h>
#include <fstream>
#include "endian.h"
#include <vector>
#include <complex>
#include <iomanip>

int dir_x=0;
int dir_y=1;
int dir_z=2;
int dir_t=3;

bool is_big_endian(void)
{
    union {
        uint32_t i;
        char c[4];
    } bint = {0x01020304};

    return bint.c[0] == 1; 
}


static __inline__ void byte_swap(float *ptr){
	char*pt=(char*)ptr;
	char tmp[4];
	tmp[0]=pt[3];
	tmp[1]=pt[2];
	tmp[2]=pt[1];
	tmp[3]=pt[0];

	pt[0]=tmp[0];
	pt[1]=tmp[1];
	pt[2]=tmp[2];
	pt[3]=tmp[3];
}

void loadGeneratorFieldRaw(std::vector<std::vector<std::vector<double> > > &A, std::string &filename, int V) { 

	std::ifstream fin(filename,std::ios::in|std::ios::binary);
	std::vector<uint32_t> L(4);

	if (fin.is_open()){
		for (int i=0;i<4;i++){ 
			fin.read(reinterpret_cast<char*>(&L[i]), sizeof(L[i]));
			L[i]=le32toh(L[i]);
		}
	}


	if ((L[dir_x]*L[dir_y]*L[dir_z]*L[dir_t])!=V){
		std::cerr << "Lx Ly Lz Lt =" << L[dir_x] << " " << L[dir_y] << " " << L[dir_z] << " " << L[dir_t] << "!=V=" << V << std::endl;
		fin.close();
		abort();
	}
	std::cout << "Lx Ly Lz Lt =" << L[dir_x] << " " << L[dir_y] << " " << L[dir_z] << " " << L[dir_t]  << std::endl;
	
	int dirs[4]={dir_x,dir_y,dir_z,dir_t};
    for(int t=0;t<L[dir_t];t++)
        for(int z=0;z<L[dir_z];z++)
            for(int y=0;y<L[dir_y];y++)
                for(int x=0;x<L[dir_x];x++) {
					int index=x+L[dir_x]*y+L[dir_x]*L[dir_y]*z+L[dir_x]*L[dir_y]*L[dir_z]*t;
                    for(int dir=0;dir<4;dir++) {
						float a[8];
						for (size_t i=0;i<8;i++) {
							unsigned char tmp[sizeof(float)];
							fin.read(reinterpret_cast<char*>(tmp), sizeof(float));
							a[i]=reinterpret_cast<float&>(tmp);
						}
						if (is_big_endian()){
							for (int i=0;i<8;i++) byte_swap(&(a[i]));
						}
						//A[0][0]
						A[dirs[dir]][index][0]=0;
						A[dirs[dir]][index][1]=(a[7]/sqrt(3)+a[2])/2.;
						
						//A[0][1]
						A[dirs[dir]][index][2]=a[1]/2.;
						A[dirs[dir]][index][3]=a[0]/2.;


						//A[0][2]
						A[dirs[dir]][index][4]=a[4]/2.;
						A[dirs[dir]][index][5]=a[3]/2.;

						//A[1][0]
						A[dirs[dir]][index][6]=-a[1]/2.;
						A[dirs[dir]][index][7]=a[0]/2.;


						//A[1][1]
						A[dirs[dir]][index][8]=0;
						A[dirs[dir]][index][9]=(-a[2]+a[7]/sqrt(3))/2.;


						//A[1][2]
						A[dirs[dir]][index][10]=a[6]/2.;
						A[dirs[dir]][index][11]=a[5]/2.;


						//A[2][0]
						A[dirs[dir]][index][12]=-a[4]/2.;
						A[dirs[dir]][index][13]=a[3]/2.;


						//A[2][1]
						A[dirs[dir]][index][14]=-a[6]/2.;
						A[dirs[dir]][index][15]=a[5]/2.;


						//A[2][2]
						A[dirs[dir]][index][16]=0;
						A[dirs[dir]][index][17]=-a[7]/sqrt(3);
                    }
                }




	fin.close();

}



int main(int argc, char *argv[]){
	
	int V=32*32*32*64;

	std::string filename;
	for (int i=1;i<argc;++i){
		if (std::string(argv[i])=="--config"){
			if (i+1<argc){
				filename=argv[i+1];
				i++;
				std::cout << "config file="<<filename<<std::endl;
			}else{
				std::cerr<<"--config has no argument"<<std::endl;
				return 1;
			}
		}
	}

	std::vector<std::vector<std::vector<double> > > A(4,std::vector<std::vector<double> >(V,std::vector<double>(18)));
	loadGeneratorFieldRaw(A, filename, V); 
	return 0;
}
