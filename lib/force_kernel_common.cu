



__constant__ int V;

__constant__ int Vhx2;
__constant__ int Vhx3;
__constant__ int Vhx4;
__constant__ int Vhx5;
__constant__ int Vhx6;
__constant__ int Vhx7;
__constant__ int Vhx8;
__constant__ int Vhx9;


static int init_kernel_cuda_flag = 0;
void
init_kernel_cuda(QudaGaugeParam* param)
{
    if (init_kernel_cuda_flag){
	return;
    }
    init_kernel_cuda_flag =1; 

    int Vh = param->X[0]*param->X[1]*param->X[2]*param->X[3]/2;
    int V = 2*Vh;
    int Vhx2 = 2*Vh;
    int Vhx3 = 3*Vh;
    int Vhx4 = 4*Vh;
    int Vhx5 = 5*Vh;
    int Vhx6 = 6*Vh;
    int Vhx7 = 7*Vh;
    int Vhx8 = 8*Vh;
    int Vhx9 = 9*Vh;

    cudaMemcpyToSymbol("V", &V, sizeof(int)); 
    cudaMemcpyToSymbol("Vhx2", &Vhx2, sizeof(int)); 
    cudaMemcpyToSymbol("Vhx3", &Vhx3, sizeof(int)); 
    cudaMemcpyToSymbol("Vhx4", &Vhx4, sizeof(int)); 
    cudaMemcpyToSymbol("Vhx5", &Vhx5, sizeof(int)); 
    cudaMemcpyToSymbol("Vhx6", &Vhx6, sizeof(int)); 
    cudaMemcpyToSymbol("Vhx7", &Vhx7, sizeof(int)); 
    cudaMemcpyToSymbol("Vhx8", &Vhx8, sizeof(int)); 
    cudaMemcpyToSymbol("Vhx9", &Vhx9, sizeof(int)); 

    return;
}
