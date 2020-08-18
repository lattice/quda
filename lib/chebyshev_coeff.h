
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <vector>
#include <iostream>
#include "Eigen/Dense"

//#define __debug_chebyshev__

#ifdef __QMP__
#define print0 if(Layout::primaryNode()) printf
#else
#define print0 printf
#endif  


class chebyshev_coef : public std::vector<double>
{
protected:
	int N,hwSize;
	double eps,prec;

public: 
	double Tn(int _n, double _z, bool calc_Tdn=false){
		int i=0;
		double z=(calc_Tdn == false)? _z:2*_z;
		double a=1,b=z,d=0;
		if (_n == 0) return a;
		if (_n == 1) return b;
		for(i=0;i<_n-1;i++){
			d = 2*_z*b-a;
			a = b;
			b = d;	
		}
		return d;
	}
	
	double chebpoly(double zz){
		double b = 0.0;
		double a = 0.0;
		double tmp;

		for(int i=N-1;i>=1;i--){
			tmp = 2*zz*a-b+data()[i];
			b = a;
			a = tmp;
		}
		return zz*a-b+data()[0];

	}

	double chebpoly(double zz, Eigen::VectorXd cc){
		double b = 0.0;
		double a = 0.0;
		double tmp;

		for(int i=N-2;i>=1;i--){
			tmp = 2*zz*a-b+cc(i);
			b = a;
			a = tmp;
		}
		return 2*zz*a-b+cc(0);
	}
	
	double zeta(double y){
		return (2*y-1-eps)/(1.0-eps);
	}	
	
	double root_residual(double y0, double y1, bool derivative=false){
		double r=y1,l=y0,m,fr,fl,fm;
                int max_iter=15;
                
		fl = residual(l,derivative);
		fr = residual(r,derivative);
		if(fabs(fr) < 1e-17) return r;
		if(fabs(fl) < 1e-17) return l;
		if(fr*fl>0) print0("ERROR: find_zeros_error called with wrong ends: (%e %e)->(%e %e)\n",
			l, r, fr, fl);
		for(int i=0;i<max_iter;i++){
			m = (fl*r-fr*l)/(fl-fr);
			fm = residual(m,derivative);
			if(fm*fl>0) {l = m; fl = fm;}
			else {r = m; fr = fm;}
		}
		return (fl*r-fr*l)/(fl-fr);
	}

      	//Calculate the polynomial order needed for given residual goal err
      	int get_order()
      	{
        	int n=int( -log(prec/.4)/(2.05*sqrt(eps)));
//        	while(compute_delta(lmin, lmax, n) > err) n+=n/100;
			N=n;
			resize(N+1);
			if(compute_coef()){
				n--;
				for(;;){
					N=n;
					resize(N+1);
					if(compute_coef()) n--;
					else return n+1;
				}
			}
			else{
				n++;
				for(;;){
					N=n;
					resize(N+1);
					if(compute_coef()) return n;
					else n++;
				}
			}
      	}

      	//Calculate the coefficients with given polynomial order n in the range [lmin, lmax]. 
      	bool compute_coef()
      	{
		Eigen::VectorXd y(N+1);
		Eigen::Map<Eigen::VectorXd> c(data(), N+1, 1);

		double PI=acos(-1.0);
//		printf("PI=%13.5e\n",PI);
		for(int i=0;i<=N;i++){
			y(i)=(cos(i*PI/N)*(1-eps)+1+eps)/2.0;
		}
		
		int max_iter=5;

		for(int iter=0;iter<max_iter;iter++){
			Eigen::MatrixXd ma(N+1,N+1);
			Eigen::VectorXd ve(N+1);
			//update c
			for(int i=0;i<=N;i++){
				ve(i)=1;
			}
			for(int i=0;i<=N;i++){
				for(int j=0;j<N;j++){
//					ma(i,j)=sqrt(y(N-i))*Tn(j,zeta(y(N-i)));
					ma(i,j)=sqrt(y(i))*Tn(j,zeta(y(i)));
				}
				ma(i,N)=1-2*(i%2);
			}
			
			c = ma.lu().solve(ve);
			
			//update y
			ve(0)=1;
			ve(N)=eps;
			for(int i=1;i<N;i++){
				ve(i)=root_residual(y(i),y(i+1));	
			}	
			for(int i=N-1;i>0;i--){
				y(i)=root_residual(ve(i),ve(i-1),true);
//				if(i)exit(0);
			}
	
			//check residual
			double res=fabs(residual(y(0)));
			for(int j=0;j<N;j++){
				double tmp=fabs(residual(y(j+1)));
				if(fabs(tmp) > res) res=tmp;
			}
//			if( res <= fabs(c(N)*1.05) ) break;
			if( res <= prec ) {
//				print0("The struct cbxv_coef does convergence within %d iterations when N = %d, residual is %.5e!\n",max_iter,N,res);
				return true;
			}
			if( iter==max_iter-1 ) {	
//				print0("The struct cbxv_coef doesn't convergence within %d iterations when N = %d, residual is %.5e!\n",max_iter,N,res); 
				return false;
			}
		}
		return true;
      	}

public:

	double residual(double yy, bool derivative=false){
		double z=zeta(yy);
		if(derivative){ 
			int j=0;
			Eigen::VectorXd cc;
			cc.resize(N-1);
			for(j=0;j<N-1;j++) cc(j)=1;
			cc(0)=data()[1];
			for(j=1;j<N-1;j++) cc(j)=(j+1)*data()[j+1];
			double pp,ppd;
			pp=chebpoly(z)/(2*sqrt(yy));
			ppd=2*sqrt(yy)*chebpoly(z,cc)/(1-eps);
			return -pp-ppd;
		}
		else return 1-sqrt(yy)*chebpoly(z);
	}

	void run(double _lmin, double _prec){
		prec=_prec;
		eps=_lmin;
		N = get_order();
		resize(N+1);
		compute_coef();
	}
	
	int get_HwSize(){
		if(hwSize==-1)
			print0("HwSize hasn't been set.\n");
		return hwSize;
	}

        void set_HwSize(int size, double _lmin=1.0, double _prec=1.0, double t1=0.0, double t2=0.0){
                if(t1<1e-6||t2<1e-6) {
                        hwSize=size;
                        return;
                } 
                double fac=pow(t1*log(_prec/.4)/(2.05*size*t2),2);
                if(fac>_lmin||size==0) fac=_lmin;
                hwSize=(int)(pow(fac/_lmin,0.25)*size);
                if(hwSize<1) hwSize=1;
        }
	
      	chebyshev_coef(){hwSize=-1;}

	chebyshev_coef(double _lmin, double _prec,bool print=false):prec(_prec),eps(_lmin)
      	{
        	N = get_order();
        	if(print==true) print0("N=%d\n",N);
        	resize(N+1);
        	compute_coef();

        	if(print==true)
		{
        		print0("--------------------------------------------\n");
        		for(int i=0; i<size(); i+=5){
        			for(int j=0;j<5;j++)
        			if(i+j<size())
	        			print0("c[%4d]=%20.10e ",i+j, data()[i+j]);
        			print0("\n");
        		}
        		print0("\nlmin: %.2e lmax: %g error: %.2e order: %zu\n", _lmin, 1.0, prec, size());
        		print0("--------------------------------------------\n\n");
        		print0("y			1-|y/sqrt(y^2)|\n");
        		for(double y=_lmin/2;y<2;y*=1.1){
    				double res=residual(y);
    				int flag=abs(fabs(res)<prec)?1:0;
    				print0("%20.15e %13.5e%2d\n",y,res,flag);
        		}  
        	}    	
     	}
};

