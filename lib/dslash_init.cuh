// these are function declarions that are included in different namespaces

void initLatticeConstants(const LatticeField &lat, TimeProfile &profile);
void initGaugeConstants(const cudaGaugeField &gauge, TimeProfile &profile);
void initDslashConstants(TimeProfile &profile);
void initStaggeredConstants(const cudaGaugeField &fatgauge, 
			    const cudaGaugeField &longgauge, TimeProfile &profile);
void initMDWFConstants(const double *b_5, const double *c_5, int dim_s, 
		       const double m5h, TimeProfile &profile);

// this needs to be called for each dslash that has its own namespace
static void initConstants(cudaGaugeField &gauge, TimeProfile &profile) {
  initLatticeConstants(gauge, profile);
  initGaugeConstants(gauge, profile);
  initDslashConstants(profile);
}

void setFace(const FaceBuffer &Face1, const FaceBuffer &Face2);
