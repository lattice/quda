namespace quda {
  double	plaquette	(const GaugeField& data, QudaFieldLocation location);
  void		APEStep		(GaugeField &dataDs, const GaugeField& dataOr, double alpha, QudaFieldLocation location);

  
  void gaugefixingOVR( cudaGaugeField& data, const unsigned int gauge_dir, \
    const unsigned int Nsteps, const unsigned int verbose_interval, const double relax_boost, \
    const double tolerance, const unsigned int reunit_interval, const unsigned int stopWtheta);


  void gaugefixingFFT( cudaGaugeField& data, const unsigned int gauge_dir, \
    const unsigned int Nsteps, const unsigned int verbose_interval, const double alpha, const unsigned int autotune, \
    const double tolerance, const unsigned int stopWtheta) ;
}
