namespace quda {
  /**
     Compute the plaquette

     @param gauge The gauge field upon which to compute the plaquette
     @param location The location of where to do the computation
     @return The plaquette
   */
  double plaquette (const GaugeField& gauge, QudaFieldLocation location);

  /**
     Apply APE smearing

     @param dataDs Destination gauge field
     @param dataOr Source gauge field
     @param alpha Smearing parameter
     @param location Location of the computation
   */
  void APEStep(GaugeField &dataDs, const GaugeField& dataOr, double alpha, QudaFieldLocation location);

  /**
     Compute the Fmunu tensor
     @param Fmunu The Fmunu tensor
     @param gauge The gauge field upon which to compute the Fmnu tensor
     @param location The location of where to do the computation
   */
  void computeFmunu(GaugeField &Fmunu, const GaugeField& gauge, QudaFieldLocation location);

}
