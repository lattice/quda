namespace quda {
  
  /**
     Compute the plaquette of the gauge field

     @param U The gauge field upon which to compute the plaquette
     @param location The locaiton where to do the computation 
     @return double3 variable returning (plaquette, spatial plaquette,
     temporal plaquette) site averages normalized such that each
     plaquette is in the range [0,1]
   */
  double3 plaquette (const GaugeField& U, QudaFieldLocation location);

  /**
     Apply APE smearing to the gauge field

     @param dataDs Output smeared field
     @param dataOr Input gauge field
     @param alpha smearing parameter
     @param location Location of the computation
  */
  void APEStep (GaugeField &dataDs, const GaugeField& dataOr, double alpha, QudaFieldLocation location);
}
