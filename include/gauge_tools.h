namespace quda {
  void plaquette (const GaugeField& data, QudaFieldLocation location, double &plqS, double &plqT);
  void APEStep   (GaugeField &dataDs, const GaugeField& dataOr, double alpha, QudaFieldLocation location);
}
