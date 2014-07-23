namespace quda {
  double	plaquette	(const GaugeField& data, QudaFieldLocation location);
  void		APEStep		(GaugeField &dataDs, const GaugeField& dataOr, double alpha, QudaFieldLocation location);
}
