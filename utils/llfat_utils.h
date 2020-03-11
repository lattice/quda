#pragma once

template <typename real> struct su3_matrix {
  std::complex<real> e[3][3];
};
template <typename real> struct su3_vector {
  std::complex<real> e[3];
};

#ifdef __cplusplus
extern "C" {
#endif

void llfat_reference(void **fatlink, void **sitelink, QudaPrecision prec, void *act_path_coeff);
void llfat_reference_mg(void **fatlink, void **sitelink, void **ghost_sitelink, void **ghost_sitelink_diag,
                        QudaPrecision prec, void *act_path_coeff);

#ifdef __cplusplus
}
#endif

template <typename su3_matrix, typename Real> void llfat_scalar_mult_su3_matrix(su3_matrix *a, Real s, su3_matrix *b)
{
  int i, j;
  for (i = 0; i < 3; i++)
    for (j = 0; j < 3; j++) { b->e[i][j] = s * a->e[i][j]; }
  return;
}

template <typename su3_matrix, typename Real>
void llfat_scalar_mult_add_su3_matrix(su3_matrix *a, su3_matrix *b, Real s, su3_matrix *c)
{
  int i, j;
  for (i = 0; i < 3; i++)
    for (j = 0; j < 3; j++) { c->e[i][j] = a->e[i][j] + s * b->e[i][j]; }
}

template <typename su3_matrix> void llfat_mult_su3_na(su3_matrix *a, su3_matrix *b, su3_matrix *c)
{
  int i, j, k;
  typename std::remove_reference<decltype(a->e[0][0])>::type x, y;
  for (i = 0; i < 3; i++)
    for (j = 0; j < 3; j++) {
      x = 0.0;
      for (k = 0; k < 3; k++) {
        y = a->e[i][k] * conj(b->e[j][k]);
        x += y;
      }
      c->e[i][j] = x;
    }
}

template <typename su3_matrix> void llfat_mult_su3_nn(su3_matrix *a, su3_matrix *b, su3_matrix *c)
{
  int i, j, k;
  typename std::remove_reference<decltype(a->e[0][0])>::type x, y;
  for (i = 0; i < 3; i++)
    for (j = 0; j < 3; j++) {
      x = 0.0;
      for (k = 0; k < 3; k++) {
        y = a->e[i][k] * b->e[k][j];
        x += y;
      }
      c->e[i][j] = x;
    }
}

template <typename su3_matrix> void llfat_mult_su3_an(su3_matrix *a, su3_matrix *b, su3_matrix *c)
{
  int i, j, k;
  typename std::remove_reference<decltype(a->e[0][0])>::type x, y;
  for (i = 0; i < 3; i++)
    for (j = 0; j < 3; j++) {
      x = 0.0;
      for (k = 0; k < 3; k++) {
        y = conj(a->e[k][i]) * b->e[k][j];
        x += y;
      }
      c->e[i][j] = x;
    }
}

template <typename su3_matrix> void llfat_add_su3_matrix(su3_matrix *a, su3_matrix *b, su3_matrix *c)
{
  int i, j;
  for (i = 0; i < 3; i++)
    for (j = 0; j < 3; j++) { c->e[i][j] = a->e[i][j] + b->e[i][j]; }
}
