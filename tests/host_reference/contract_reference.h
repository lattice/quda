#pragma once

#include <host_utils.h>
#include <quda_internal.h>
#include "color_spinor_field.h"

extern int Z[4];
extern int Vh;
extern int V;

using namespace quda;
using namespace std;

template <typename Float> void contractDegrandRossi(Float *h_result_)
{

  // Put data in complex form
  complex<Float> temp[16];
  complex<Float> *h_result = (complex<Float> *)(Float *)(h_result_);
  complex<Float> I(0.0, 1.0);

  for (int site = 0; site < V; site++) {

    // Spin contract: <\phi(x)_{\mu} \Gamma_{mu,nu}^{rho,tau} \phi(y)_{\nu}>
    // The rho index runs slowest.
    // Layout is defined in enum_quda.h: G_idx = 4*rho + tau

    int idx = 16 * site;

    int G_idx = 0;
    // G_idx = 0: I
    temp[G_idx] = 0.0;
    temp[G_idx] += h_result[idx + 4 * 0 + 0];
    temp[G_idx] += h_result[idx + 4 * 1 + 1];
    temp[G_idx] += h_result[idx + 4 * 2 + 2];
    temp[G_idx] += h_result[idx + 4 * 3 + 3];
    G_idx++;

    // G_idx = 1: \gamma_1
    temp[G_idx] = 0.0;
    temp[G_idx] += I * h_result[idx + 4 * 0 + 3];
    temp[G_idx] += I * h_result[idx + 4 * 1 + 2];
    temp[G_idx] -= I * h_result[idx + 4 * 2 + 1];
    temp[G_idx] -= I * h_result[idx + 4 * 3 + 0];
    G_idx++;

    // G_idx = 2: \gamma_2
    temp[G_idx] = 0.0;
    temp[G_idx] -= h_result[idx + 4 * 0 + 3];
    temp[G_idx] += h_result[idx + 4 * 1 + 2];
    temp[G_idx] += h_result[idx + 4 * 2 + 1];
    temp[G_idx] -= h_result[idx + 4 * 3 + 0];
    G_idx++;

    // G_idx = 3: \gamma_3
    temp[G_idx] = 0.0;
    temp[G_idx] += I * h_result[idx + 4 * 0 + 2];
    temp[G_idx] -= I * h_result[idx + 4 * 1 + 3];
    temp[G_idx] -= I * h_result[idx + 4 * 2 + 0];
    temp[G_idx] += I * h_result[idx + 4 * 3 + 1];
    G_idx++;

    // G_idx = 4: \gamma_4
    temp[G_idx] = 0.0;
    temp[G_idx] += h_result[idx + 4 * 0 + 2];
    temp[G_idx] += h_result[idx + 4 * 1 + 3];
    temp[G_idx] += h_result[idx + 4 * 2 + 0];
    temp[G_idx] += h_result[idx + 4 * 3 + 1];
    G_idx++;

    // G_idx = 5: \gamma_5
    temp[G_idx] = 0.0;
    temp[G_idx] += h_result[idx + 4 * 0 + 0];
    temp[G_idx] += h_result[idx + 4 * 1 + 1];
    temp[G_idx] -= h_result[idx + 4 * 2 + 2];
    temp[G_idx] -= h_result[idx + 4 * 3 + 3];
    G_idx++;

    // G_idx = 6: \gamma_5\gamma_1
    temp[G_idx] = 0.0;
    temp[G_idx] += I * h_result[idx + 4 * 0 + 3];
    temp[G_idx] += I * h_result[idx + 4 * 1 + 2];
    temp[G_idx] += I * h_result[idx + 4 * 2 + 1];
    temp[G_idx] += I * h_result[idx + 4 * 3 + 0];
    G_idx++;

    // G_idx = 7: \gamma_5\gamma_2
    temp[G_idx] = 0.0;
    temp[G_idx] -= h_result[idx + 4 * 0 + 3];
    temp[G_idx] += h_result[idx + 4 * 1 + 2];
    temp[G_idx] -= h_result[idx + 4 * 2 + 1];
    temp[G_idx] += h_result[idx + 4 * 3 + 0];
    G_idx++;

    // G_idx = 8: \gamma_5\gamma_3
    temp[G_idx] = 0.0;
    temp[G_idx] += I * h_result[idx + 4 * 0 + 2];
    temp[G_idx] -= I * h_result[idx + 4 * 1 + 3];
    temp[G_idx] += I * h_result[idx + 4 * 2 + 0];
    temp[G_idx] -= I * h_result[idx + 4 * 3 + 1];
    G_idx++;

    // G_idx = 9: \gamma_5\gamma_4
    temp[G_idx] = 0.0;
    temp[G_idx] += h_result[idx + 4 * 0 + 2];
    temp[G_idx] += h_result[idx + 4 * 1 + 3];
    temp[G_idx] -= h_result[idx + 4 * 2 + 0];
    temp[G_idx] -= h_result[idx + 4 * 3 + 1];
    G_idx++;

    // G_idx = 10: (i/2) * [\gamma_1, \gamma_2]
    temp[G_idx] = 0.0;
    temp[G_idx] += h_result[idx + 4 * 0 + 0];
    temp[G_idx] -= h_result[idx + 4 * 1 + 1];
    temp[G_idx] += h_result[idx + 4 * 2 + 2];
    temp[G_idx] -= h_result[idx + 4 * 3 + 3];
    G_idx++;

    // G_idx = 11: (i/2) * [\gamma_1, \gamma_3]
    temp[G_idx] = 0.0;
    temp[G_idx] -= I * h_result[idx + 4 * 0 + 2];
    temp[G_idx] -= I * h_result[idx + 4 * 1 + 3];
    temp[G_idx] += I * h_result[idx + 4 * 2 + 0];
    temp[G_idx] += I * h_result[idx + 4 * 3 + 1];
    G_idx++;

    // G_idx = 12: (i/2) * [\gamma_1, \gamma_4]
    temp[G_idx] = 0.0;
    temp[G_idx] -= h_result[idx + 4 * 0 + 1];
    temp[G_idx] -= h_result[idx + 4 * 1 + 0];
    temp[G_idx] += h_result[idx + 4 * 2 + 3];
    temp[G_idx] += h_result[idx + 4 * 3 + 2];
    G_idx++;

    // G_idx = 13: (i/2) * [\gamma_2, \gamma_3]
    temp[G_idx] = 0.0;
    temp[G_idx] += h_result[idx + 4 * 0 + 1];
    temp[G_idx] += h_result[idx + 4 * 1 + 0];
    temp[G_idx] += h_result[idx + 4 * 2 + 3];
    temp[G_idx] += h_result[idx + 4 * 3 + 2];
    G_idx++;

    // G_idx = 14: (i/2) * [\gamma_2, \gamma_3]
    temp[G_idx] = 0.0;
    temp[G_idx] -= I * h_result[idx + 4 * 0 + 1];
    temp[G_idx] += I * h_result[idx + 4 * 1 + 0];
    temp[G_idx] += I * h_result[idx + 4 * 2 + 3];
    temp[G_idx] -= I * h_result[idx + 4 * 3 + 2];
    G_idx++;

    // G_idx = 14: (i/2) * [\gamma_2, \gamma_3]
    temp[G_idx] = 0.0;
    temp[G_idx] -= h_result[idx + 4 * 0 + 0];
    temp[G_idx] -= h_result[idx + 4 * 1 + 1];
    temp[G_idx] += h_result[idx + 4 * 2 + 2];
    temp[G_idx] += h_result[idx + 4 * 3 + 3];
    G_idx++;

    // Replace data in h_result with spin contracted data
    for (int i = 0; i < 16; i++) h_result[idx + i] = temp[i];
  }
}

template <typename Float> void contractColor(Float *spinorX, Float *spinorY, Float *h_result)
{

  Float re = 0.0, im = 0.0;

  // Conjugate spinorX
  for (int i = 0; i < V; i++) {
    for (int s1 = 0; s1 < 4; s1++) {
      for (int s2 = 0; s2 < 4; s2++) {

        re = im = 0.0;
        for (int c = 0; c < 3; c++) {
          re += (((Float *)spinorX)[24 * i + 6 * s1 + 2 * c + 0] * ((Float *)spinorY)[24 * i + 6 * s2 + 2 * c + 0]
                 + ((Float *)spinorX)[24 * i + 6 * s1 + 2 * c + 1] * ((Float *)spinorY)[24 * i + 6 * s2 + 2 * c + 1]);

          im += (((Float *)spinorX)[24 * i + 6 * s1 + 2 * c + 0] * ((Float *)spinorY)[24 * i + 6 * s2 + 2 * c + 1]
                 - ((Float *)spinorX)[24 * i + 6 * s1 + 2 * c + 1] * ((Float *)spinorY)[24 * i + 6 * s2 + 2 * c + 0]);
        }

        ((Float *)h_result)[2 * (i * 16 + 4 * s1 + s2) + 0] = re;
        ((Float *)h_result)[2 * (i * 16 + 4 * s1 + s2) + 1] = im;
      }
    }
  }
}

template <typename Float>
int contraction_reference(Float *spinorX, Float *spinorY, Float *d_result, QudaContractType cType, int X[])
{

  int faults = 0;
  Float tol = (sizeof(Float) == sizeof(double) ? 1e-9 : 2e-5);
  void *h_result = malloc(V * 2 * 16 * sizeof(Float));

  // compute spin elementals
  contractColor(spinorX, spinorY, (Float *)h_result);

  // Apply gamma insertion on host spin elementals
  if (cType == QUDA_CONTRACT_TYPE_DR) contractDegrandRossi((Float *)h_result);

  // compare each contraction
  for (int j = 0; j < 16; j++) {
    bool pass = true;
    for (int i = 0; i < V; i++) {
      if (abs(((Float *)h_result)[32 * i + 2 * j] - ((Float *)d_result)[32 * i + 2 * j]) > tol) {
        faults++;
        pass = false;
        // printfQuda("Contraction %d %d failed\n", i, j);
      } else {
        // printfQuda("Contraction %d %d passed\n", i, j);
      }
      // printfQuda("%.16f %.16f\n", ((Float*)h_result)[32*i + 2*j],((Float*)d_result)[32*i + 2*j]);
      if (abs(((Float *)h_result)[32 * i + 2 * j + 1] - ((Float *)d_result)[32 * i + 2 * j + 1]) > tol) {
        faults++;
        pass = false;
        // printfQuda("Contraction %d %d failed\n", i, j);
      } else {
        // printfQuda("Contraction %d %d passed\n", i, j);
      }
      // printfQuda("%.16f %.16f\n", ((Float*)h_result)[32*i+2*j+1],((Float*)d_result)[32*i+2*j+1]);
    }
    if (pass)
      printfQuda("Contraction %d passed\n", j);
    else
      printfQuda("Contraction %d failed\n", j);
  }

  free(h_result);
  return faults;
};
