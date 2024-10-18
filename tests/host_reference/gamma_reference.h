#pragma once

// clang-format off
static const double projector[10][4][4][2] = {
  {
    {{1,0}, {0,0}, {0,0}, {0,-1}},
    {{0,0}, {1,0}, {0,-1}, {0,0}},
    {{0,0}, {0,1}, {1,0}, {0,0}},
    {{0,1}, {0,0}, {0,0}, {1,0}}
  },
  {
    {{1,0}, {0,0}, {0,0}, {0,1}},
    {{0,0}, {1,0}, {0,1}, {0,0}},
    {{0,0}, {0,-1}, {1,0}, {0,0}},
    {{0,-1}, {0,0}, {0,0}, {1,0}}
  },
  {
    {{1,0}, {0,0}, {0,0}, {1,0}},
    {{0,0}, {1,0}, {-1,0}, {0,0}},
    {{0,0}, {-1,0}, {1,0}, {0,0}},
    {{1,0}, {0,0}, {0,0}, {1,0}}
  },
  {
    {{1,0}, {0,0}, {0,0}, {-1,0}},
    {{0,0}, {1,0}, {1,0}, {0,0}},
    {{0,0}, {1,0}, {1,0}, {0,0}},
    {{-1,0}, {0,0}, {0,0}, {1,0}}
  },
  {
    {{1,0}, {0,0}, {0,-1}, {0,0}},
    {{0,0}, {1,0}, {0,0}, {0,1}},
    {{0,1}, {0,0}, {1,0}, {0,0}},
    {{0,0}, {0,-1}, {0,0}, {1,0}}
  },
  {
    {{1,0}, {0,0}, {0,1}, {0,0}},
    {{0,0}, {1,0}, {0,0}, {0,-1}},
    {{0,-1}, {0,0}, {1,0}, {0,0}},
    {{0,0}, {0,1}, {0,0}, {1,0}}
  },
  {
    {{1,0}, {0,0}, {-1,0}, {0,0}},
    {{0,0}, {1,0}, {0,0}, {-1,0}},
    {{-1,0}, {0,0}, {1,0}, {0,0}},
    {{0,0}, {-1,0}, {0,0}, {1,0}}
  },
  {
    {{1,0}, {0,0}, {1,0}, {0,0}},
    {{0,0}, {1,0}, {0,0}, {1,0}},
    {{1,0}, {0,0}, {1,0}, {0,0}},
    {{0,0}, {1,0}, {0,0}, {1,0}}
  },
  { // P_+ = P_R
    {{0, 0}, {0, 0}, {0, 0}, {0, 0}},
    {{0, 0}, {0, 0}, {0, 0}, {0, 0}},
    {{0, 0}, {0, 0}, {2, 0}, {0, 0}},
    {{0, 0}, {0, 0}, {0, 0}, {2, 0}}
  },
  { // P_- = P_L
    {{2, 0}, {0, 0}, {0, 0}, {0, 0}},
    {{0, 0}, {2, 0}, {0, 0}, {0, 0}},
    {{0, 0}, {0, 0}, {0, 0}, {0, 0}},
    {{0, 0}, {0, 0}, {0, 0}, {0, 0}}
  }
};

static const double local_gamma[5][4][4][2] = {
  {// x
    {{0, 0}, {0, 0}, {0, 0}, {0, -1}},
    {{0, 0}, {0, 0}, {0, -1}, {0, 0}},
    {{0, 0}, {0, 1}, {0, 0}, {0, 0}},
    {{0, 1}, {0, 0}, {0, 0}, {0, 0}}
  },
  {// Y
    {{0, 0}, {0, 0}, {0, 0}, {1, 0}},
    {{0, 0}, {0, 0}, {-1, 0}, {0, 0}},
    {{0, 0}, {-1, 0}, {0, 0}, {0, 0}},
    {{1, 0}, {0, 0}, {0, 0}, {0, 0}}
  },
  {// Z
    {{0, 0}, {0, 0}, {0, -1}, {0, 0}},
    {{0, 0}, {0, 0}, {0, 0}, {0, 1}},
    {{0, 1}, {0, 0}, {0, 0}, {0, 0}},
    {{0, 0}, {0, -1}, {0, 0}, {0, 0}}
  },
  {// T
    {{0, 0}, {0, 0}, {-1, 0}, {0, 0}},
    {{0, 0}, {0, 0}, {0, 0}, {-1, 0}},
    {{-1, 0}, {0, 0}, {0, 0}, {0, 0}},
    {{0, 0}, {-1, 0}, {0, 0}, {0, 0}}
  },
  {// 5
    {{-1, 0}, {0, 0}, {0, 0}, {0, 0}},
    {{0, 0}, {-1, 0}, {0, 0}, {0, 0}},
    {{0, 0}, {0, 0}, {1, 0}, {0, 0}},
    {{0, 0}, {0, 0}, {0, 0}, {1, 0}}
  }
};
// clang-format on

/**
 * @brief Multiplies a spinor by a Dirac projector
 * @tparam real_t The floating point type used for the spinor
 * @param[out] res The resulting spinor after multiplication with the projector
 * @param[in] projIdx The index of the Dirac projector to use
 * @param[in] spinorIn The input spinor to be multiplied by the projector
 */
template <typename real_t> void multiplySpinorByDiracProjector(real_t *res, int projIdx, const real_t *spinorIn)
{
  for (int i = 0; i < 4 * 3 * 2; i++) res[i] = 0.0;

  for (int s = 0; s < 4; s++) {
    for (int t = 0; t < 4; t++) {
      real_t projRe = projector[projIdx][s][t][0];
      real_t projIm = projector[projIdx][s][t][1];

      for (int m = 0; m < 3; m++) {
        real_t spinorRe = spinorIn[t * (3 * 2) + m * (2) + 0];
        real_t spinorIm = spinorIn[t * (3 * 2) + m * (2) + 1];
        res[s * (3 * 2) + m * (2) + 0] += projRe * spinorRe - projIm * spinorIm;
        res[s * (3 * 2) + m * (2) + 1] += projRe * spinorIm + projIm * spinorRe;
      }
    }
  }
}

/**
 * @brief Multiplies a spinor by a Dirac matrix
 *
 * @tparam real_t The floating point type used for the spinor
 * @param[out] res The resulting spinor after multiplication with the Dirac matrix
 * @param[in] projIdx The index of the Dirac matrix to use
 * @param[in] spinorIn The input spinor to be multiplied by the Dirac matrix
 */
template <typename real_t> void multiplySpinorByDiracGamma(real_t *res, int gammaIdx, const real_t *spinorIn)
{
  for (int i = 0; i < 4 * 3 * 2; i++) res[i] = 0.0;

  for (int s = 0; s < 4; s++) {
    for (int t = 0; t < 4; t++) {
      real_t projRe = local_gamma[gammaIdx][s][t][0];
      real_t projIm = local_gamma[gammaIdx][s][t][1];

      for (int m = 0; m < 3; m++) {
        real_t spinorRe = spinorIn[t * (3 * 2) + m * (2) + 0];
        real_t spinorIm = spinorIn[t * (3 * 2) + m * (2) + 1];
        res[s * (3 * 2) + m * (2) + 0] += projRe * spinorRe - projIm * spinorIm;
        res[s * (3 * 2) + m * (2) + 1] += projRe * spinorIm + projIm * spinorRe;
      }
    }
  }
}
