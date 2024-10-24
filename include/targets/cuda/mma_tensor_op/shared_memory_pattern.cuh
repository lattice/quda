#pragma once
#include <type_traits>

namespace quda
{

  // from tmp shared memory to real-imag shared memory
  /**
     The source smem are of type `complex<T>` with 2-d layout with leading dimension of `tmp_ld`:
     - when `x` is false, source smem is row-major (n index goes faster)
     - when `x` is true, source smem is column-major (m index goes faster)
     The destination smem are of type `load_t` with 2-d layout with leading dimension of `smem_ld`:
     - destination smem is always column-major (m index goes faster)
   */
  template <class T, class load_t, bool x, int tmp_ld, int smem_ld, class Enable = void> struct tmp2s_smem_t {
  };

  /****** Specializations for float -> float, x == true ******/

  template <int tmp_ld, int smem_ld>
  struct tmp2s_smem_t<float, float, true, tmp_ld, smem_ld, std::enable_if_t<(tmp_ld % 8 == 0) && (tmp_ld / 8) % 2 == 1, void>> {

    static constexpr int get_wm() { return 8; }

    static constexpr int get_wn() { return 4 * 2; }

    static constexpr int get_number_phases() { return 2; }

    static constexpr int get_m(int lane_id, int) { return lane_id / 4; }

    static constexpr int get_n(int lane_id, int phase_id)
    {
      if (phase_id == 0) {
        return (lane_id % 4) + ((lane_id / 4) << 1 & 4);
      } else {
        return (lane_id % 4) + 4 - ((lane_id / 4) << 1 & 4);
      }
    }
  };

  template <int tmp_ld, int smem_ld>
  struct tmp2s_smem_t<float, float, true, tmp_ld, smem_ld, std::enable_if_t<(tmp_ld % 8 == 0) && (tmp_ld / 8) % 2 == 0, void>> {

    static constexpr int get_wm() { return 8; }

    static constexpr int get_wn() { return 16; }

    static constexpr int get_number_phases() { return 4; }

    static constexpr int get_m(int lane_id, int) { return lane_id / 4; }

    static constexpr int get_n(int lane_id, int phase_id) { return lane_id % 4 + ((lane_id / 4 % 4) ^ phase_id) * 4; }
  };

  /****** Specializations for short -> float, x == true ******/

  template <int tmp_ld, int smem_ld>
  struct tmp2s_smem_t<short, float, true, tmp_ld, smem_ld, std::enable_if_t<(tmp_ld % 8 == 0) && (tmp_ld / 8) % 4 == 0, void>> {

    static constexpr int get_wm() { return 8; }

    static constexpr int get_wn() { return 32; }

    static constexpr int get_number_phases() { return 8; }

    static constexpr int get_m(int lane_id, int) { return lane_id / 4; }

    static constexpr int get_n(int lane_id, int p) { return lane_id % 4 + ((lane_id / 4) ^ p) * 4; }
  };

  template <int tmp_ld, int smem_ld>
  struct tmp2s_smem_t<short, float, true, tmp_ld, smem_ld, std::enable_if_t<(tmp_ld % 8 == 0) && (tmp_ld / 8) % 4 == 2, void>> {

    static constexpr int get_wm() { return 8; }

    static constexpr int get_wn() { return 16; }

    static constexpr int get_number_phases() { return 4; }

    static constexpr int get_m(int lane_id, int) { return lane_id / 4; }

    static constexpr int get_n(int lane_id, int phase_id)
    {
      return lane_id % 4 + ((lane_id / 4 % 4) ^ (lane_id / 16 * 3 ^ phase_id)) * 4;
    }
  };

  template <int tmp_ld, int smem_ld>
  struct tmp2s_smem_t<short, float, true, tmp_ld, smem_ld,
                      std::enable_if_t<(tmp_ld % 8 == 0) && (tmp_ld / 8) % 4 % 2 == 1, void>> {

    static constexpr int get_wm() { return 8; }

    static constexpr int get_wn() { return 8; }

    static constexpr int get_number_phases() { return 2; }

    static constexpr int get_m(int lane_id, int) { return lane_id / 4; }

    static constexpr int get_n(int lane_id, int p) { return lane_id % 4 + ((lane_id / 16) ^ p) * 4; }
  };

  /****** Specializations for float -> half2, x == true ******/

  template <int tmp_ld, int smem_ld>
  struct tmp2s_smem_t<float, half2, true, tmp_ld, smem_ld, std::enable_if_t<(tmp_ld % 8 == 0) && (tmp_ld / 8) % 2 == 1, void>> {

    static constexpr int get_wm() { return 8; }

    static constexpr int get_wn() { return 4 * 2; }

    static constexpr int get_number_phases() { return 1; }

    static constexpr int get_m(int lane_id, int) { return (lane_id / 8) * 2; }

    static constexpr int get_n(int lane_id, int) { return lane_id % 8; }
  };

  template <int tmp_ld, int smem_ld>
  struct tmp2s_smem_t<float, half2, true, tmp_ld, smem_ld, std::enable_if_t<(tmp_ld % 8 == 0) && (tmp_ld / 8) % 2 == 0, void>> {

    static constexpr int get_wm() { return 8; }

    static constexpr int get_wn() { return 16; }

    static constexpr int get_number_phases() { return 2; }

    static constexpr int get_m(int lane_id, int) { return (lane_id / 4 % 4) * 2; }

    static constexpr int get_n(int lane_id, int p)
    {
      return lane_id % 4 + ((lane_id / 4 % 4) ^ (lane_id / 4 / 4 + 2 * p)) * 4;
    }
  };

  /****** Specializations for short -> half2, x == true ******/

  template <int tmp_ld, int smem_ld>
  struct tmp2s_smem_t<short, half2, true, tmp_ld, smem_ld, std::enable_if_t<(tmp_ld % 8 == 0) && (tmp_ld / 8) % 4 == 0, void>> {

    static constexpr int get_wm() { return 8; }

    static constexpr int get_wn() { return 32; }

    static constexpr int get_number_phases() { return 4; }

    static constexpr int get_m(int lane_id, int) { return (lane_id / 4 % 4) * 2; }

    static constexpr int get_n(int lane_id, int p)
    {
      return lane_id % 4 + ((lane_id / 4) ^ (p ^ (lane_id / 16 * 3))) * 4;
    }
  };

  template <int tmp_ld, int smem_ld>
  struct tmp2s_smem_t<short, half2, true, tmp_ld, smem_ld, std::enable_if_t<(tmp_ld % 8 == 0) && (tmp_ld / 8) % 4 == 2, void>> {

    static constexpr int get_wm() { return 8; }

    static constexpr int get_wn() { return 16; }

    static constexpr int get_number_phases() { return 2; }

    static constexpr int get_m(int lane_id, int) { return (lane_id / 4 % 4) * 2; }

    static constexpr int get_n(int lane_id, int p)
    {
      return lane_id % 4 + ((lane_id / 4 % 4) ^ (lane_id / 4 / 4 + 2 * p)) * 4;
    }
  };

  template <int tmp_ld, int smem_ld>
  struct tmp2s_smem_t<short, half2, true, tmp_ld, smem_ld,
                      std::enable_if_t<(tmp_ld % 8 == 0) && (tmp_ld / 8) % 4 % 2 == 1, void>> {

    static constexpr int get_wm() { return 8; }

    static constexpr int get_wn() { return 8; }

    static constexpr int get_number_phases() { return 1; }

    static constexpr int get_m(int lane_id, int) { return (lane_id / 8) * 2; }

    static constexpr int get_n(int lane_id, int) { return lane_id % 8; }
  };

  /****** Specializations for float -> float, x == false ******/

  template <int tmp_ld, int smem_ld>
  struct tmp2s_smem_t<float, float, false, tmp_ld, smem_ld, std::enable_if_t<(tmp_ld % 8 == 0) && (tmp_ld / 8) % 2 == 0, void>> {

    static constexpr int get_wm() { return 8 * 2; }

    static constexpr int get_wn() { return 4; }

    static constexpr int get_number_phases() { return 2; }

    static constexpr int get_m(int lane_id, int) { return ((lane_id / 4) % 4) + (lane_id % 4) * 4; }

    static constexpr int get_n(int lane_id, int phase_id) { return (lane_id % 4) ^ ((lane_id / 16) * 2 + phase_id); }
  };

  template <int tmp_ld, int smem_ld>
  struct tmp2s_smem_t<float, float, false, tmp_ld, smem_ld, std::enable_if_t<(tmp_ld % 8 == 0) && (tmp_ld / 8) % 2 == 1, void>> {

    static constexpr int get_wm() { return 8; }

    static constexpr int get_wn() { return 4; }

    static constexpr int get_number_phases() { return 1; }

    static constexpr int get_m(int lane_id, int) { return lane_id % 8; }

    static constexpr int get_n(int lane_id, int phase_id) { return lane_id / 8; }
  };

  /****** Specializations for short -> float, x == false ******/

  template <int tmp_ld, int smem_ld>
  struct tmp2s_smem_t<short, float, false, tmp_ld, smem_ld, std::enable_if_t<(tmp_ld % 8 == 0) && (tmp_ld / 8) % 4 == 2, void>> {

    static constexpr int get_wm() { return 16; }

    static constexpr int get_wn() { return 4; }

    static constexpr int get_number_phases() { return 2; }

    static constexpr int get_m(int lane_id, int) { return lane_id % 16; }

    static constexpr int get_n(int lane_id, int p)
    {
      if (p == 0) {
        return ((lane_id / 8) & 2) / 2 + (((lane_id / 8) & 2) | ((lane_id / 8) & 1) * 2);
      } else {
        return (((lane_id / 8) & 2) / 2 + (((lane_id / 8) & 2) | ((lane_id / 8) & 1) * 2) + 2) % 4;
      }
    }
  };

  template <int tmp_ld, int smem_ld>
  struct tmp2s_smem_t<short, float, false, tmp_ld, smem_ld,
                      std::enable_if_t<(tmp_ld % 8 == 0) && (tmp_ld / 8) % 4 % 2 == 1, void>> {

    static constexpr int get_wm() { return 8; }

    static constexpr int get_wn() { return 4; }

    static constexpr int get_number_phases() { return 1; }

    static constexpr int get_m(int lane_id, int) { return lane_id % 8; }

    static constexpr int get_n(int lane_id, int) { return lane_id / 8; }
  };

  template <int tmp_ld, int smem_ld>
  struct tmp2s_smem_t<short, float, false, tmp_ld, smem_ld, std::enable_if_t<(tmp_ld % 8 == 0) && (tmp_ld / 8) % 4 == 0, void>> {

    static constexpr int get_wm() { return 32; }

    static constexpr int get_wn() { return 1; }

    static constexpr int get_number_phases() { return 1; }

    static constexpr int get_m(int lane_id, int) { return lane_id; }

    static constexpr int get_n(int lane_id, int) { return 0; }
  };

  /****** Specializations for float -> half2, x == false ******/

  template <int tmp_ld, int smem_ld>
  struct tmp2s_smem_t<float, half2, false, tmp_ld, smem_ld, std::enable_if_t<(tmp_ld % 8 == 0) && (tmp_ld / 8) % 2 == 0, void>> {

    static constexpr int get_wm() { return 8 * 2; }

    static constexpr int get_wn() { return 8; }

    static constexpr int get_number_phases() { return 2; }

    static constexpr int get_m(int lane_id, int) { return (lane_id % 8) * 2; }

    static constexpr int get_n(int lane_id, int p) { return lane_id / 8 * 2 + p; }
  };

  template <int tmp_ld, int smem_ld>
  struct tmp2s_smem_t<float, half2, false, tmp_ld, smem_ld, std::enable_if_t<(tmp_ld % 8 == 0) && (tmp_ld / 8) % 2 == 1, void>> {

    static constexpr int get_wm() { return 8; }

    static constexpr int get_wn() { return 8; }

    static constexpr int get_number_phases() { return 1; }

    static constexpr int get_m(int lane_id, int) { return (lane_id % 4) * 2; }

    static constexpr int get_n(int lane_id, int) { return lane_id / 4; }
  };

  /****** Specializations for short -> half2, x == false ******/

  template <int tmp_ld, int smem_ld>
  struct tmp2s_smem_t<short, half2, false, tmp_ld, smem_ld, std::enable_if_t<(tmp_ld % 8 == 0) && (tmp_ld / 8) % 4 == 2, void>> {

    static constexpr int get_wm() { return 16; }

    static constexpr int get_wn() { return 8; }

    static constexpr int get_number_phases() { return 2; }

    static constexpr int get_m(int lane_id, int)
    {
      return lane_id % 8 * 2;
      ;
    }

    static constexpr int get_n(int lane_id, int p)
    {
      if (p == 0) {
        return (lane_id / 16 * 4) + (((lane_id & 8) / 4) & ((lane_id & 4) / 2)) + (lane_id / 8 % 2);
      } else {
        return ((lane_id / 16 * 4) + (((lane_id & 8) / 4) & ((lane_id & 4) / 2)) + (lane_id / 8 % 2) + 2) % 8;
      }
    }
  };

  template <int tmp_ld, int smem_ld>
  struct tmp2s_smem_t<short, half2, false, tmp_ld, smem_ld,
                      std::enable_if_t<(tmp_ld % 8 == 0) && (tmp_ld / 8) % 4 % 2 == 1, void>> {

    static constexpr int get_wm() { return 8; }

    static constexpr int get_wn() { return 8; }

    static constexpr int get_number_phases() { return 1; }

    static constexpr int get_m(int lane_id, int) { return (lane_id % 4) * 2; }

    static constexpr int get_n(int lane_id, int) { return lane_id / 4; }
  };

  template <int tmp_ld, int smem_ld>
  struct tmp2s_smem_t<short, half2, false, tmp_ld, smem_ld, std::enable_if_t<(tmp_ld % 8 == 0) && (tmp_ld / 8) % 4 == 0, void>> {

    static constexpr int get_wm() { return 32; }

    static constexpr int get_wn() { return 8; }

    static constexpr int get_number_phases() { return 4; }

    static constexpr int get_m(int lane_id, int) { return (lane_id % 16) * 2; }

    static constexpr int get_n(int lane_id, int p) { return lane_id / 16 * 4 + p; }
  };
} // namespace quda
