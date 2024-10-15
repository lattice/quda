#pragma once
#include <type_traits>

namespace quda {

  // from tmp shared memory to real-imag shared memory
  template <class T, bool x, int tmp_ld, int smem_ld, class Enable = void>
    struct tmp2s_smem_t { };

  template <int tmp_ld, int smem_ld>
    struct tmp2s_smem_t<float, false, tmp_ld, smem_ld, std::enable_if_t<(tmp_ld % 8 == 0) && (tmp_ld / 8) % 2 == 0, void>> {

      static constexpr int get_wm() { return 8 * 2; }

      static constexpr int get_wn() { return 4; }

      static constexpr int get_number_phases() { return 2; }
                
      static constexpr int get_m(int lane_id, int) {
        return ((lane_id / 4) % 4) + (lane_id % 4) * 4;
      }

      static constexpr int get_n(int lane_id, int phase_id) {
        return (lane_id % 4) ^ ((lane_id / 16) * 2 + phase_id);
      }

    };

  template <int tmp_ld, int smem_ld>
    struct tmp2s_smem_t<float, false, tmp_ld, smem_ld, std::enable_if_t<(tmp_ld % 8 == 0) && (tmp_ld / 8) % 2 == 1, void>> {

      static constexpr int get_wm() { return 8; }

      static constexpr int get_wn() { return 4; }

      static constexpr int get_number_phases() { return 1; }
                
      static constexpr int get_m(int lane_id, int) {
        return lane_id % 8;
      }

      static constexpr int get_n(int lane_id, int phase_id) {
        return lane_id / 8;
      }

    };

  template <int tmp_ld, int smem_ld>
    struct tmp2s_smem_t<float, true, tmp_ld, smem_ld, std::enable_if_t<(tmp_ld % 8 == 0) && (tmp_ld / 8) % 2 == 1, void>> {

      static constexpr int get_wm() { return 8; }

      static constexpr int get_wn() { return 4 * 2; }

      static constexpr int get_number_phases() { return 2; }

      static constexpr int get_m(int lane_id, int) {
        return lane_id / 4;
      }

      static constexpr int get_n(int lane_id, int phase_id) {
        if (phase_id == 0) {
          return (lane_id % 4) + ((lane_id / 4) << 1 & 4);
        } else {
          return (lane_id % 4) + 4 - ((lane_id / 4) << 1 & 4);
        }
      }

    };

  template <int tmp_ld, int smem_ld>
    struct tmp2s_smem_t<float, true, tmp_ld, smem_ld, std::enable_if_t<(tmp_ld % 8 == 0) && (tmp_ld / 8) % 2 == 0, void>> {

      static constexpr int get_wm() { return 8; }

      static constexpr int get_wn() { return 16; }

      static constexpr int get_number_phases() { return 4; }

      static constexpr int get_m(int lane_id, int) {
        return lane_id / 4;
      }

      static constexpr int get_n(int lane_id, int phase_id) {
        return lane_id % 4 + ((lane_id / 4 % 4) ^ phase_id) * 4;
      }

    };

  template <int tmp_ld, int smem_ld>
    struct tmp2s_smem_t<half2, false, tmp_ld, smem_ld, std::enable_if_t<(tmp_ld % 8 == 0) && (tmp_ld / 8) % 2 == 0, void>> {

      static constexpr int get_wm() { return 8 * 2; }

      static constexpr int get_wn() { return 8; }

      static constexpr int get_number_phases() { return 2; }
                
      static constexpr int get_m(int lane_id, int) {
        return (lane_id % 8) * 2;
      }

      static constexpr int get_n(int lane_id, int p) {
        return lane_id / 8 * 2 + p;
      }

    };

  template <int tmp_ld, int smem_ld>
    struct tmp2s_smem_t<half2, false, tmp_ld, smem_ld, std::enable_if_t<(tmp_ld % 8 == 0) && (tmp_ld / 8) % 2 == 1, void>> {

      static constexpr int get_wm() { return 8; }

      static constexpr int get_wn() { return 8; }

      static constexpr int get_number_phases() { return 1; }
                
      static constexpr int get_m(int lane_id, int) {
        return (lane_id % 4) * 2;
      }

      static constexpr int get_n(int lane_id, int) {
        return lane_id / 4;
      }

    };

  template <int tmp_ld, int smem_ld>
    struct tmp2s_smem_t<half2, true, tmp_ld, smem_ld, std::enable_if_t<(tmp_ld % 8 == 0) && (tmp_ld / 8) % 2 == 1, void>> {

      static constexpr int get_wm() { return 8; }

      static constexpr int get_wn() { return 4 * 2; }

      static constexpr int get_number_phases() { return 1; }

      static constexpr int get_m(int lane_id, int) {
        return (lane_id / 8) * 2;
      }

      static constexpr int get_n(int lane_id, int) {
        return lane_id % 8;
      }

    };

  template <int tmp_ld, int smem_ld>
    struct tmp2s_smem_t<half2, true, tmp_ld, smem_ld, std::enable_if_t<(tmp_ld % 8 == 0) && (tmp_ld / 8) % 2 == 0, void>> {

      static constexpr int get_wm() { return 8; }

      static constexpr int get_wn() { return 16; }

      static constexpr int get_number_phases() { return 2; }

      static constexpr int get_m(int lane_id, int) {
        return (lane_id / 4 % 4) * 2;
      }

      static constexpr int get_n(int lane_id, int p) {
        return lane_id % 4 + ((lane_id / 4 % 4) ^ (lane_id / 4 / 4 + 2 * p)) * 4;
      }

    };

}
