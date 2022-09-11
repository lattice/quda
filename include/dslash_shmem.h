#include <shmem_helper.cuh>

namespace quda
{

  namespace dslash
  {
    /**
     * @brief type used for shmem signaling
     */
    using shmem_sync_t = uint64_t;

    /**
     * @brief Get the shmem sync counter
     *
     * @return shmem_sync_t
     */
    shmem_sync_t get_dslash_shmem_sync_counter();
    shmem_sync_t get_exchangeghost_shmem_sync_counter();
    /**
     * @brief Set the shmem sync counter to count
     *
     * @param count
     * @return shmem_sync_t
     */
    shmem_sync_t set_dslash_shmem_sync_counter(shmem_sync_t count);
    shmem_sync_t set_exchangeghost_shmem_sync_counter(shmem_sync_t count);
    /**
     * @brief increase the shmem sync counter for the next dslash application
     *
     * @return shmem_sync_t
     */
    shmem_sync_t inc_dslash_shmem_sync_counter();
    shmem_sync_t inc_exchangeghost_shmem_sync_counter();

#ifdef NVSHMEM_COMMS

    void shmem_signal_wait_all();

    using shmem_retcount_intra_t = cuda::atomic<int, cuda::thread_scope_system>;
    using shmem_retcount_inter_t = cuda::atomic<int, cuda::thread_scope_device>;
    using shmem_interior_done_t = cuda::atomic<shmem_sync_t, cuda::thread_scope_device>;
    using shmem_interior_count_t = cuda::atomic<int, cuda::thread_scope_block>;

    /**
     * @brief Get the shmem sync arr which is used for signaling which exterior halos have arrived
     *
     * @return shmem_sync_t*
     */
    shmem_sync_t *get_dslash_shmem_sync_arr();
    shmem_sync_t *get_exchangeghost_shmem_sync_arr();
    /**
     * @brief Get the array[2*QUDA_MAX_DIM] of atomic to count which intra node packing blocks have finished per dim/dir
     *
     * @return shmem_retcount_intra_t*
     */
    shmem_retcount_intra_t *get_shmem_retcount_intra();

    /**
     * @brief Get the array[2*QUDA_MAX_DIM] of atomic to count which inter node packing blocks have finished per dim/dir
     *
     * @return shmem_retcount_inter_t*
     */
    shmem_retcount_inter_t *get_shmem_retcount_inter();

    /**
     * @brief Get the atomic object used for signaling that the interior Dslash has been applied. Used in the uber kernel.
     *
     * @return shmem_interior_done_t*
     */
    shmem_interior_done_t *get_shmem_interior_done();

    /**
     * @brief Get the atomic counter for tracking how many of the interior blocks have finished. See also above.
     *
     * @return shmem_interior_count_t*
     */
    shmem_interior_count_t *get_shmem_interior_count();
#endif

  } // namespace dslash

} // namespace quda
