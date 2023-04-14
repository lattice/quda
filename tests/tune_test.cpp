#include <chrono>
#include <thread>
#include <atomic>
#include <tune_quda.h>
#include <test.h>

/*
   This test checks we can perform autotuning on arbitrary
   ranks.  We perform autotuning on tune_rank, and all other threads
   are put sleep for a period greater than the time out.  If we don't
   complete autotuning before the time out, then the test fails.
 */

using namespace quda;
constexpr auto test_timeout_secs = std::chrono::seconds(10);   // Set a timeout of 1 second
constexpr auto non_tune_rank_sleep = std::chrono::seconds(20); // Non-tuning processes sleep time longer than above

struct TuneRankTest : public Tunable, ::testing::TestWithParam<int> {

  int tune_rank;
  TuneRankTest()
  {
    switch (GetParam()) {
    case 0: tune_rank = 0; break;
    default: tune_rank = (comm_size() - 1) / GetParam();
    }
  }

  bool advanceTuneParam(TuneParam &) const override { return false; }
  unsigned int sharedBytesPerThread() const override { return 0; }
  unsigned int sharedBytesPerBlock(const TuneParam &) const override { return 0; }
  TuneKey tuneKey() const override
  {
    return TuneKey(std::to_string(comm_size()).c_str(), typeid(*this).name(), std::to_string(tune_rank).c_str());
  }
  int32_t getTuneRank() const override { return tune_rank; }

  void apply(const qudaStream_t &) override
  {
    auto tp = tuneLaunch(*this, getTuning(), getVerbosity());
    // if not the tuning rank sleep for longer than the timeout
    if (comm_rank() != tune_rank && activeTuning()) std::this_thread::sleep_for(non_tune_rank_sleep);
  }
};

TEST_P(TuneRankTest, verify)
{
  printfQuda("Running tuning on rank %d\n", tune_rank);
  std::atomic<bool> done = false; // atomic variable we use to check for completion

  // spawn a thread that does tuning
  std::thread t([&]() {
    device::init_thread(); // initalize the present device for this thread
    apply(device::get_default_stream());
    done = true; // flag completion
  });

  const auto start_time = std::chrono::steady_clock::now();
  while (!done && std::chrono::steady_clock::now() - start_time < test_timeout_secs) {
    // The thread is still running, wait a bit longer
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  // check that the thread has completed
  EXPECT_TRUE(done) << "Thread did not complete within the timeout period";
  t.join(); // thread cleanup
}

INSTANTIATE_TEST_SUITE_P(TuneTest, TuneRankTest, ::testing::Values(0, 1, 2, 3));

int main(int argc, char **argv)
{
  quda_test test("tune_rank_test", argc, argv);
  test.init();
  return test.execute();
}
