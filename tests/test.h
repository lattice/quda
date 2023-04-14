#include <util_quda.h>
#include <host_utils.h>
#include <gtest/gtest.h>
#include <command_line_params.h>

struct quda_test {

  int argc;
  char **argv;
  const std::string test_name;

  virtual void display_info() const
  {
    printfQuda("running %s on %lu ranks:\n", test_name.c_str(), quda::comm_size());
    printfQuda("Grid partition info:     X  Y  Z  T\n");
    printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),
               dimPartitioned(3));
  }

  virtual void add_command_line_group(std::shared_ptr<QUDAApp> app) const
  {
    add_comms_option_group(app);
    add_testing_option_group(app);
  }

  quda_test(const std::string &test_name, int argc, char **argv) : argc(argc), argv(argv), test_name(test_name) { }

  void init()
  {
    // Start Google Test Suite
    ::testing::InitGoogleTest(&argc, argv);

    // Parse command line options
    auto app = make_app();
    add_command_line_group(app);
    try {
      app->parse(argc, argv);
    } catch (const CLI::ParseError &e) {
      exit(app->exit(e));
    }

    // initialize QMP/MPI, QUDA comms grid and RNG (host_utils.cpp)
    initComms(argc, argv, gridsize_from_cmdline);

    display_info();

    // Initialize the QUDA library
    initQuda(device_ordinal);

    setVerbosity(verbosity);
    setQudaPrecisions(); // Set values for precisions via the command line

    initRand(); // call srand() with a rank-dependent seed

    int X[4] = {xdim, ydim, zdim, tdim};
    setDims(X);
  }

  int execute()
  {
    ::testing::TestEventListeners &listeners = ::testing::UnitTest::GetInstance()->listeners();
    if (quda::comm_rank() != 0) { delete listeners.Release(listeners.default_result_printer()); }
    return RUN_ALL_TESTS();
  }

  virtual ~quda_test()
  {
    // finalize the QUDA library
    endQuda();
    finalizeComms();
  }
};
