#include <quda.h>
#include <communicator_quda.h>
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

// Run on one GPU with QUDA_ENABLE_MPS=1 and P2P enabled.  This checks
// original-rank membership across communicator renumbering in both orderings.
struct Mapping {
  int dims[4];
  bool t_fast;
};

int rank_from_coords(const int *coords, void *data)
{
  auto &map = *static_cast<Mapping *>(data);
  int rank = 0;
  for (int i = 0; i < 4; i++) {
    int d = map.t_fast ? i : 3 - i;
    rank = rank * map.dims[d] + coords[d];
  }
  return rank;
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  int global_rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size < 4 || size % 4 || !getenv("QUDA_TEST_P2P_ISLANDS")) MPI_Abort(MPI_COMM_WORLD, 2);
  Mapping map {{2, 1, 1, size / 2}, argc > 1 && std::string(argv[1]) == "t-fast"};
  initCommsGridQuda(4, map.dims, rank_from_coords, &map);
  initQuda(-1);
  std::vector<int> islands;
  std::stringstream input(getenv("QUDA_TEST_P2P_ISLANDS"));
  std::string token;
  while (std::getline(input, token, ',')) islands.push_back(std::stoi(token));
  int failures = 0;
  auto check = [&]() {
    std::vector<int> original_ranks(quda::comm_size());
    for (int rank = 0; rank < static_cast<int>(original_ranks.size()); rank++) {
      original_ranks[rank] = global_rank;
      quda::comm_broadcast(&original_ranks[rank], sizeof(int), rank);
    }
    for (int dim = 0; dim < 4; dim++) {
      for (int dir = 0; dir < 2; dir++) {
        int neighbor = quda::comm_neighbor_rank(dir, dim);
        bool expected = neighbor != quda::comm_rank() && islands[global_rank] == islands[original_ranks[neighbor]];
        if (quda::comm_peer2peer_enabled(dir, dim) != expected) failures++;
      }
    }
  };
  check();
  for (int repeat = 0; repeat < 3; repeat++) {
    quda::push_communicator({1, 1, 1, 2});
    check();
    quda::push_communicator(quda::default_comm_key);
    check();
  }
  MPI_Allreduce(MPI_IN_PLACE, &failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (!global_rank) printf("P2P island topology checks: %s (%d mismatches)\n", failures ? "FAIL" : "PASS", failures);
  endQuda();
  MPI_Finalize();
  return failures ? 1 : 0;
}
