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
  if (!getenv("QUDA_TEST_P2P_ISLANDS")) MPI_Abort(MPI_COMM_WORLD, 2);
  Mapping map {{2, 1, 1, size / 2}, true};
  int split[4] = {1, 1, 1, 2};
  for (int i = 1; i < argc; i++) {
    std::string option(argv[i]);
    if (option == "--rank-order" && i + 1 < argc) {
      std::string order(argv[++i]);
      if (order != "col" && order != "row") MPI_Abort(MPI_COMM_WORLD, 2);
      map.t_fast = order == "col";
    } else if ((option == "--gridsize" || option == "--grid-partition") && i + 4 < argc) {
      int *values = option == "--gridsize" ? map.dims : split;
      for (int d = 0; d < 4; d++) values[d] = std::atoi(argv[++i]);
    } else {
      if (!global_rank) fprintf(stderr, "Usage: --rank-order col|row --gridsize X Y Z T --grid-partition X Y Z T\n");
      MPI_Abort(MPI_COMM_WORLD, 2);
    }
  }
  int volume = 1;
  for (int d = 0; d < 4; d++) {
    if (map.dims[d] < 1 || split[d] < 1 || map.dims[d] % split[d]) MPI_Abort(MPI_COMM_WORLD, 2);
    volume *= map.dims[d];
  }
  if (volume != size) MPI_Abort(MPI_COMM_WORLD, 2);
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
    quda::push_communicator({split[0], split[1], split[2], split[3]});
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
