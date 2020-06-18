#include <communicator_quda.h>
#include <map>
#include <array>

std::map<CommKey, Communicator> communicator_stack;

Communicator *current_communicator = nullptr;

Communicator *get_current_communicator() {
  return current_communicator;
}

static void print(const CommKey &key) {
  printf("%3dx%3dx%3dx%3d", key[0], key[1], key[2], key[3]);
}

constexpr CommKey default_key = {1, 1, 1, 1};

void init_communicator_stack(int argc, char **argv, int *const commDims) {
  communicator_stack.emplace(
          std::piecewise_construct,
          std::forward_as_tuple(default_key),
          std::forward_as_tuple(argc, argv, commDims)
  );
}

void finalize_communicator_stack() {
  communicator_stack.clear();
}

static Communicator &get_default_communicator() {
  auto search = communicator_stack.find(default_key);
  if (search != communicator_stack.end()) {
    return search->second;
  } else {
    assert(false);
  }
}

void push_to_current(const CommKey &split_key) {
  auto search = communicator_stack.find(split_key);
  if (search != communicator_stack.end()) {
    
    printf("Found communicator for key ");
    print(split_key);
    printf(".\n");
    
    current_communicator = &(search->second);
  
  } else {
     
    communicator_stack.emplace(
          std::piecewise_construct,
          std::forward_as_tuple(split_key),
          std::forward_as_tuple(get_default_communicator(), split_key.data())
    );
    
    printf("Communicator for key ");
    print(split_key);
    printf(" added.\n");

    current_communicator = &(communicator_stack[split_key]);
  }
}


