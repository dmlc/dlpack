// Copyright by contributors
#include <dlpack/dlpack.h>
#include <dlpack/dlpackcpp.h>

int CheckFlags(DLManagedTensorVersioned *data) {
  if (data->flags & DLPACK_FLAG_BITMASK_READ_ONLY) {
    return 0;
  } else {
    return 1;
  }
}

int main() {
  dlpack::DLTContainer c;
  return 0;
}
