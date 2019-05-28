#include <stdio.h>
#include <stdlib.h>
#include <dlpack/dlpack.h>

DLManagedTensor *given = NULL;

void display(DLManagedTensor a) {
  puts("On C side:");
  int i;
  int ndim = a.dl_tensor.ndim;
  printf("data = %p\n", a.dl_tensor.data);
  printf("ctx = (device_type = %d, device_id = %d)\n",
          (int) a.dl_tensor.ctx.device_type,
          (int) a.dl_tensor.ctx.device_id);
  printf("dtype = (code = %d, bits = %d, lanes = %d)\n",
          (int) a.dl_tensor.dtype.code,
          (int) a.dl_tensor.dtype.bits,
          (int) a.dl_tensor.dtype.lanes);
  printf("ndim = %d\n",
          (int) a.dl_tensor.ndim);
  printf("shape = (");
  for (i = 0; i < ndim; ++i) {
    if (i != 0) {
      printf(", ");
    }
    printf("%d", (int) a.dl_tensor.shape[i]);
  }
  printf(")\n");
  printf("strides = (");
  for (i = 0; i < ndim; ++i) {
    if (i != 0) {
      printf(", ");
    }
    printf("%d", (int) a.dl_tensor.strides[i]);
  }
  printf(")\n");
}

void Give(DLManagedTensor dl_managed_tensor) {
  display(dl_managed_tensor);
  given = (DLManagedTensor *) malloc(sizeof(DLManagedTensor));
  *given = dl_managed_tensor;
}

void Finalize() {
  given->deleter(given);
}

void FreeHandle() {
  free(given);
  given = NULL;
}
