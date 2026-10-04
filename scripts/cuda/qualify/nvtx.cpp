// NVTX3 is header-only; this shim contains no device computation
#include <nvtx3/nvToolsExt.h>
extern "C" void qualify_push(const char* name) { nvtxRangePushA(name); }
extern "C" void qualify_pop() { nvtxRangePop(); }
