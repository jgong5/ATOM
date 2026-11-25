// SPDX-License-Identifier: MIT
// Each thread's wall time inside HIP's blocking calls, per call: a
// rocprofiler-sdk tool that `blocked.py` builds and reads. rocprofiler-sdk
// loads it from ROCP_TOOL_LIBRARIES when HIP initializes and hands it HIP's
// dispatch table, whose entries for the blocking calls it replaces with timed
// ones, so a caller that resolved HIP with dlsym is timed too and no other
// call pays anything. A call another timed call makes is timed as its caller's.
#include <rocprofiler-sdk/hip.h>
#include <rocprofiler-sdk/intercept_table.h>
#include <rocprofiler-sdk/registration.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <ctime>

namespace {

constexpr int kMaxCalls = 32;

const char* names[kMaxCalls];
int calls = 0;
bool hooked = false;
thread_local uint64_t spent[kMaxCalls];
thread_local int depth = 0;

uint64_t now() {
  timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1000000000ull + t.tv_nsec;
}

template <auto Entry, typename F> struct Timed;
template <auto Entry, typename R, typename... A> struct Timed<Entry, R (*)(A...)> {
  static inline R (*call)(A...) = nullptr;
  static inline int index = 0;
  static R timed(A... args) {
    uint64_t start = depth++ ? 0 : now();
    R reply = call(args...);
    if (--depth == 0) spent[index] += now() - start;
    return reply;
  }
  static void hook(HipDispatchTable* table, const char* name) {
    call = table->*Entry;
    table->*Entry = timed;
    index = calls;
    names[calls++] = name;
  }
};

// The calls that return only once the device has caught up: stream, event and
// device synchronizes, synchronous copies and frees.
#define HOOK(name)                                                \
  if (offsetof(HipDispatchTable, name##_fn) < table->size)        \
    Timed<&HipDispatchTable::name##_fn, decltype(table->name##_fn)>::hook(table, #name);

void on_table(rocprofiler_intercept_table_t, uint64_t, uint64_t, void** tables, uint64_t,
              void*) {
  auto* table = static_cast<HipDispatchTable*>(tables[0]);
  HOOK(hipDeviceSynchronize) HOOK(hipCtxSynchronize) HOOK(hipEventSynchronize)
  HOOK(hipStreamSynchronize) HOOK(hipStreamSynchronize_spt)
  HOOK(hipMemcpy) HOOK(hipMemcpy_spt) HOOK(hipMemcpyWithStream) HOOK(hipMemcpyDtoH)
  HOOK(hipMemcpyHtoD) HOOK(hipMemcpyDtoD) HOOK(hipMemcpy2D) HOOK(hipMemcpy2D_spt)
  HOOK(hipMemcpyPeer) HOOK(hipMemcpyFromSymbol) HOOK(hipMemcpyToSymbol)
  HOOK(hipFree) HOOK(hipHostFree)
  hooked = true;
}

}  // namespace

extern "C" {

// The calls timed, or -1 while HIP has handed this process's tool no table.
int compass_blocked_calls() { return hooked ? calls : -1; }

const char* compass_blocked_name(int call) { return names[call]; }

// The calling thread's nanoseconds inside each timed call, by index.
void compass_blocked_read(uint64_t* out) { memcpy(out, spent, sizeof(uint64_t) * calls); }

rocprofiler_tool_configure_result_t* rocprofiler_configure(uint32_t, const char*, uint32_t,
                                                           rocprofiler_client_id_t* id) {
  id->name = "compass-blocked";
  rocprofiler_at_intercept_table_registration(on_table, ROCPROFILER_HIP_RUNTIME_TABLE, nullptr);
  return nullptr;
}
}
