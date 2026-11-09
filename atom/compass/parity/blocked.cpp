// SPDX-License-Identifier: MIT
// Each thread's wall time inside HIP's blocking calls, per call: a
// rocprofiler-sdk tool that `blocked.py` builds and reads. rocprofiler-sdk
// loads it from ROCP_TOOL_LIBRARIES when HIP initializes and traces the calls
// through HIP's dispatch table, so a caller that resolved HIP with dlsym is
// timed too. A call another traced call makes is timed as its caller's.
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <cstdint>
#include <cstring>
#include <ctime>

namespace {

constexpr auto kKind = ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API;
constexpr int kMaxCalls = 64;

const char* names[kMaxCalls];
rocprofiler_tracing_operation_t operations[kMaxCalls];
int slot[ROCPROFILER_HIP_RUNTIME_API_ID_LAST];
int calls = 0;
bool started = false;
thread_local uint64_t spent[kMaxCalls];
thread_local int depth = 0;

uint64_t now() {
  timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1000000000ull + t.tv_nsec;
}

// The calls that return only once the device has caught up: stream, event and
// device synchronizes, synchronous copies and frees.
bool blocking(const char* name) {
  if (strstr(name, "Async")) return false;
  return strstr(name, "Synchronize") || strncmp(name, "hipMemcpy", 9) == 0 ||
         strncmp(name, "hipDrvMemcpy", 12) == 0 || strncmp(name, "hipFree", 7) == 0 ||
         strcmp(name, "hipHostFree") == 0;
}

int collect(rocprofiler_callback_tracing_kind_t, rocprofiler_tracing_operation_t op,
            void*) {
  const char* name = nullptr;
  rocprofiler_query_callback_tracing_kind_operation_name(kKind, op, &name, nullptr);
  if (name && blocking(name) && op >= 0 && op < ROCPROFILER_HIP_RUNTIME_API_ID_LAST) {
    if (calls == kMaxCalls) return -1;
    slot[op] = calls;
    names[calls] = name;
    operations[calls++] = op;
  }
  return 0;
}

void on_call(rocprofiler_callback_tracing_record_t record,
             rocprofiler_user_data_t* entered, void*) {
  if (record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER) {
    entered->value = depth++ ? 0 : now();
  } else if (record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT) {
    --depth;
    if (entered->value) spent[slot[record.operation]] += now() - entered->value;
  }
}

int initialize(rocprofiler_client_finalize_t, void*) {
  rocprofiler_context_id_t context{};
  if (rocprofiler_iterate_callback_tracing_kind_operations(kKind, collect, nullptr) !=
          ROCPROFILER_STATUS_SUCCESS ||
      calls == kMaxCalls ||
      rocprofiler_create_context(&context) != ROCPROFILER_STATUS_SUCCESS ||
      rocprofiler_configure_callback_tracing_service(context, kKind, operations, calls,
                                                     on_call, nullptr) !=
          ROCPROFILER_STATUS_SUCCESS ||
      rocprofiler_start_context(context) != ROCPROFILER_STATUS_SUCCESS)
    return -1;
  started = true;
  return 0;
}

}  // namespace

extern "C" {

// The calls traced, or -1 while no tracing has started in this process.
int compass_blocked_calls() { return started ? calls : -1; }

const char* compass_blocked_name(int call) { return names[call]; }

// The calling thread's nanoseconds inside each traced call, by index.
void compass_blocked_read(uint64_t* out) { memcpy(out, spent, sizeof(uint64_t) * calls); }

rocprofiler_tool_configure_result_t* rocprofiler_configure(uint32_t, const char*, uint32_t,
                                                           rocprofiler_client_id_t* id) {
  id->name = "compass-blocked";
  static rocprofiler_tool_configure_result_t result{
      sizeof(rocprofiler_tool_configure_result_t), initialize, nullptr, nullptr};
  return &result;
}
}
