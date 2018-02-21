#ifndef DEVICE_MATRIX_PROFILE_H
#define DEVICE_MATRIX_PROFILE_H

#include <nvToolsExt.h>

namespace cuda {

class ScopedProfiler {
 public:
  inline explicit ScopedProfiler(const std::string& range_name)
          : range_name_(range_name) {
      level_ = nvtxRangePush(range_name.c_str());

      VLOG_IF(4, level_ < 0) << "Unable to push '" << range_name << "' range.";
  }

  inline virtual ~ScopedProfiler() {
      if (level_ >= 0) {
          nvtxRangePop();
      }
  }

 private:
  const std::string range_name_;
  int32 level_;

  DISALLOW_COPY_AND_ASSIGN(ScopedProfiler);
};

#define PROFILE_FUNCTION() ScopedProfiler profiler(__FUNCTION__)
#define PROFILE_FUNCTION_WITH_STREAM(STREAM)\
    std::stringstream ss; ss << __FUNCTION__ << " on stream " << (STREAM);\
    VLOG(4) << "Running " << __FUNCTION__ << " on " << (STREAM) << ".";\
    ScopedProfiler profiler(ss.str())

}  // namespace cuda

#endif /* DEVICE_MATRIX_PROFILE_H */