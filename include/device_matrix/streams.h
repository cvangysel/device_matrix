#ifndef DEVICE_MATRIX_STREAMS_H
#define DEVICE_MATRIX_STREAMS_H

namespace cuda {

class Streams {
 public:
  virtual ~Streams() {}

  virtual void synchronize() = 0;
  virtual cudaStream_t next() = 0;

  virtual size_t size() const = 0;
};

class DefaultStream : public Streams {
 public:
  static Streams* get() {
    static DefaultStream instance;

    return &instance;
  }

  virtual void synchronize() override {}

  virtual cudaStream_t next() override {
      return 0;
  }

  virtual size_t size() const override {
      return 1;
  }
};

class ScopedStreams : public Streams {
 public:
  explicit ScopedStreams(const size_t num_streams) : streams_(), it_() {
      CHECK_GT(num_streams, 0);

      initialize(num_streams);
  }

  virtual ~ScopedStreams() {
      for (size_t stream_idx = 0;
           stream_idx < streams_.size();
           ++stream_idx) {
          CCE(cudaStreamDestroy(streams_[stream_idx]));
      }
  }

  virtual inline void synchronize() override {
      for (size_t stream_idx = 0;
           stream_idx < streams_.size();
           ++stream_idx) {
          CCE(cudaStreamSynchronize(streams_[stream_idx]));
      }
  }

  virtual inline cudaStream_t next() override {
      DCHECK(!streams_.empty());

      if (it_ == streams_.end()) {
          it_ = streams_.begin();
      }

      const cudaStream_t stream = *(it_++);
      return stream;
  }

  virtual inline size_t size() const override {
      return streams_.size();
  }

 protected:
  ScopedStreams() : streams_() {}

  void initialize(const size_t num_streams) {
      streams_.resize(num_streams);

      VLOG(2) << "ScopedStreams with " << num_streams << " streams.";

      for (size_t stream_idx = 0;
           stream_idx < streams_.size();
           ++stream_idx) {
          CCE(cudaStreamCreate(&streams_[stream_idx]));
          CHECK_EQ(CNMEM_STATUS_SUCCESS, cnmemRegisterStream(streams_[stream_idx]));
      }

      it_ = streams_.begin();
  }

  std::vector<cudaStream_t> streams_;
  std::vector<cudaStream_t>::iterator it_;

 private:
  DISALLOW_COPY_AND_ASSIGN(ScopedStreams);
};

}  // namespace cuda

#endif /* DEVICE_MATRIX_STREAMS_H */