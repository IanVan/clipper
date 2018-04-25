#include <memory>
#include <random>
// uncomment to disable assert()
// #define NDEBUG
#include <cassert>
#include <sstream>

#include <clipper/metrics.hpp>
#include <clipper/task_executor.hpp>
#include <clipper/util.hpp>

constexpr long MY_PREDICTION_CACHE_SIZE_BYTES = 33554432;
const int INTERVAL = 100;
constexpr long RESIZE_FACTOR = 1024;

namespace clipper {

CacheEntry::CacheEntry() {}

PredictionCacheWrapper::PredictionCacheWrapper(size_t size_bytes)
    : cache1_(std::make_unique<PredictionCache>(MY_PREDICTION_CACHE_SIZE_BYTES, Policy::rand)),
      cache2_(std::make_unique<PredictionCache>(0, Policy::rand)),
      max_size_bytes_(size_bytes) {
  total_bytes = MY_PREDICTION_CACHE_SIZE_BYTES + 0;
  lookups_counter_ = metrics::MetricsRegistry::get_metrics().create_counter(
      "internal:prediction_cache_lookups_wrapper");
  hit_ratio_ = metrics::MetricsRegistry::get_metrics().create_ratio_counter(
      "internal:prediction_cache_hit_ratio_wrapper");
  modelName1 = "model1";
}

folly::Future<Output> PredictionCacheWrapper::fetch(
    const VersionedModelId &model, const std::shared_ptr<Input> &input) {
  lookups_counter_->increment(1);
  if ((lookups_counter_->value())%INTERVAL == 0) {
    CacheChange dec1 = cache1_->cacheDecision();
    CacheChange dec2 = cache2_->cacheDecision();
    //growing function
  }
  if (model.get_name().compare(modelName1)) {
    log_info_formatted("CONTAINER",
                             "called fetch in cache 1 for model name: {} and id {}",
                             model.get_name(),
                             model.get_id());
    return cache1_->fetch(model,input);
  }
  else {
    log_info_formatted("CONTAINER",
                             "called fetch in cache 2 for model name: {} and id {}",
                             model.get_name(),
                             model.get_id());
    return cache2_->fetch(model,input);
  }
}


void PredictionCacheWrapper::put(const VersionedModelId &model,
                          const std::shared_ptr<Input> &input,
                          const Output &output) {
  if (model.get_name().compare(modelName1)) {
    cache1_->put(model,input,output);
  }
  else {
    cache2_->put(model,input,output);
  }
}

void PredictionCacheWrapper::grow() {
  if (total_bytes == max_size_bytes_) {
    return;
  }
  if (cache2_->prevEpoch == increase && 
      cache1_->prevEpoch == increase && 
      (total_bytes + RESIZE_FACTOR * 2) > max_size_bytes_) {
    if (cache1_->lookups_counter_->value() > cache2_->lookups_counter_->value()) {
      if ((RESIZE_FACTOR + total_bytes) < max_size_bytes_) {
        cache1_->max_size_bytes_ += RESIZE_FACTOR;
        total_bytes += RESIZE_FACTOR;
        return;
      }
    }
    else {
      if ((RESIZE_FACTOR + total_bytes) <= max_size_bytes_) {
        cache2_->max_size_bytes_ += RESIZE_FACTOR;
        total_bytes += RESIZE_FACTOR;
        return;
      }
    }
  }
  if (cache1_->prevEpoch == increase && (total_bytes + RESIZE_FACTOR <= max_size_bytes_ )){
    cache1_->max_size_bytes_ += RESIZE_FACTOR;
    total_bytes += RESIZE_FACTOR;
  }
  if (cache2_->prevEpoch == increase && (total_bytes + RESIZE_FACTOR <= max_size_bytes_ )){
    cache2_->max_size_bytes_ += RESIZE_FACTOR;
    total_bytes += RESIZE_FACTOR;
  }
  return;
}

void PredictionCacheWrapper::shrink() {
  if (cache1_->prevEpoch == decrease) {
    cache1_->max_size_bytes_ -= RESIZE_FACTOR;
    total_bytes -= RESIZE_FACTOR;
  }
  if (cache2_->prevEpoch == decrease) {
    cache2_->max_size_bytes_ -= RESIZE_FACTOR;
    total_bytes -= RESIZE_FACTOR;
  }
}

PredictionCache::PredictionCache(size_t size_bytes, Policy policy_name) 
    : max_size_bytes_(size_bytes),
      replacement_policy_(policy_name) {
  lookups_counter_ = metrics::MetricsRegistry::get_metrics().create_counter(
      "internal:prediction_cache_lookups");
  hit_ratio_ = metrics::MetricsRegistry::get_metrics().create_ratio_counter(
      "internal:prediction_cache_hit_ratio");
}

CacheChange PredictionCache::cacheDecision() {
  double delta = hit_ratio_->get_ratio() - prevHitRatio;
  double absDelta = std::abs(delta);
  if (absDelta < 0.025) {
    return CacheChange::steady;
  }
  else if (delta < 0) {
    switch(prevEpoch) {
      case(increase) : return decrease;
      case(steady) : return increase;
      case(decrease) : return steady;
    }
  }
  else {
    return prevEpoch;
  }
}

folly::Future<Output> PredictionCache::fetch(
    const VersionedModelId &model, const std::shared_ptr<Input> &input) {
  std::unique_lock<std::mutex> l(m_);
  auto key = hash(model, input->hash());
  auto search = entries_.find(key);
  lookups_counter_->increment(1);
  if (search != entries_.end()) {
    // cache entry exists
    if (search->second.completed_) {
      // value already in cache
      hit_ratio_->increment(1, 1);
      search->second.used_ = true;
      // `makeFuture` takes an rvalue reference, so moving/forwarding
      // the cache value directly would destroy it. Therefore, we use
      // copy assignment to `value` and move the copied object instead
      Output value = search->second.value_;
      return folly::makeFuture<Output>(std::move(value));
    } else {
      // value not in cache yet
      folly::Promise<Output> new_promise;
      folly::Future<Output> new_future = new_promise.getFuture();
      search->second.value_promises_.push_back(std::move(new_promise));
      hit_ratio_->increment(0, 1);
      return new_future;
    }
  } else {
    // cache entry doesn't exist yet, so create entry
    CacheEntry new_entry;
    // create promise/future pair for this request
    folly::Promise<Output> new_promise;
    folly::Future<Output> new_future = new_promise.getFuture();
    new_entry.value_promises_.push_back(std::move(new_promise));
    insert_entry(key, new_entry);
    hit_ratio_->increment(0, 1);
    return new_future;
  }
}

void PredictionCache::put(const VersionedModelId &model,
                          const std::shared_ptr<Input> &input,
                          const Output &output) {
  std::unique_lock<std::mutex> l(m_);
  auto key = hash(model, input->hash());
  auto search = entries_.find(key);
  if (search != entries_.end()) {
    CacheEntry &entry = search->second;
    if (!entry.completed_) {
      // Complete the outstanding promises
      for (auto &p : entry.value_promises_) {
        p.setValue(std::move(output));
      }
      entry.completed_ = true;
      entry.value_ = output;
      size_bytes_ += output.y_hat_.size();
      evict_entries(size_bytes_ - max_size_bytes_);
    }
  } else {
    CacheEntry new_entry;
    new_entry.value_ = output;
    new_entry.completed_ = true;
    insert_entry(key, new_entry);
  }
}

void PredictionCache::insert_entry(const long key, CacheEntry &value) {
  size_t entry_size_bytes = value.completed_ ? value.value_.y_hat_.size() : 0;
  if (entry_size_bytes <= max_size_bytes_) {
    evict_entries(size_bytes_ + entry_size_bytes - max_size_bytes_);
    page_buffer_.insert(page_buffer_.begin() + page_buffer_index_, key);
    page_buffer_index_ = (page_buffer_index_ + 1) % page_buffer_.size();
    size_bytes_ += entry_size_bytes;
    entries_.insert(std::make_pair(key, std::move(value)));
  } else {
    // This entry is too large to cache
    log_error_formatted(LOGGING_TAG_TASK_EXECUTOR,
                        "Received an output of size: {} bytes that exceeds "
                        "cache size of: {} bytes",
                        entry_size_bytes, max_size_bytes_);
  }
}

void PredictionCache::evict_entries(long space_needed_bytes) {
  if (space_needed_bytes <= 0) {
    return;
  }
  while (space_needed_bytes > 0 && !page_buffer_.empty()) {
    if (replacement_policy_ == Policy::clock) {
      long page_key = page_buffer_[page_buffer_index_];
      auto page_entry_search = entries_.find(page_key);
      if (page_entry_search == entries_.end()) {
        throw std::runtime_error(
            "Failed to find corresponding cache entry for a buffer page!");
      }
      CacheEntry &page_entry = page_entry_search->second;
      if (page_entry.used_ || !page_entry.completed_) {
        page_entry.used_ = false;
        page_buffer_index_ = (page_buffer_index_ + 1) % page_buffer_.size();
      } else {
        page_buffer_.erase(page_buffer_.begin() + page_buffer_index_);
        page_buffer_index_ = page_buffer_.size() > 0
                                 ? page_buffer_index_ % page_buffer_.size()
                                 : 0;
        size_bytes_ -= page_entry.value_.y_hat_.size();
        space_needed_bytes -= page_entry.value_.y_hat_.size();
        entries_.erase(page_entry_search);
      }
    } else if (replacement_policy_ == Policy::lifo) {
      long page_key = page_buffer_[page_buffer_evict_pos_];
      auto page_entry_search = entries_.find(page_key);
      if (page_entry_search == entries_.end()) {
        throw std::runtime_error(
            "Failed to find corresponding cache entry for a buffer page!");
      }
      CacheEntry &page_entry = page_entry_search->second;
      if (!page_entry.completed_) {
        page_buffer_evict_pos_ = (page_buffer_evict_pos_ + 1) % page_buffer_.size();
      } else {
        page_buffer_.erase(page_buffer_.begin() + page_buffer_evict_pos_);
        page_buffer_evict_pos_ = page_buffer_.size() > 0
                                 ? page_buffer_evict_pos_ % page_buffer_.size()
                                 : 0;
        size_bytes_ -= page_entry.value_.y_hat_.size();
        space_needed_bytes -= page_entry.value_.y_hat_.size();
        entries_.erase(page_entry_search);
      }
    }
    else {
      //Random replacement
      size_t rand_page_index_ = std::rand() % page_buffer_.size();
      long page_key = page_buffer_[rand_page_index_];
      auto page_entry_search = entries_.find(page_key);
      if (page_entry_search == entries_.end()) {
        throw std::runtime_error(
            "Failed to find corresponding cache entry for a buffer page!");
      }
      CacheEntry &page_entry = page_entry_search->second;
      if (page_entry.completed_) {
        page_buffer_.erase(page_buffer_.begin() + rand_page_index_);
        size_bytes_ -= page_entry.value_.y_hat_.size();
        space_needed_bytes -= page_entry.value_.y_hat_.size();
        entries_.erase(page_entry_search);
      }
    }
  }
}

size_t PredictionCache::hash(const VersionedModelId &model,
                             size_t input_hash) const {
  std::size_t seed = 0;
  size_t model_hash = std::hash<clipper::VersionedModelId>()(model);
  boost::hash_combine(seed, model_hash);
  boost::hash_combine(seed, input_hash);
  return seed;
}

}  // namespace clipper
