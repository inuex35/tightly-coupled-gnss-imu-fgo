// Shared small types: insertion-ordered map, the per-epoch diagnostics
// record (the Python ``info`` dict), and the exception taxonomy the
// pipeline's catch sites are written against.
#pragma once

#include <Eigen/Core>

#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace gnss_fgo {

using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;

/// (sat, freq) — the identity of one ambiguity / one signal.
using SfKey = std::pair<int, int>;

/// Map that iterates in insertion order (Python dict semantics). Lookups
/// go through an index; erasure is not needed by any caller.
template <typename K, typename V>
class OrderedMap {
 public:
  using value_type = std::pair<K, V>;

  V& operator[](const K& k) {
    auto it = index_.find(k);
    if (it != index_.end()) return items_[it->second].second;
    index_.emplace(k, items_.size());
    items_.emplace_back(k, V{});
    return items_.back().second;
  }
  const V* find(const K& k) const {
    auto it = index_.find(k);
    return it == index_.end() ? nullptr : &items_[it->second].second;
  }
  V* find(const K& k) {
    auto it = index_.find(k);
    return it == index_.end() ? nullptr : &items_[it->second].second;
  }
  V get(const K& k, const V& dflt) const {
    const V* v = find(k);
    return v ? *v : dflt;
  }
  bool contains(const K& k) const { return index_.count(k) > 0; }
  std::size_t size() const { return items_.size(); }
  bool empty() const { return items_.empty(); }
  void clear() {
    items_.clear();
    index_.clear();
  }
  auto begin() { return items_.begin(); }
  auto end() { return items_.end(); }
  auto begin() const { return items_.begin(); }
  auto end() const { return items_.end(); }

 private:
  std::vector<value_type> items_;
  std::map<K, std::size_t> index_;
};

/// Python ``max(d, key=d.get)``: the first key holding the maximum value.
template <typename K>
std::optional<std::pair<K, double>> first_argmax(const OrderedMap<K, double>& m) {
  std::optional<std::pair<K, double>> best;
  for (const auto& [k, v] : m)
    if (!best || v > best->second) best = std::make_pair(k, v);
  return best;
}

/// One DDPR pair row of the main residual diagnostics.
struct PairRow {
  int ref = 0, sat = 0, freq = 0;
  double res = 0.0;
};

/// Per-epoch diagnostics: the Python ``info`` dict. Scalar entries keep
/// their insertion order for the CSV dump; the structured entries the
/// pipeline reads back are typed members.
class Info {
 public:
  using Value = std::variant<bool, int64_t, double, std::string, Vec3>;

  void set(const std::string& k, bool v) { put(k, Value(v)); }
  void set(const std::string& k, int v) { put(k, Value(int64_t(v))); }
  void set(const std::string& k, int64_t v) { put(k, Value(v)); }
  void set(const std::string& k, std::size_t v) { put(k, Value(int64_t(v))); }
  void set(const std::string& k, double v) { put(k, Value(v)); }
  void set(const std::string& k, const char* v) { put(k, Value(std::string(v))); }
  void set(const std::string& k, const std::string& v) { put(k, Value(v)); }
  void set(const std::string& k, const Vec3& v) { put(k, Value(v)); }

  bool has(const std::string& k) const { return values_.contains(k); }
  const Value* raw(const std::string& k) const { return values_.find(k); }

  /// ``float(info.get(k, dflt) or dflt)`` for numeric/bool entries.
  double num(const std::string& k, double dflt = 0.0) const {
    const Value* v = values_.find(k);
    if (!v) return dflt;
    if (auto p = std::get_if<double>(v)) return *p;
    if (auto p = std::get_if<int64_t>(v)) return double(*p);
    if (auto p = std::get_if<bool>(v)) return *p ? 1.0 : 0.0;
    return dflt;
  }
  /// Python truthiness of ``info.get(k)``.
  bool truthy(const std::string& k) const {
    const Value* v = values_.find(k);
    if (!v) return false;
    if (auto p = std::get_if<bool>(v)) return *p;
    if (auto p = std::get_if<int64_t>(v)) return *p != 0;
    if (auto p = std::get_if<double>(v)) return *p != 0.0;
    if (auto p = std::get_if<std::string>(v)) return !p->empty();
    return true;
  }
  std::string str(const std::string& k) const {
    const Value* v = values_.find(k);
    if (!v) return {};
    if (auto p = std::get_if<std::string>(v)) return *p;
    return {};
  }
  const OrderedMap<std::string, Value>& values() const { return values_; }

  // Structured entries.
  std::optional<OrderedMap<int, double>> main_ddpr_per_sat;
  std::optional<std::pair<int, double>> main_ddpr_sat_worst;
  std::optional<std::vector<PairRow>> main_ddpr_pairs;
  std::optional<std::map<int, int>> ref_sats;
  std::optional<OrderedMap<int, double>> sat_el_deg;
  std::optional<OrderedMap<int, double>> sat_snr_dbhz;
  std::vector<int> ar_subset_drop_sats;

 private:
  void put(const std::string& k, Value v) { values_[k] = std::move(v); }
  OrderedMap<std::string, Value> values_;
};

/// The Python port catches ``RuntimeError`` / ``ValueError`` /
/// ``IndexError`` at specific sites; pybind11 maps C++ exceptions onto
/// those three. Classify a caught std::exception the same way so every
/// catch site keeps its original reach.
enum class PyErr { Runtime, Value, Index };

inline PyErr classify(const std::exception& e) {
  if (dynamic_cast<const std::out_of_range*>(&e)) return PyErr::Index;
  if (dynamic_cast<const std::invalid_argument*>(&e) ||
      dynamic_cast<const std::domain_error*>(&e) ||
      dynamic_cast<const std::length_error*>(&e) ||
      dynamic_cast<const std::range_error*>(&e))
    return PyErr::Value;
  return PyErr::Runtime;
}

/// Python-style ``f"{type(ex).__name__}: {ex}"`` label.
inline std::string exc_label(const std::exception& e) {
  switch (classify(e)) {
    case PyErr::Index: return std::string("IndexError: ") + e.what();
    case PyErr::Value: return std::string("ValueError: ") + e.what();
    default: return std::string("RuntimeError: ") + e.what();
  }
}

/// Raised for rethrow-on-mismatch at catch sites that only cover some
/// categories (mirrors an uncaught Python exception).
#define GNSS_FGO_CATCH_ONLY(e, ...)                                     \
  do {                                                                  \
    const PyErr _k = classify(e);                                       \
    const PyErr _allowed[] = {__VA_ARGS__};                             \
    bool _ok = false;                                                   \
    for (PyErr a : _allowed) _ok = _ok || a == _k;                      \
    if (!_ok) throw;                                                    \
  } while (0)

inline double nan() { return std::numeric_limits<double>::quiet_NaN(); }

}  // namespace gnss_fgo
