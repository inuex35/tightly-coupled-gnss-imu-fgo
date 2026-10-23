// Two-phase IMU/GNSS tight coupling (C++ port of examples/run_imu_gnss_tc.py).
//
// Phase 1: GNSS-only Pose3 RTK (stationary, lever = 0)
// Phase 2: CombinedImuFactor + DD factors (moving, lever arm)
//
// Usage:
//   run_imu_gnss_tc <rover.obs> <base.obs> <nav_file> <imu.csv> <reference.csv>
//
// Environment variables:
//   MAX_EP        : max epochs (default: end of rover.obs)
//   LEVER_ARM     : lever arm x,y,z in body FLU (default "0.31,0,0.55")
//   GNSS_CUT_FROM : force a GNSS outage from this epoch on
//   BASE_MAXAGE   : max base observation age [s] (default 30)
//   SAVE_CSV      : write per-epoch diagnostics to this path
//   (pipeline knobs: every TcConfig field is an env var — see config.h)

#include "gnss_fgo/runner.h"

#include <algorithm>
#include <cstdarg>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace gnss_fgo;

namespace {

struct RefRow {
  double tow;
  Vec3 ecef, vel;
};

std::vector<RefRow> load_reference(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::string line;
  std::getline(f, line);
  std::vector<std::string> cols;
  {
    std::stringstream ss(line);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
      const auto b = tok.find_first_not_of(" \t\r");
      const auto e = tok.find_last_not_of(" \t\r");
      cols.push_back(b == std::string::npos ? "" : tok.substr(b, e - b + 1));
    }
  }
  auto col = [&](const std::string& name) {
    auto it = std::find(cols.begin(), cols.end(), name);
    if (it == cols.end()) throw std::runtime_error("reference.csv lacks column " + name);
    return int(it - cols.begin());
  };
  const int c_tow = col("GPS TOW (s)"), c_x = col("ECEF X (m)"), c_y = col("ECEF Y (m)"),
            c_z = col("ECEF Z (m)"), c_ve = col("East Velocity (m/s)"),
            c_vn = col("North Velocity (m/s)"), c_vu = col("Up Velocity (m/s)");
  std::vector<RefRow> out;
  while (std::getline(f, line)) {
    if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
    std::vector<double> v;
    std::stringstream ss(line);
    std::string tok;
    while (std::getline(ss, tok, ',')) v.push_back(std::stod(tok));
    out.push_back({v[c_tow], Vec3(v[c_x], v[c_y], v[c_z]), Vec3(v[c_ve], v[c_vn], v[c_vu])});
  }
  return out;
}

/// Index of the reference row nearest in time (first one on a tie).
int nearest_ref(const std::vector<RefRow>& ref, double tow) {
  auto it = std::lower_bound(ref.begin(), ref.end(), tow,
                             [](const RefRow& r, double t) { return r.tow < t; });
  int i = int(it - ref.begin());
  if (i >= int(ref.size())) return int(ref.size()) - 1;
  if (i > 0 && std::abs(ref[i - 1].tow - tow) <= std::abs(ref[i].tow - tow)) return i - 1;
  return i;
}

std::string fmt(const char* f, ...) {
  char buf[512];
  va_list ap;
  va_start(ap, f);
  std::vsnprintf(buf, sizeof(buf), f, ap);
  va_end(ap);
  return buf;
}

struct Record {
  int ne = 0;
  std::string tag;
  int phase = 1, nb = 0;
  double err = 0;
  Vec3 enu = Vec3::Zero(), sol = Vec3::Zero(), truth = Vec3::Zero();
  Info info;
};

double env_d(const char* name, double dflt) {
  const char* v = std::getenv(name);
  return v && *v ? std::stod(v) : dflt;
}

void summary(const std::string& label, const std::vector<const Record*>& subset) {
  if (subset.empty()) return;
  int nfix = 0, nflt = 0;
  std::vector<double> err, fix_err;
  Vec3 sq = Vec3::Zero();
  for (const Record* r : subset) {
    nfix += r->tag == "FIX";
    nflt += r->tag == "FLT";
    err.push_back(r->err);
    sq += r->enu.cwiseProduct(r->enu);
    if (r->tag == "FIX") fix_err.push_back(r->err);
  }
  const double n = double(subset.size());
  double ss = 0;
  for (double e : err) ss += e * e;
  std::vector<double> s = err;
  std::sort(s.begin(), s.end());
  const std::size_t m = s.size();
  const double median = m % 2 ? s[m / 2] : 0.5 * (s[m / 2 - 1] + s[m / 2]);
  std::printf("\n%s (%zu ep, %d fix, %d flt):\n", label.c_str(), subset.size(), nfix, nflt);
  std::printf("  3D RMS: %.4fm, median: %.4fm\n", std::sqrt(ss / n), median);
  std::printf("  E RMS:  %.4fm\n", std::sqrt(sq[0] / n));
  std::printf("  N RMS:  %.4fm\n", std::sqrt(sq[1] / n));
  std::printf("  U RMS:  %.4fm\n", std::sqrt(sq[2] / n));
  if (!fix_err.empty()) {
    double fs = 0;
    for (double e : fix_err) fs += e * e;
    std::printf("  Fix 3D RMS: %.4fm\n", std::sqrt(fs / fix_err.size()));
  }
}

void save_csv(const std::string& path, const std::vector<Record>& results) {
  std::vector<std::string> keys;
  std::set<std::string> seen;
  for (const auto& r : results)
    for (const auto& [k, v] : r.info.values()) {
      if (std::holds_alternative<std::string>(v)) continue;
      if (seen.insert(k).second) keys.push_back(k);
    }
  std::sort(keys.begin(), keys.end());
  std::ofstream f(path);
  f << "ne,tag,phase,nb,err3d,e,n,u,sol_x,sol_y,sol_z,truth_x,truth_y,truth_z";
  for (const auto& k : keys) {
    const Info::Value* probe = nullptr;
    for (const auto& r : results)
      if ((probe = r.info.raw(k))) break;
    if (probe && std::holds_alternative<Vec3>(*probe)) f << "," << k << "_0," << k << "_1," << k << "_2";
    else f << "," << k;
  }
  f << "\n";
  char buf[64];
  auto num = [&](double v) {
    if (std::isnan(v)) return std::string("nan");
    std::snprintf(buf, sizeof(buf), "%.10g", v);
    return std::string(buf);
  };
  for (const auto& r : results) {
    f << r.ne << "," << r.tag << "," << r.phase << "," << r.nb << "," << num(r.err);
    for (int i = 0; i < 3; ++i) f << "," << num(r.enu[i]);
    for (int i = 0; i < 3; ++i) f << "," << num(r.sol[i]);
    for (int i = 0; i < 3; ++i) f << "," << num(r.truth[i]);
    for (const auto& k : keys) {
      const Info::Value* v = r.info.raw(k);
      bool vec = false;
      for (const auto& rr : results)
        if (const Info::Value* p = rr.info.raw(k)) {
          vec = std::holds_alternative<Vec3>(*p);
          break;
        }
      if (vec) {
        const Vec3 x = v && std::holds_alternative<Vec3>(*v) ? std::get<Vec3>(*v) : Vec3::Constant(nan());
        f << "," << num(x[0]) << "," << num(x[1]) << "," << num(x[2]);
      } else {
        f << "," << (v ? num(r.info.num(k, nan())) : std::string("nan"));
      }
    }
    f << "\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 6) {
    std::fprintf(stderr,
                 "Usage: %s <rover.obs> <base.obs> <nav_file> <imu.csv> <reference.csv>\n",
                 argv[0]);
    return 1;
  }
  const std::string obsfile = argv[1], basefile = argv[2], navfile = argv[3],
                    imufile = argv[4], reffile = argv[5];
  try {
    // RTKLIB frequency slots (GPS/QZSS L1/L2/L5, Galileo E1/E5b/E5a,
    // BeiDou B1I/B2I/B2a) with its code priorities. GLONASS is excluded:
    // FDMA inter-channel biases do not cancel in the double difference.
    const int nav_nf = 3;
    RinexObsReader dec(obsfile), decb(basefile);
    const auto plan = std::make_shared<const SignalPlan>(
        rtklib_signal_plan({gnss::GPS, gnss::GAL, gnss::QZS, gnss::BDS}, nav_nf));
    dec.set_signals(plan, nav_nf);
    decb.set_signals(plan, nav_nf);
    NavData navdata;
    if (!navdata.read(navfile)) throw std::runtime_error("cannot read " + navfile);

    const Vec3 base_ecef = decb.approx_pos();
    if (base_ecef.norm() == 0) {
      std::printf("Error: base position not found in RINEX header\n");
      return 1;
    }
    const std::vector<RefRow> ref = load_reference(reffile);
    std::vector<ImuSample> imu = load_imu_csv(imufile);
    std::printf("Loaded %zu IMU samples, %zu reference points\n", imu.size(), ref.size());

    const char* lever_env = std::getenv("LEVER_ARM");
    const Vec3 lever = parse_lever(lever_env ? lever_env : "0.31,0,0.55");
    const Vec3 pos0 = dec.approx_pos().norm() > 0 ? dec.approx_pos() : base_ecef;
    const TcConfig cfg = TcConfig::from_env();
    ImuGnssTc tc(navdata, nav_nf, pos0, base_ecef, std::move(imu), lever, cfg);

    long nep = 1000000000L;
    std::string nep_label = "all (until rover end)";
    if (const char* e = std::getenv("MAX_EP")) {
      std::string s(e);
      if (!s.empty() && s != "all" && s != "auto" && s != "0") {
        nep = std::stol(s);
        nep_label = s;
      }
    }
    const long gnss_cut_from = long(env_d("GNSS_CUT_FROM", -1));
    std::printf("Base: [%.4f %.4f %.4f]\n", base_ecef[0], base_ecef[1], base_ecef[2]);
    std::printf("Lever arm (FLU input): [%g, %g, %g]\n", lever[0], lever[1], lever[2]);
    std::printf("Max epochs: %s\n", nep_label.c_str());
    if (gnss_cut_from >= 0) std::printf("Forced GNSS cut from epoch: %ld\n", gnss_cut_from);
    std::printf("\n");

    std::vector<Record> results;
    double ref_match_dt_max = 0.0;
    SyncObsHold sync(dec, decb, env_d("BASE_MAXAGE", 30.0));
    auto truth_of = [&](const ObsEpoch& obs, int* ri) {
      const double tow = gnss::time2gpst(obs.t);
      *ri = nearest_ref(ref, tow);
      return tow;
    };
    auto enu_err = [&](const Vec3& sol, const Vec3& truth) {
      return gnss::ecef2enu(gnss::ecef2pos(truth), sol - truth);
    };

    for (long ne = 0; ne < nep; ++ne) {
      ObsPtr obs, obsb;
      double dt_sync;
      if (!sync.next(&obs, &obsb, &dt_sync)) break;
      const bool forced_cut = gnss_cut_from >= 0 && ne >= gnss_cut_from;
      if (forced_cut) obsb = nullptr;

      std::optional<DDMeasurements> prep;
      if (obsb) {
        if (ne == 0) tc.nav.t = obs->t;
        prep = tc.prepare_double_difference_measurements(*obs, obsb.get());
      }
      const int ns = prep ? int(prep->sat.size()) : 0;
      if (!obsb || ns < 4) {
        // Base missing / too few sats: advance the graph by IMU only.
        EpochResult r = tc.process_imu_only(*obs);
        int ri;
        truth_of(*obs, &ri);
        const Vec3 e = enu_err(r.sol, ref[ri].ecef);
        const double err3 = (r.sol - ref[ri].ecef).norm();
        const std::string reason = !obsb ? (forced_cut ? "FORCED_GNSS_CUT" : "SKIP_NOBASE")
                                         : fmt("SKIP_FEWSATS(ns=%d)", ns);
        std::printf("Ep %4ld IMU: %s E=%+.4f N=%+.4f U=%+.4f 3D=%.4fm %s\n", ne, r.tag.c_str(),
                    e[0], e[1], e[2], err3, reason.c_str());
        results.push_back({int(ne), r.tag, tc.phase, r.nb, err3, e, r.sol, ref[ri].ecef, r.info});
        continue;
      }

      EpochInput in;
      in.obs = prep->obs;     // edited by the front-end
      in.obsb = prep->obsb;
      in.rs = prep->rs;
      in.vs = prep->vs;
      in.rsb = prep->rsb;
      in.sat = prep->sat;
      in.el = prep->el;
      in.iu = prep->iu;
      for (std::size_t i = 0; i < in.obsb->sat.size(); ++i) in.ir_map[in.obsb->sat[i]] = int(i);

      int ri;
      const double tow = truth_of(*obs, &ri);
      ref_match_dt_max = std::max(ref_match_dt_max, std::abs(ref[ri].tow - tow));
      const Vec3 ref_ecef = ref[ri].ecef, ref_vel = ref[ri].vel;

      EpochResult r = tc.process(in);
      const Info& info = r.info;
      const Vec3 e = enu_err(r.sol, ref_ecef);
      const double err3 = (r.sol - ref_ecef).norm();
      const int phase = int(info.num("phase", tc.phase));
      results.push_back({int(ne), r.tag, phase, r.nb, err3, e, r.sol, ref_ecef, r.info});

      if (info.has("transition")) {
        std::printf("\n--- Phase 2 transition at epoch %ld ---\n", ne);
        std::printf("    pitch=%.1f roll=%.1f heading=%.1f deg\n", info.num("pitch"),
                    info.num("roll"), info.num("heading"));
        std::printf("    %d ambiguities (fresh init, DD at all %d collected epochs)\n",
                    int(info.num("n_amb_tc")), int(info.num("n_collected")));
        std::printf("    lever_arm=[%g, %g, %g]\n", lever[0], lever[1], lever[2]);
        if (const auto* ba = info.raw("bias_acc_init")) {
          const Vec3 b = std::get<Vec3>(*ba);
          const Vec3 g = std::get<Vec3>(*info.raw("bias_gyro_init")) * 180.0 / gnss::PI;
          std::printf("    bias_acc0=[%+.4f,%+.4f,%+.4f]\n", b[0], b[1], b[2]);
          std::printf("    bias_gyro0=[%+.4f,%+.4f,%+.4f] deg/s\n", g[0], g[1], g[2]);
        }
        std::printf("\n");
      }

      const double vel_mag = ref_vel.head<2>().norm();
      std::string phase_tag = "P" + std::to_string(phase);
      if (info.truthy("collecting") && phase == 1)
        phase_tag = "P1c" + std::to_string(int(info.num("n_collected")));
      std::string extra;
      if (info.truthy("ddpr_recover")) extra += fmt(" DDPR_RECOVER(n=%d)", int(info.num("ddpr_recover")));
      if (info.has("ddpr_fast_recover"))
        extra += fmt(" DDPR_FAST(%.2f,worst=%.1f)", info.num("ddpr_fast_recover"),
                     info.num("ddpr_fast_worst_sat_res"));
      if (info.has("prev_pose_missing"))
        extra += fmt(" PREV_MISSING(k=%d)", int(info.num("prev_pose_missing")));
      if (info.has("error") && !info.str("error").empty()) {
        std::string m = info.str("error");
        for (char& c : m) {
          if (c == '\n') c = '|';
          if (c == '(') c = '[';
          if (c == ')') c = ']';
        }
        extra += " ERR[" + m.substr(0, 200) + "]";
      }
      if (phase == 2 && info.has("bias_acc")) {
        if (int slip = int(info.num("n_slip")); slip > 0) extra += fmt(" slip=%d", slip);
        if (int fde = int(info.num("fde_reject")); fde > 0) extra += fmt(" FDE=%d", fde);
        if (info.truthy("gnss_skip"))
          extra += fmt(" SKIP(gdop=%.1f,ns=%d)", info.num("gdop"), int(info.num("nsat")));
        if (info.truthy("nhc")) extra += " NHC";
        if (info.has("lambda_correction")) extra += fmt(" λcorr=%.3f", info.num("lambda_correction"));
        if (info.truthy("ar_subset_used")) {
          std::string drop;
          for (int s : info.ar_subset_drop_sats) drop += (drop.empty() ? "" : "+") + gnss::sat_id(s);
          extra += fmt(" SUBSET_AR(%s,nb=%d,r=%.2f)", drop.empty() ? "-" : drop.c_str(),
                       int(info.num("ar_subset_nb")), info.num("ar_subset_ratio"));
        }
        if (info.has("lambda_corr_hard_reject"))
          extra += fmt(" LCHR(%.2f)", info.num("lambda_corr_hard_reject"));
        if (info.truthy("low_nb_fix_reject"))
          extra += fmt(" LOW_NB_REJ(nb=%d,lc=%.2f,mres=%.2f)", int(info.num("low_nb_fix_reject_nb")),
                       info.num("low_nb_fix_reject_lc"), info.num("low_nb_fix_reject_main_ddpr_res"));
        if (info.has("prev_pose_drift")) extra += fmt(" prev_drift=%.3f", info.num("prev_pose_drift"));
        if (info.truthy("pim_discontinuity")) extra += " PIM_BREAK";
        if (info.has("main_ddpr_res")) {
          extra += fmt(" mres=%.2f", info.num("main_ddpr_res"));
          if (info.main_ddpr_sat_worst)
            extra += fmt(" worst=%s:%.1f", gnss::sat_id(info.main_ddpr_sat_worst->first).c_str(),
                         info.main_ddpr_sat_worst->second);
        }
        if (int(info.num("held_release_flt_count")) > 0)
          extra += fmt(" HREL(%sf%d,score=%.1f,res=%.1f)",
                       gnss::sat_id(int(info.num("held_release_flt_sat"))).c_str(),
                       int(info.num("held_release_flt_freq")), info.num("held_release_flt_score"),
                       info.num("held_release_flt_res"));
      }
      std::printf("Ep %3ld %s: %s E=%+.4f N=%+.4f U=%+.4f 3D=%.4fm nb=%d vel=%.2f%s\n", ne,
                  phase_tag.c_str(), r.tag.c_str(), e[0], e[1], e[2], err3, r.nb, vel_mag,
                  extra.c_str());
    }

    std::printf("\n============================================================\n");
    std::printf("IMU/GNSS TC Results: %zu epochs\n", results.size());
    std::printf("============================================================\n");
    std::printf("  ref match max|dt|: %.3fs\n", ref_match_dt_max);
    if (results.empty()) return 0;
    std::vector<const Record*> p1, p2, all;
    for (const auto& r : results) {
      (r.phase == 1 ? p1 : p2).push_back(&r);
      all.push_back(&r);
    }
    summary("Phase 1", p1);
    summary("Phase 2", p2);
    summary("All", all);

    if (const char* path = std::getenv("SAVE_CSV")) {
      save_csv(path, results);
      std::printf("\nSaved per-epoch diagnostics to %s\n", path);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
