// dft_benchmark_cpu.cpp
// CPU-only 2D DFT baseline (NO OpenCV, NO CUDA)
// Now additionally saves the spectrum image (magnitude, log-scaled) for EACH saved image in each test.
//
// Tests:
//  A Gaussian low-pass blur
//  B Gaussian high-pass edges
//  C Band-pass texture emphasis
//  D Checkerboard (stress)
//  E Periodic noise -> auto notch denoise (auto only)
//  F Radial rings -> band-stop
//
// Build:
//   g++ -O3 -std=c++17 dft_filters_cpu.cpp -o dft_filters_cpu
//
// Run:
//   ./dft_filters_cpu 1024 3 out
// Args: N runs outdir
//
// Notes about spectrum saving:
//  - This code uses dft2(..., shift=true), so the DC/low-frequency is ALREADY centered in F.
//  - Therefore spectrum_to_image(..., shift_center=false) is used (no extra swapping).

#include <cmath>
#include <complex>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <limits>

using cd = std::complex<double>;
static constexpr double PI = 3.14159265358979323846;

// -------------------- Timer --------------------
struct Timer
{
    std::chrono::high_resolution_clock::time_point t0;
    void tic() { t0 = std::chrono::high_resolution_clock::now(); }
    double toc_ms() const
    {
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
};

template <typename Fn>
double avg_ms(int runs, Fn fn)
{
    Timer t;
    double total = 0.0;
    for (int i = 0; i < runs; ++i)
    {
        t.tic();
        fn();
        total += t.toc_ms();
    }
    return total / std::max(1, runs);
}

// -------------------- Image --------------------
struct Image
{
    int w, h;
    std::vector<double> px; // grayscale double
    Image() : w(0), h(0) {}
    Image(int W, int H) : w(W), h(H), px((size_t)W * H, 0.0) {}
    inline double &at(int x, int y) { return px[(size_t)y * w + x]; }
    inline double at(int x, int y) const { return px[(size_t)y * w + x]; }
};

static inline double clamp255(double v)
{
    if (v < 0.0)
        return 0.0;
    if (v > 255.0)
        return 255.0;
    return v;
}

bool write_pgm(const std::string &path, const Image &img)
{
    std::ofstream out(path, std::ios::binary);
    if (!out)
        return false;
    out << "P5\n"
        << img.w << " " << img.h << "\n255\n";
    for (int y = 0; y < img.h; ++y)
    {
        for (int x = 0; x < img.w; ++x)
        {
            uint8_t v = (uint8_t)std::lround(clamp255(img.at(x, y)));
            out.write((const char *)&v, 1);
        }
    }
    return true;
}

Image normalize_to_0_255(const Image &in)
{
    Image out(in.w, in.h);
    double mn = 1e300, mx = -1e300;
    for (double v : in.px)
    {
        mn = std::min(mn, v);
        mx = std::max(mx, v);
    }
    double den = (mx - mn) + 1e-12;
    for (size_t i = 0; i < in.px.size(); ++i)
        out.px[i] = (in.px[i] - mn) / den * 255.0;
    return out;
}

// -------------------- Spectrum visualization --------------------
// Convert complex spectrum F to an image: log(1+|F|) -> normalize to 0..255.
// shift_center=true would swap quadrants to move DC to center.
// BUT: in this program, dft2 uses shift=true (multiplying (-1)^(x+y)), so DC is already centered.
// Therefore we will call spectrum_to_image(F, ..., shift_center=false).
Image spectrum_to_image(const std::vector<cd> &F, int w, int h, bool shift_center)
{
    std::vector<double> mag((size_t)w * h, 0.0);
    auto idx = [&](int x, int y)
    { return (size_t)y * w + x; };

    double maxv = 1e-12;
    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            int sx = x, sy = y;
            if (shift_center)
            {
                sx = (x + w / 2) % w;
                sy = (y + h / 2) % h;
            }
            double m = std::abs(F[idx(sx, sy)]);
            double v = std::log(1.0 + m);
            mag[idx(x, y)] = v;
            maxv = std::max(maxv, v);
        }
    }

    Image out(w, h);
    for (size_t i = 0; i < mag.size(); ++i)
        out.px[i] = (mag[i] / maxv) * 255.0;
    return out;
}

// -------------------- DFT (separable 1D) --------------------
struct Twiddle1D
{
    int N;
    std::vector<cd> W;    // forward: exp(-i2pi kn/N)
    std::vector<cd> Winv; // inverse: exp(+i2pi kn/N)
    explicit Twiddle1D(int n) : N(n), W((size_t)n * n), Winv((size_t)n * n)
    {
        for (int k = 0; k < N; ++k)
        {
            for (int n0 = 0; n0 < N; ++n0)
            {
                double ang = 2.0 * PI * (double)k * (double)n0 / (double)N;
                W[(size_t)k * N + n0] = cd(std::cos(-ang), std::sin(-ang));
                Winv[(size_t)k * N + n0] = cd(std::cos(+ang), std::sin(+ang));
            }
        }
    }
};

void dft1d(const cd *in, cd *out, int N, const std::vector<cd> &table)
{
    for (int k = 0; k < N; ++k)
    {
        cd s(0, 0);
        const cd *row = &table[(size_t)k * N];
        for (int n = 0; n < N; ++n)
            s += in[n] * row[n];
        out[k] = s;
    }
}

// shift=true multiplies input by (-1)^(x+y) so DC is centered in frequency
std::vector<cd> dft2(const Image &img, const Twiddle1D &tw, bool shift = true)
{
    int w = img.w, h = img.h;
    if (w != h)
    {
        std::cerr << "Only square supported.\n";
        std::exit(1);
    }
    int N = w;
    auto id = [&](int x, int y)
    { return (size_t)y * N + x; };

    std::vector<cd> temp((size_t)N * N);
    std::vector<cd> F((size_t)N * N);

    std::vector<cd> inrow(N), outrow(N);

    // Row DFT
    for (int y = 0; y < N; ++y)
    {
        for (int x = 0; x < N; ++x)
        {
            double f = img.at(x, y);
            if (shift && ((x + y) & 1))
                f = -f;
            inrow[x] = cd(f, 0.0);
        }
        dft1d(inrow.data(), outrow.data(), N, tw.W);
        for (int u = 0; u < N; ++u)
            temp[id(u, y)] = outrow[u];
    }

    // Col DFT
    std::vector<cd> incol(N), outcol(N);
    for (int u = 0; u < N; ++u)
    {
        for (int y = 0; y < N; ++y)
            incol[y] = temp[id(u, y)];
        dft1d(incol.data(), outcol.data(), N, tw.W);
        for (int v = 0; v < N; ++v)
            F[id(u, v)] = outcol[v];
    }

    return F;
}

Image idft2(const std::vector<cd> &F, const Twiddle1D &tw, int N, bool shift = true)
{
    auto id = [&](int x, int y)
    { return (size_t)y * N + x; };

    std::vector<cd> temp((size_t)N * N);
    std::vector<cd> out((size_t)N * N);

    // inverse columns
    std::vector<cd> incol(N), outcol(N);
    for (int u = 0; u < N; ++u)
    {
        for (int v = 0; v < N; ++v)
            incol[v] = F[id(u, v)];
        dft1d(incol.data(), outcol.data(), N, tw.Winv);
        for (int y = 0; y < N; ++y)
            temp[id(u, y)] = outcol[y];
    }

    // inverse rows
    std::vector<cd> inrow(N), outrow(N);
    for (int y = 0; y < N; ++y)
    {
        for (int u = 0; u < N; ++u)
            inrow[u] = temp[id(u, y)];
        dft1d(inrow.data(), outrow.data(), N, tw.Winv);
        for (int x = 0; x < N; ++x)
            out[id(x, y)] = outrow[x];
    }

    Image img(N, N);
    double inv = 1.0 / (double)(N * N);
    for (int y = 0; y < N; ++y)
    {
        for (int x = 0; x < N; ++x)
        {
            double f = out[id(x, y)].real() * inv;
            if (shift && ((x + y) & 1))
                f = -f;
            img.at(x, y) = f;
        }
    }
    return img;
}

// -------------------- Synthetic inputs --------------------
Image make_scene_mix(int N)
{
    Image img(N, N);
    int cx = N / 2, cy = N / 2;

    // smooth gradient
    for (int y = 0; y < N; ++y)
    {
        for (int x = 0; x < N; ++x)
        {
            double g = 40.0 + 120.0 * (double)x / (N - 1) + 80.0 * (double)y / (N - 1);
            img.at(x, y) = g;
        }
    }

    // big bright rectangle
    int r0 = N / 4, r1 = 3 * N / 4;
    for (int y = r0; y < r1; ++y)
    {
        for (int x = r0; x < r1; ++x)
            img.at(x, y) = 220.0;
    }

    // ring
    double R = N * 0.28;
    for (int y = 0; y < N; ++y)
    {
        for (int x = 0; x < N; ++x)
        {
            double dx = x - cx, dy = y - cy;
            double d = std::sqrt(dx * dx + dy * dy);
            if (std::abs(d - R) < 2.0)
                img.at(x, y) = 30.0;
        }
    }

    // diagonal line
    for (int i = 0; i < N; ++i)
    {
        int x = i, y = (int)(0.6 * i);
        if (y >= 0 && y < N)
            img.at(x, y) = 10.0;
    }

    return img;
}

Image make_checkerboard(int N, int block)
{
    Image img(N, N);
    for (int y = 0; y < N; ++y)
    {
        for (int x = 0; x < N; ++x)
        {
            int bx = x / block, by = y / block;
            img.at(x, y) = ((bx + by) & 1) ? 220.0 : 30.0;
        }
    }
    return img;
}

Image make_circles(int N)
{
    Image img(N, N);
    int cx = N / 2, cy = N / 2;
    for (int y = 0; y < N; ++y)
    {
        for (int x = 0; x < N; ++x)
        {
            double dx = x - cx, dy = y - cy;
            double d = std::sqrt(dx * dx + dy * dy);
            double v = 128.0 + 80.0 * std::sin(2.0 * PI * d / (N * 0.18));
            img.at(x, y) = clamp255(v);
        }
    }
    return img;
}

Image add_periodic_noise_multi(const Image &src, double amp,
                               const std::vector<std::pair<int, int>> &ks)
{
    Image out(src.w, src.h);
    int N = src.w;
    for (int y = 0; y < N; ++y)
    {
        for (int x = 0; x < N; ++x)
        {
            double n = 0.0;
            for (auto [kx, ky] : ks)
            {
                n += std::sin(2.0 * PI * ((double)kx * x / N + (double)ky * y / N));
            }
            n = amp * n / std::max(1.0, (double)ks.size());
            out.at(x, y) = clamp255(src.at(x, y) + n);
        }
    }
    return out;
}

// -------------------- Frequency masks --------------------

// -----------------------------------------------------------------------------
// Generate a 2D Gaussian Low-Pass Filter (GLPF) in frequency domain.
//
// H(u,v) = exp( - D(u,v)^2 / (2*sigma^2) )
//
// Where:
//   D(u,v) = distance from frequency point (u,v) to the center of spectrum
//   sigma  = standard deviation controlling blur strength
//
// Notes:
//   - Smaller sigma  -> stronger blur (narrow passband)
//   - Larger sigma   -> weaker blur (wide passband)
//   - Assumes spectrum is already centered (DC at N/2, N/2)
// -----------------------------------------------------------------------------
std::vector<double> gaussian_lowpass(int N, double sigma)
{
    // Allocate N x N frequency mask (row-major layout)
    std::vector<double> H((size_t)N * N, 0.0);

    // Center of frequency domain (DC component location)
    int cx = N / 2;
    int cy = N / 2;

    // Helper to convert 2D index (x,y) to 1D index
    auto id = [&](int x, int y)
    {
        return (size_t)y * N + x;
    };

    // Precompute denominator term: 2*sigma^2
    // Add small epsilon to avoid division by zero if sigma is extremely small
    double two_sigma2 = 2.0 * sigma * sigma + 1e-12;

    // Loop over all frequency coordinates
    for (int y = 0; y < N; ++y)
    {
        for (int x = 0; x < N; ++x)
        {
            // Distance from current frequency point to spectrum center
            double du = x - cx;
            double dv = y - cy;

            // Squared distance
            double d2 = du * du + dv * dv;

            // Gaussian response
            H[id(x, y)] = std::exp(-d2 / two_sigma2);
        }
    }

    return H;
}

std::vector<double> gaussian_highpass(int N, double sigma)
{
    auto glp = gaussian_lowpass(N, sigma);
    for (auto &v : glp)
        v = 1.0 - v;
    return glp;
}

std::vector<double> ideal_bandpass(int N, double r0, double r1)
{
    std::vector<double> H((size_t)N * N, 0.0);
    int cx = N / 2, cy = N / 2;
    auto id = [&](int x, int y)
    { return (size_t)y * N + x; };
    for (int y = 0; y < N; ++y)
    {
        for (int x = 0; x < N; ++x)
        {
            double du = x - cx, dv = y - cy;
            double d = std::sqrt(du * du + dv * dv);
            H[id(x, y)] = (d >= r0 && d <= r1) ? 1.0 : 0.0;
        }
    }
    return H;
}

std::vector<double> ideal_bandstop(int N, double r0, double r1)
{
    auto bp = ideal_bandpass(N, r0, r1);
    for (auto &v : bp)
        v = 1.0 - v;
    return bp;
}

std::vector<cd> apply_mask(const std::vector<cd> &F, const std::vector<double> &H)
{
    std::vector<cd> G(F.size());
    for (size_t i = 0; i < F.size(); ++i)
        G[i] = F[i] * H[i];
    return G;
}

// Auto notch: detect large spikes outside center region and zero out small disks around them
std::vector<double> auto_notch_mask(const std::vector<cd> &F, int N,
                                    double center_keep = 24.0,
                                    double threshold_factor = 12.0,
                                    int notch_radius = 3)
{
    std::vector<double> H((size_t)N * N, 1.0);
    int cx = N / 2, cy = N / 2;
    auto id = [&](int x, int y)
    { return (size_t)y * N + x; };

    std::vector<double> mags;
    mags.reserve((size_t)N * N);
    for (const auto &v : F)
        mags.push_back(std::abs(v));
    std::nth_element(mags.begin(), mags.begin() + mags.size() / 2, mags.end());
    double med = mags[mags.size() / 2] + 1e-12;

    for (int y = 0; y < N; ++y)
    {
        for (int x = 0; x < N; ++x)
        {
            double du = x - cx, dv = y - cy;
            double d = std::sqrt(du * du + dv * dv);
            if (d <= center_keep)
                continue;

            double m = std::abs(F[id(x, y)]);
            if (m > threshold_factor * med)
            {
                for (int yy = y - notch_radius; yy <= y + notch_radius; ++yy)
                {
                    for (int xx = x - notch_radius; xx <= x + notch_radius; ++xx)
                    {
                        if (xx < 0 || xx >= N || yy < 0 || yy >= N)
                            continue;
                        double rr = std::sqrt((xx - x) * (xx - x) + (yy - y) * (yy - y));
                        if (rr <= notch_radius)
                            H[id(xx, yy)] = 0.0;
                    }
                }
            }
        }
    }
    return H;
}

// -------------------- Metrics (hash + error stats) --------------------
static inline uint64_t fnv1a64_update(uint64_t h, const void *data, size_t n)
{
    const uint8_t *p = (const uint8_t *)data;
    const uint64_t FNV_PRIME = 1099511628211ULL;
    for (size_t i = 0; i < n; ++i)
    {
        h ^= (uint64_t)p[i];
        h *= FNV_PRIME;
    }
    return h;
}

uint64_t hash_image_bytes_u8(const Image &img)
{
    uint64_t h = 14695981039346656037ULL;
    for (double v : img.px)
    {
        uint8_t b = (uint8_t)std::lround(clamp255(v));
        h = fnv1a64_update(h, &b, 1);
    }
    return h;
}

uint64_t hash_image_f64(const Image &img)
{
    uint64_t h = 14695981039346656037ULL;
    h = fnv1a64_update(h, img.px.data(), img.px.size() * sizeof(double));
    return h;
}

struct ErrorStats
{
    double l2 = 0.0;
    double rmse = 0.0;
    double max_abs = 0.0;
    double psnr = 0.0;
};

ErrorStats compare_images(const Image &a, const Image &b)
{
    if (a.w != b.w || a.h != b.h)
    {
        std::cerr << "compare_images: size mismatch\n";
        std::exit(1);
    }
    long double sse = 0.0L;
    double maxabs = 0.0;
    size_t n = a.px.size();
    for (size_t i = 0; i < n; ++i)
    {
        double d = a.px[i] - b.px[i];
        sse += (long double)d * (long double)d;
        maxabs = std::max(maxabs, std::abs(d));
    }
    double sse_d = (double)sse;
    ErrorStats st;
    st.l2 = std::sqrt(sse_d);
    st.rmse = std::sqrt(sse_d / (double)n);
    st.max_abs = maxabs;

    double mse = sse_d / (double)n;
    if (mse < 1e-20)
        st.psnr = std::numeric_limits<double>::infinity();
    else
        st.psnr = 10.0 * std::log10((255.0 * 255.0) / mse);
    return st;
}

struct MetricsRow
{
    std::string test;
    int N;
    std::string name;
    std::string ref_name; // empty if none
    uint64_t hash_u8 = 0;
    uint64_t hash_f64 = 0;
    double l2 = 0.0, rmse = 0.0, max_abs = 0.0, psnr = 0.0;
};

void write_metrics_csv(const std::string &path, const std::vector<MetricsRow> &rows)
{
    std::ofstream out(path);
    out << "test,N,name,ref,hash_u8,hash_f64,l2,rmse,max_abs,psnr\n";
    for (const auto &r : rows)
    {
        out << r.test << "," << r.N << "," << r.name << "," << r.ref_name << ","
            << r.hash_u8 << "," << r.hash_f64 << ","
            << r.l2 << "," << r.rmse << "," << r.max_abs << "," << r.psnr << "\n";
    }
}

// -------------------- Performance --------------------
struct PerfRow
{
    std::string test;
    int N;
    std::string stage;
    double ms;
};

void write_perf_csv(const std::string &path, const std::vector<PerfRow> &rows)
{
    std::ofstream out(path);
    out << "test,N,stage,ms\n";
    for (const auto &r : rows)
        out << r.test << "," << r.N << "," << r.stage << "," << r.ms << "\n";
}

// -------------------- Main --------------------
int main(int argc, char **argv)
{
    int N = 1024;
    int runs = 3;
    std::string outdir = ".";

    if (argc >= 2)
        N = std::stoi(argv[1]);
    if (argc >= 3)
        runs = std::stoi(argv[2]);
    if (argc >= 4)
        outdir = argv[3];

    if (N <= 0 || (N & 1))
    {
        std::cerr << "N must be positive and even (recommended power of 2).\n";
        return 1;
    }

    std::cout << "CPU DFT benchmark N=" << N << " runs=" << runs << " outdir=" << outdir << "\n";
    Twiddle1D tw(N);

    std::vector<PerfRow> perf;
    std::vector<MetricsRow> metrics;

    auto save = [&](const std::string &fname, const Image &img)
    {
        std::string path = outdir + "/" + fname;
        if (!write_pgm(path, img))
            std::cerr << "Failed writing: " << path << "\n";
    };

    auto record_metrics = [&](const std::string &test, const std::string &name,
                              const Image &img, const std::string &ref_name,
                              const Image *ref_img)
    {
        MetricsRow r;
        r.test = test;
        r.N = N;
        r.name = name;
        r.ref_name = ref_name;
        r.hash_u8 = hash_image_bytes_u8(img);
        r.hash_f64 = hash_image_f64(img);
        if (ref_img)
        {
            auto st = compare_images(img, *ref_img);
            r.l2 = st.l2;
            r.rmse = st.rmse;
            r.max_abs = st.max_abs;
            r.psnr = st.psnr;
        }
        metrics.push_back(r);
    };

    // Helper: save spectrum image for a given complex spectrum (already centered)
    auto save_spectrum_from_F = [&](const std::string &fname, const std::vector<cd> &F)
    {
        Image spec = spectrum_to_image(F, N, N, /*shift_center=*/false);
        save(fname, spec);
        // metrics for spectrum (no reference)
        record_metrics("SPECTRUM", fname, spec, "", nullptr);
    };

    // Helper: save spectrum for an Image (computes DFT internally; NOT included in perf timing)
    auto save_spectrum_from_img = [&](const std::string &test, const std::string &tag, const Image &img)
    {
        std::vector<cd> F = dft2(img, tw, true);
        Image spec = spectrum_to_image(F, N, N, /*shift_center=*/false);
        std::string fname = test + "_" + tag + "_spectrum.pgm";
        save(fname, spec);
        record_metrics(test, fname, spec, "", nullptr);
    };

    // ---------------- A: Gaussian low-pass blur ----------------
    {
        std::string test = "A_gauss_lowpass_blur";
        Image src = make_scene_mix(N);
        save(test + "_0_src.pgm", src);
        record_metrics(test, test + "_0_src", src, "", nullptr);
        save_spectrum_from_img(test, "0_src", src);

        std::vector<cd> F;
        double dft_ms = avg_ms(runs, [&]
                               { F = dft2(src, tw, true); });
        perf.push_back({test, N, "DFT2", dft_ms});
        // Also save spectrum from already-computed F (same as src spectrum)
        save_spectrum_from_F(test + "_0_src_spectrum_fromF.pgm", F);

        auto H = gaussian_lowpass(N, (double)N * 0.03);
        std::vector<cd> G;
        double mask_ms = avg_ms(runs, [&]
                                { G = apply_mask(F, H); });
        perf.push_back({test, N, "Mask_gaussian_lowpass", mask_ms});

        Image blur;
        double idft_ms = avg_ms(runs, [&]
                                { blur = idft2(G, tw, N, true); });
        perf.push_back({test, N, "IDFT2", idft_ms});

        save(test + "_1_blur.pgm", blur);
        record_metrics(test, test + "_1_blur", blur, test + "_0_src", &src);
        save_spectrum_from_img(test, "1_blur", blur);
    }

    // ---------------- B: Gaussian high-pass edges ----------------
    {
        std::string test = "B_gauss_highpass_edges";
        Image src = make_scene_mix(N);
        save(test + "_0_src.pgm", src);
        record_metrics(test, test + "_0_src", src, "", nullptr);
        save_spectrum_from_img(test, "0_src", src);

        std::vector<cd> F;
        double dft_ms = avg_ms(runs, [&]
                               { F = dft2(src, tw, true); });
        perf.push_back({test, N, "DFT2", dft_ms});
        save_spectrum_from_F(test + "_0_src_spectrum_fromF.pgm", F);

        auto H = gaussian_highpass(N, (double)N * 0.02);
        std::vector<cd> G;
        double mask_ms = avg_ms(runs, [&]
                                { G = apply_mask(F, H); });
        perf.push_back({test, N, "Mask_gaussian_highpass", mask_ms});

        Image edges;
        double idft_ms = avg_ms(runs, [&]
                                { edges = idft2(G, tw, N, true); });
        perf.push_back({test, N, "IDFT2", idft_ms});

        Image edges_vis = normalize_to_0_255(edges);
        save(test + "_1_edges_norm.pgm", edges_vis);
        record_metrics(test, test + "_1_edges_norm", edges_vis, "", nullptr);
        save_spectrum_from_img(test, "1_edges_norm", edges_vis);
    }

    // ---------------- C: Band-pass texture emphasis ----------------
    {
        std::string test = "C_bandpass_texture";
        Image src = make_scene_mix(N);
        save(test + "_0_src.pgm", src);
        record_metrics(test, test + "_0_src", src, "", nullptr);
        save_spectrum_from_img(test, "0_src", src);

        std::vector<cd> F;
        double dft_ms = avg_ms(runs, [&]
                               { F = dft2(src, tw, true); });
        perf.push_back({test, N, "DFT2", dft_ms});
        save_spectrum_from_F(test + "_0_src_spectrum_fromF.pgm", F);

        auto H = ideal_bandpass(N, (double)N * 0.06, (double)N * 0.15);
        std::vector<cd> G;
        double mask_ms = avg_ms(runs, [&]
                                { G = apply_mask(F, H); });
        perf.push_back({test, N, "Mask_bandpass_ideal", mask_ms});

        Image band;
        double idft_ms = avg_ms(runs, [&]
                                { band = idft2(G, tw, N, true); });
        perf.push_back({test, N, "IDFT2", idft_ms});

        Image band_vis = normalize_to_0_255(band);
        save(test + "_1_bandpass_norm.pgm", band_vis);
        record_metrics(test, test + "_1_bandpass_norm", band_vis, "", nullptr);
        save_spectrum_from_img(test, "1_bandpass_norm", band_vis);
    }

    // ---------------- D: Checkerboard stress ----------------
    {
        std::string test = "D_checkerboard";

        // Define block size (controls checkerboard frequency)
        int block = std::max(4, N / 64);

        // Generate checkerboard test image (high-frequency stress pattern)
        Image src = make_checkerboard(N, block);

        // Save original image
        save(test + "_0_src.pgm", src);

        // Record metrics for the original image (hash, stats, etc.)
        record_metrics(test, test + "_0_src", src, "", nullptr);

        // Save magnitude spectrum computed directly from the image
        save_spectrum_from_img(test, "0_src", src);

        std::vector<cd> F;

        // Measure average runtime of 2D DFT (with frequency shift enabled)
        double dft_ms = avg_ms(runs, [&]
                               { F = dft2(src, tw, true); });

        // Store DFT performance result
        perf.push_back({test, N, "DFT2", dft_ms});

        // Save spectrum directly from complex frequency result
        save_spectrum_from_F(test + "_0_src_spectrum_fromF.pgm", F);

        // Create Gaussian low-pass filter mask (keeps low frequencies, removes high)
        auto H = gaussian_lowpass(N, (double)N * 0.03);

        std::vector<cd> G;

        // Apply frequency-domain mask and measure runtime
        double mask_ms = avg_ms(runs, [&]
                                { G = apply_mask(F, H); });

        // Store masking performance result
        perf.push_back({test, N, "Mask_gaussian_lowpass", mask_ms});

        Image out;

        // Perform inverse DFT to reconstruct filtered image
        double idft_ms = avg_ms(runs, [&]
                                { out = idft2(G, tw, N, true); });

        // Store IDFT performance result
        perf.push_back({test, N, "IDFT2", idft_ms});

        // Save filtered (low-pass) result image
        save(test + "_1_lowpass_gaussian.pgm", out);

        // Record metrics comparing output to original image
        record_metrics(test, test + "_1_lowpass_gaussian", out, test + "_0_src", &src);

        // Save spectrum of the filtered image
        save_spectrum_from_img(test, "1_lowpass_gaussian", out);
    }

    // ---------------- E: Periodic noise -> auto notch denoise (auto only) ----------------
    {
        std::string test = "E_periodic_noise_auto_notch";
        Image base = make_scene_mix(N);
        save(test + "_0_base_clean.pgm", base);
        record_metrics(test, test + "_0_base_clean", base, "", nullptr);
        save_spectrum_from_img(test, "0_base_clean", base);

        std::vector<std::pair<int, int>> ks = {{12, 0}, {0, 18}, {22, 14}};
        Image noisy = add_periodic_noise_multi(base, 45.0, ks);
        save(test + "_1_noisy.pgm", noisy);
        record_metrics(test, test + "_1_noisy", noisy, test + "_0_base_clean", &base);
        save_spectrum_from_img(test, "1_noisy", noisy);

        std::vector<cd> F;
        double dft_ms = avg_ms(runs, [&]
                               { F = dft2(noisy, tw, true); });
        perf.push_back({test, N, "DFT2", dft_ms});
        save_spectrum_from_F(test + "_1_noisy_spectrum_fromF.pgm", F);

        std::vector<double> Hnotch;
        double notch_build_ms = avg_ms(runs, [&]
                                       { Hnotch = auto_notch_mask(F, N, 28.0, 14.0, 3); });
        perf.push_back({test, N, "Build_auto_notch", notch_build_ms});

        std::vector<cd> G;
        double mask_ms = avg_ms(runs, [&]
                                { G = apply_mask(F, Hnotch); });
        perf.push_back({test, N, "Mask_auto_notch", mask_ms});

        Image den;
        double idft_ms = avg_ms(runs, [&]
                                { den = idft2(G, tw, N, true); });
        perf.push_back({test, N, "IDFT2", idft_ms});

        save(test + "_2_denoised_auto_notch.pgm", den);
        record_metrics(test, test + "_2_denoised_auto_notch", den, test + "_0_base_clean", &base);
        save_spectrum_from_img(test, "2_denoised_auto_notch", den);
    }

    // ---------------- F: Radial rings -> band-stop ----------------
    {
        std::string test = "F_radial_rings_bandstop";
        Image src = make_circles(N);
        save(test + "_0_src.pgm", src);
        record_metrics(test, test + "_0_src", src, "", nullptr);
        save_spectrum_from_img(test, "0_src", src);

        std::vector<cd> F;
        double dft_ms = avg_ms(runs, [&]
                               { F = dft2(src, tw, true); });
        perf.push_back({test, N, "DFT2", dft_ms});
        save_spectrum_from_F(test + "_0_src_spectrum_fromF.pgm", F);

        auto H = ideal_bandstop(N, (double)N * 0.08, (double)N * 0.14);
        std::vector<cd> G;
        double mask_ms = avg_ms(runs, [&]
                                { G = apply_mask(F, H); });
        perf.push_back({test, N, "Mask_bandstop_ideal", mask_ms});

        Image out;
        double idft_ms = avg_ms(runs, [&]
                                { out = idft2(G, tw, N, true); });
        perf.push_back({test, N, "IDFT2", idft_ms});

        save(test + "_1_bandstop_removed_rings.pgm", out);
        record_metrics(test, test + "_1_bandstop_removed_rings", out, test + "_0_src", &src);
        save_spectrum_from_img(test, "1_bandstop_removed_rings", out);
    }

    // Write CSVs
    write_perf_csv(outdir + "/performance.csv", perf);
    write_metrics_csv(outdir + "/metrics.csv", metrics);

    std::cout << "Done. Wrote PGM images + spectrum PGM images + performance.csv + metrics.csv\n";
    return 0;
}
