// -----------------------------------------------------------------------------
// Common CUDA optimizations used:
//
// (1) Transpose to avoid strided column access
//     - 2D DFT is implemented as: RowDFT -> Transpose -> RowDFT -> Transpose back
//     - Why: Doing a "column DFT" directly would read/write memory with a stride of N,
//            which is usually not coalesced (slow global memory). Transpose turns
//            columns into rows so the second pass is also row-coalesced.
//
// (2) Tiled shared-memory transpose (classic 32x32 tile + padding)
//     - Uses shared memory tile[TILE_DIM][TILE_DIM+1]
//     - Why: shared memory staging makes global reads/writes coalesced for transpose,
//            and the +1 padding reduces shared-memory bank conflicts.
//
// (3) On-the-fly twiddle (sincos) + simple recurrence in the row DFT
//     - Each output (row y, frequency k) computes w_step = exp(sign*i*2πk/N) once,
//       then iterates w *= w_step across x.
//     - Why: avoids a large precomputed twiddle table (bandwidth heavy).
//            This is a common tradeoff: slightly more math, less memory traffic.
//
// (4) __restrict__ pointers
//     - Mark kernel pointers __restrict__ to help compiler assume no aliasing.
//     - Why: enables better scheduling and sometimes fewer redundant loads.
//
// (5) Fuse scale + inverse-shift into final output kernel
//     - Output kernel does: real-part extraction + (1/(N*N)) scaling + (-1)^(x+y)
//     - Why: reduces extra kernel passes (less launch overhead, less global traffic).
//
// Not used (intentionally, to keep it "common" / not over-optimized):
//   - No shared-memory caching of full row data for DFT (would be heavier optimization)
//   - No warp shuffles / warp reductions
//   - No pinned host memory
//   - No multi-stream overlap
//   - No cuFFT / no OpenCV
//
// Build:
//   nvcc -O3 -std=c++17 -arch=sm_120 dft_filters_gpu.cu -o dft_filters_gpu
//
// Run:
//   ./dft_filter_gpu outdir 3 2048
//     outdir: output folder
//     runs: optional, default 3
//     N: optional, image size (NxN). default 2048
//
// Output files & names follow the CPU baseline pattern.
//
// Notes about exactness:
// - Math matches the CPU baseline DFT/IDFT definitions.
// - Using recurrence twiddle may introduce tiny floating differences vs CPU table.
// - Metrics (L2/RMSE/PSNR) should still be close for validation.
//
// Size: N is runtime-configurable (default 2048).

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <complex>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <limits>

using cd = std::complex<double>;
static constexpr double PI = 3.14159265358979323846;

// -------------------- CUDA helpers --------------------
#define CUDA_CHECK(call) do {                                  \
    cudaError_t e = (call);                                    \
    if (e != cudaSuccess) {                                    \
        std::cerr << "CUDA error: " << cudaGetErrorString(e)   \
                  << " at " << __FILE__ << ":" << __LINE__     \
                  << "\n";                                     \
        std::exit(1);                                          \
    }                                                          \
} while(0)

struct CudaTimer {
    cudaEvent_t a{}, b{};
    CudaTimer() { CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b)); }
    ~CudaTimer(){ cudaEventDestroy(a); cudaEventDestroy(b); }
    void tic(cudaStream_t s=0){ CUDA_CHECK(cudaEventRecord(a, s)); }
    double toc_ms(cudaStream_t s=0){
        CUDA_CHECK(cudaEventRecord(b, s));
        CUDA_CHECK(cudaEventSynchronize(b));
        float ms=0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
        return (double)ms;
    }
};

template<typename Fn>
double avg_ms_cuda(int runs, Fn fn, cudaStream_t stream=0) {
    CudaTimer t;
    double total = 0.0;
    for(int i=0;i<runs;++i){
        t.tic(stream);
        fn();
        total += t.toc_ms(stream);
    }
    return total / std::max(1, runs);
}

// -------------------- Image (CPU) --------------------
struct Image {
    int w, h;
    std::vector<double> px; // grayscale double
    Image() : w(0), h(0) {}
    Image(int W, int H) : w(W), h(H), px((size_t)W*H, 0.0) {}
    inline double& at(int x, int y) { return px[(size_t)y*w + x]; }
    inline double  at(int x, int y) const { return px[(size_t)y*w + x]; }
};

static inline double clamp255(double v) {
    if (v < 0.0) return 0.0;
    if (v > 255.0) return 255.0;
    return v;
}

bool write_pgm(const std::string& path, const Image& img) {
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    out << "P5\n" << img.w << " " << img.h << "\n255\n";
    for (int y = 0; y < img.h; ++y) {
        for (int x = 0; x < img.w; ++x) {
            uint8_t v = (uint8_t)std::lround(clamp255(img.at(x,y)));
            out.write((const char*)&v, 1);
        }
    }
    return true;
}

Image normalize_to_0_255(const Image& in) {
    Image out(in.w, in.h);
    double mn = 1e300, mx = -1e300;
    for (double v : in.px) { mn = std::min(mn, v); mx = std::max(mx, v); }
    double den = (mx - mn) + 1e-12;
    for (size_t i=0;i<in.px.size();++i) out.px[i] = (in.px[i] - mn) / den * 255.0;
    return out;
}

// -------------------- Spectrum visualization (CPU) --------------------
// Same as baseline: log(1+|F|)->normalize 0..255.
// shift_center=false because shift was applied in DFT, so DC is already centered in F.
Image spectrum_to_image(const std::vector<cd>& F, int w, int h, bool shift_center) {
    std::vector<double> mag((size_t)w*h, 0.0);
    auto idx = [&](int x,int y){ return (size_t)y*w + x; };

    double maxv = 1e-12;
    for (int y=0;y<h;++y){
        for (int x=0;x<w;++x){
            int sx=x, sy=y;
            if (shift_center) { sx = (x + w/2) % w; sy = (y + h/2) % h; }
            double m = std::abs(F[idx(sx,sy)]);
            double v = std::log(1.0 + m);
            mag[idx(x,y)] = v;
            maxv = std::max(maxv, v);
        }
    }
    Image out(w,h);
    for (size_t i=0;i<mag.size();++i) out.px[i] = (mag[i]/maxv)*255.0;
    return out;
}

// -------------------- Synthetic inputs (CPU) --------------------
Image make_scene_mix(int n) {
    Image img(n,n);
    int cx=n/2, cy=n/2;

    for(int y=0;y<n;++y){
        for(int x=0;x<n;++x){
            double g = 40.0 + 120.0*(double)x/(n-1) + 80.0*(double)y/(n-1);
            img.at(x,y) = g;
        }
    }
    int r0 = n/4, r1 = 3*n/4;
    for(int y=r0;y<r1;++y){
        for(int x=r0;x<r1;++x) img.at(x,y) = 220.0;
    }
    double R = n*0.28;
    for(int y=0;y<n;++y){
        for(int x=0;x<n;++x){
            double dx=x-cx, dy=y-cy;
            double d=std::sqrt(dx*dx+dy*dy);
            if (std::abs(d - R) < 2.0) img.at(x,y) = 30.0;
        }
    }
    for(int i=0;i<n;++i){
        int x=i, y=(int)(0.6*i);
        if (y>=0 && y<n) img.at(x,y)=10.0;
    }
    return img;
}

Image make_checkerboard(int n, int block) {
    Image img(n,n);
    for(int y=0;y<n;++y){
        for(int x=0;x<n;++x){
            int bx = x/block, by = y/block;
            img.at(x,y) = ((bx+by)&1) ? 220.0 : 30.0;
        }
    }
    return img;
}

Image make_circles(int n) {
    Image img(n,n);
    int cx=n/2, cy=n/2;
    for(int y=0;y<n;++y){
        for(int x=0;x<n;++x){
            double dx=x-cx, dy=y-cy;
            double d=std::sqrt(dx*dx+dy*dy);
            double v = 128.0 + 80.0*std::sin(2.0*PI*d/(n*0.18));
            img.at(x,y) = clamp255(v);
        }
    }
    return img;
}

Image add_periodic_noise_multi(const Image& src, double amp,
                               const std::vector<std::pair<int,int>>& ks) {
    Image out(src.w, src.h);
    int n=src.w;
    for(int y=0;y<n;++y){
        for(int x=0;x<n;++x){
            double nn=0.0;
            for(auto [kx,ky]: ks) {
                nn += std::sin(2.0*PI*((double)kx*x/n + (double)ky*y/n));
            }
            nn = amp * nn / std::max(1.0, (double)ks.size());
            out.at(x,y) = clamp255(src.at(x,y) + nn);
        }
    }
    return out;
}

// -------------------- Frequency masks (CPU build; GPU apply) --------------------
std::vector<double> gaussian_lowpass(int n, double sigma)
{
    std::vector<double> H((size_t)n*n, 0.0);
    int cx = n / 2;
    int cy = n / 2;
    auto id = [&](int x, int y) { return (size_t)y * n + x; };
    double two_sigma2 = 2.0 * sigma * sigma + 1e-12;

    for (int y = 0; y < n; ++y){
        for (int x = 0; x < n; ++x){
            double du = x - cx;
            double dv = y - cy;
            double d2 = du * du + dv * dv;
            H[id(x, y)] = std::exp(-d2 / two_sigma2);
        }
    }
    return H;
}

std::vector<double> gaussian_highpass(int n, double sigma) {
    auto glp = gaussian_lowpass(n, sigma);
    for(auto& v : glp) v = 1.0 - v;
    return glp;
}

std::vector<double> ideal_bandpass(int n, double r0, double r1) {
    std::vector<double> H((size_t)n*n, 0.0);
    int cx=n/2, cy=n/2;
    auto id = [&](int x,int y){ return (size_t)y*n + x; };
    for(int y=0;y<n;++y){
        for(int x=0;x<n;++x){
            double du=x-cx, dv=y-cy;
            double d=std::sqrt(du*du+dv*dv);
            H[id(x,y)] = (d>=r0 && d<=r1) ? 1.0 : 0.0;
        }
    }
    return H;
}

std::vector<double> ideal_bandstop(int n, double r0, double r1) {
    auto bp = ideal_bandpass(n, r0, r1);
    for(auto& v : bp) v = 1.0 - v;
    return bp;
}

// Auto notch mask built on CPU (same as baseline logic)
std::vector<double> auto_notch_mask(const std::vector<cd>& F, int n,
                                    double center_keep=24.0,
                                    double threshold_factor=12.0,
                                    int notch_radius=3) {
    std::vector<double> H((size_t)n*n, 1.0);
    int cx=n/2, cy=n/2;
    auto id = [&](int x,int y){ return (size_t)y*n + x; };

    std::vector<double> mags;
    mags.reserve((size_t)n*n);
    for(const auto& v : F) mags.push_back(std::abs(v));
    std::nth_element(mags.begin(), mags.begin()+mags.size()/2, mags.end());
    double med = mags[mags.size()/2] + 1e-12;

    for(int y=0;y<n;++y){
        for(int x=0;x<n;++x){
            double du=x-cx, dv=y-cy;
            double d=std::sqrt(du*du+dv*dv);
            if (d <= center_keep) continue;

            double m = std::abs(F[id(x,y)]);
            if (m > threshold_factor * med) {
                for(int yy=y-notch_radius; yy<=y+notch_radius; ++yy){
                    for(int xx=x-notch_radius; xx<=x+notch_radius; ++xx){
                        if(xx<0||xx>=n||yy<0||yy>=n) continue;
                        double rr = std::sqrt((xx-x)*(xx-x) + (yy-y)*(yy-y));
                        if (rr <= notch_radius) H[id(xx,yy)] = 0.0;
                    }
                }
            }
        }
    }
    return H;
}

// -------------------- Metrics (CPU) --------------------
static inline uint64_t fnv1a64_update(uint64_t h, const void* data, size_t n) {
    const uint8_t* p = (const uint8_t*)data;
    const uint64_t FNV_PRIME = 1099511628211ULL;
    for (size_t i=0;i<n;++i) {
        h ^= (uint64_t)p[i];
        h *= FNV_PRIME;
    }
    return h;
}

uint64_t hash_image_bytes_u8(const Image& img) {
    uint64_t h = 14695981039346656037ULL;
    for (double v : img.px) {
        uint8_t b = (uint8_t)std::lround(clamp255(v));
        h = fnv1a64_update(h, &b, 1);
    }
    return h;
}

uint64_t hash_image_f64(const Image& img) {
    uint64_t h = 14695981039346656037ULL;
    h = fnv1a64_update(h, img.px.data(), img.px.size() * sizeof(double));
    return h;
}

struct ErrorStats {
    double l2 = 0.0;
    double rmse = 0.0;
    double max_abs = 0.0;
    double psnr = 0.0;
};

ErrorStats compare_images(const Image& a, const Image& b) {
    if (a.w != b.w || a.h != b.h) {
        std::cerr << "compare_images: size mismatch\n";
        std::exit(1);
    }
    long double sse = 0.0L;
    double maxabs = 0.0;
    size_t n = a.px.size();
    for (size_t i=0;i<n;++i) {
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
    if (mse < 1e-20) st.psnr = std::numeric_limits<double>::infinity();
    else st.psnr = 10.0 * std::log10((255.0*255.0) / mse);
    return st;
}

// (Kept for compatibility; this file uses MetricsRow2 below)
struct MetricsRow {
    std::string test;
    int N;
    std::string name;
    std::string ref_name;
    uint64_t hash_u8 = 0;
    uint64_t hash_f64 = 0;
    double l2 = 0.0, rmse = 0.0, max_abs = 0.0, psnr = 0.0;
};

void write_metrics_csv(const std::string& path, const std::vector<MetricsRow>& rows) {
    std::ofstream out(path);
    out << "test,N,name,ref,hash_u8,hash_f64,l2,rmse,max_abs,psnr\n";
    for (const auto& r: rows) {
        out << r.test << "," << r.N << "," << r.name << "," << r.ref_name << ","
            << r.hash_u8 << "," << r.hash_f64 << ","
            << r.l2 << "," << r.rmse << "," << r.max_abs << "," << r.psnr << "\n";
    }
}

struct PerfRow {
    std::string test;
    int N;
    std::string stage;
    double ms;
};

void write_perf_csv(const std::string& path, const std::vector<PerfRow>& rows) {
    std::ofstream out(path);
    out << "test,N,stage,ms\n";
    for (const auto& r: rows) out << r.test << "," << r.N << "," << r.stage << "," << r.ms << "\n";
}

// -------------------- GPU complex --------------------
struct cdouble2 { double x, y; }; // x=real, y=imag
__device__ __forceinline__ cdouble2 cadd(cdouble2 a, cdouble2 b){ return {a.x+b.x, a.y+b.y}; }
__device__ __forceinline__ cdouble2 cmul(cdouble2 a, cdouble2 b){
    return {a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x};
}

// -------------------- Kernels --------------------

// (Common micro-optimization #4: __restrict__ pointers)
// __restrict__ helps the compiler assume img/out do not alias, improving load/store scheduling.
//
// Convert input image -> complex, applying shift (-1)^(x+y).
// Shift is a standard DFT trick to center DC (low frequencies) in the spectrum image.
// This is NOT a "performance optimization"; it matches the baseline math/visualization.
__global__ void img_to_complex_shift(const double* __restrict__ img,
                                     cdouble2* __restrict__ out,
                                     int n, int shift_on)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x>=n || y>=n) return;
    int idx = y*n + x;
    double f = img[idx];
    if (shift_on && ((x+y)&1)) f = -f;
    out[idx] = {f, 0.0};
}

// (Common optimization #2: tiled shared-memory transpose + padding)
//
// Why this transpose is fast (common pattern):
// - Global reads are coalesced: threads read consecutive elements from a row.
// - Data is staged into shared memory (tile).
// - Global writes are coalesced: threads write consecutive elements to the output row.
// - tile[TILE_DIM][TILE_DIM+1] uses +1 padding to reduce shared-memory bank conflicts
//   when threads access tile transposed (classic CUDA transpose trick).
template<int TILE_DIM, int BLOCK_ROWS>
__global__ void transpose_cdouble2(const cdouble2* __restrict__ in,
                                   cdouble2* __restrict__ out,
                                   int n)
{
    __shared__ cdouble2 tile[TILE_DIM][TILE_DIM+1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Load tile from global -> shared (coalesced global reads)
    for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS){
        int yy = y + i;
        if (x < n && yy < n) tile[threadIdx.y + i][threadIdx.x] = in[yy*n + x];
    }
    __syncthreads();

    // Compute transposed output coordinates
    int ox = blockIdx.y * TILE_DIM + threadIdx.x;
    int oy = blockIdx.x * TILE_DIM + threadIdx.y;

    // Store tile from shared -> global (coalesced global writes)
    for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS){
        int oyy = oy + i;
        if (ox < n && oyy < n) out[oyy*n + ox] = tile[threadIdx.x][threadIdx.y + i];
    }
}

// (Common optimization #3: on-the-fly twiddle via sincos + recurrence)
//
// Row DFT using twiddle recurrence:
// - Each thread computes ONE output coefficient: out[y, k]
// - w_step = exp(sign*i*2πk/N) computed once with sincos()
// - w starts at 1 and updates each input sample: w *= w_step
//
// Why this is a "common" optimization:
// - Avoids large twiddle tables (global memory bandwidth).
// - Uses a single sincos call per (y,k) and then cheap complex multiplies.
// - Tradeoff: more FLOPs, but less memory traffic.
//
// Note: This is still O(N^3) overall for 2D DFT (not FFT). It is meant as a baseline.
__global__ void dft_rows_recurrence(const cdouble2* __restrict__ in,
                                   cdouble2* __restrict__ out,
                                   int n, int inverse)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;  // frequency index
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // row index
    if (k>=n || y>=n) return;

    double sign = inverse ? +1.0 : -1.0;
    double ang_step = sign * (2.0 * PI * (double)k / (double)n);

    // sincos() is a common minor optimization: compute sin and cos together.
    double sn, cs;
    sincos(ang_step, &sn, &cs);

    cdouble2 w_step{cs, sn}; // exp(i*ang_step) = cos + i sin
    cdouble2 w{1.0, 0.0};
    cdouble2 sum{0.0, 0.0};

    // Inputs along x are contiguous in memory for a fixed y (coalesced when many threads share y).
    // NOTE: We do NOT cache the whole row in shared memory (intentionally avoided optimization).
    int base = y*n;
    for(int x=0; x<n; ++x){
        cdouble2 a = in[base + x];
        sum = cadd(sum, cmul(a, w));
        w = cmul(w, w_step);
    }
    out[base + k] = sum;
}

// Simple elementwise mask multiply (common, straightforward).
// __restrict__ helps compiler (optimization #4).
__global__ void apply_mask_kernel(const cdouble2* __restrict__ F,
                                  const double* __restrict__ H,
                                  cdouble2* __restrict__ G,
                                  int npx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i>=npx) return;
    double h = H[i];
    G[i] = {F[i].x * h, F[i].y * h};
}

// (Common optimization #5: fused final pass)
//
// Fuse operations that would otherwise be separate passes:
// - take real part
// - scale by 1/(N*N) for inverse DFT normalization
// - undo shift (-1)^(x+y) to match spatial domain image
//
// Benefit: one kernel launch + one global write instead of multiple passes.
__global__ void complex_to_img_scale_invshift(const cdouble2* __restrict__ in,
                                              double* __restrict__ img,
                                              int n, int shift_on)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x>=n || y>=n) return;
    int idx = y*n + x;

    double inv = 1.0 / (double)(n*n);
    double f = in[idx].x * inv;         // take real part + scale
    if (shift_on && ((x+y)&1)) f = -f;  // undo shift
    img[idx] = f;
}

// -------------------- GPU Pipeline Context --------------------
struct GpuContext {
    int n = 0;
    size_t npx = 0;

    double*   d_img = nullptr;
    cdouble2* d_c0  = nullptr; // complex buffer A
    cdouble2* d_c1  = nullptr; // complex buffer B
    cdouble2* d_c2  = nullptr; // transpose temp
    double*   d_out = nullptr;
    double*   d_H   = nullptr;

    cudaStream_t stream = 0;

    void init(int n_) {
        n = n_;
        npx = (size_t)n * (size_t)n;

        CUDA_CHECK(cudaStreamCreate(&stream));

        CUDA_CHECK(cudaMalloc(&d_img, npx*sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_c0,  npx*sizeof(cdouble2)));
        CUDA_CHECK(cudaMalloc(&d_c1,  npx*sizeof(cdouble2)));
        CUDA_CHECK(cudaMalloc(&d_c2,  npx*sizeof(cdouble2)));
        CUDA_CHECK(cudaMalloc(&d_out, npx*sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_H,   npx*sizeof(double)));
    }

    void destroy() {
        if(d_H) cudaFree(d_H);
        if(d_out) cudaFree(d_out);
        if(d_c2) cudaFree(d_c2);
        if(d_c1) cudaFree(d_c1);
        if(d_c0) cudaFree(d_c0);
        if(d_img) cudaFree(d_img);
        if(stream) cudaStreamDestroy(stream);
        *this = {};
    }

    // (Common optimization #1: transpose-based 2D DFT to avoid strided column access)
    //
    // Forward 2D DFT with shift:
    //   img -> complex(shift)
    //   RowDFT forward                (coalesced rows)
    //   transpose                     (tiled transpose)
    //   RowDFT forward on transposed   (this equals column DFT but now as rows, coalesced)
    //   transpose back                (restore layout)
    //
    // Output spectrum ends in d_c0 in row-major layout [v*n + u] (same as CPU indexing).
    void dft2_forward_shift(const Image& img) {
        CUDA_CHECK(cudaMemcpyAsync(d_img, img.px.data(), npx*sizeof(double),
                                   cudaMemcpyHostToDevice, stream));

        // Convert to complex + apply shift.
        dim3 blk_img(16,16);
        dim3 grd_img((n+blk_img.x-1)/blk_img.x, (n+blk_img.y-1)/blk_img.y);
        img_to_complex_shift<<<grd_img, blk_img, 0, stream>>>(d_img, d_c0, n, 1);

        // Row DFT pass #1
        // Common choice: 16x16 block for 2D launch mapping (k across x, rows across y).
        dim3 blk_dft(16,16);
        dim3 grd_dft((n+blk_dft.x-1)/blk_dft.x, (n+blk_dft.y-1)/blk_dft.y);
        dft_rows_recurrence<<<grd_dft, blk_dft, 0, stream>>>(d_c0, d_c1, n, 0);

        // Transpose (tiled shared-memory transpose, optimization #2)
        dim3 blkT(32,8);
        dim3 grdT((n+32-1)/32, (n+32-1)/32);
        transpose_cdouble2<32,8><<<grdT, blkT, 0, stream>>>(d_c1, d_c2, n);

        // Row DFT pass #2 (operates on transposed data -> equivalent to column DFT)
        dft_rows_recurrence<<<grd_dft, blk_dft, 0, stream>>>(d_c2, d_c1, n, 0);

        // Transpose back to original layout
        transpose_cdouble2<32,8><<<grdT, blkT, 0, stream>>>(d_c1, d_c0, n);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // Apply mask: d_c1 = d_c0 * H
    // This is a simple elementwise kernel; no special optimization beyond __restrict__.
    void apply_mask_to_F(const std::vector<double>& Hhost) {
        CUDA_CHECK(cudaMemcpyAsync(d_H, Hhost.data(), npx*sizeof(double),
                                   cudaMemcpyHostToDevice, stream));
        int threads = 256;
        int blocks = (int)((npx + threads - 1) / threads);
        apply_mask_kernel<<<blocks, threads, 0, stream>>>(d_c0, d_H, d_c1, (int)npx);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // (Common optimization #1 + #5 again)
    //
    // Inverse 2D IDFT with shift undo:
    // Input G in d_c1.
    //   transpose
    //   Row IDFT
    //   transpose back
    //   Row IDFT
    //   final fused output: scale + invshift (optimization #5) -> d_out
    void idft2_inverse_shift_from_G() {
        dim3 blkT(32,8);
        dim3 grdT((n+32-1)/32, (n+32-1)/32);

        // transpose (to make "column IDFT" into row-coalesced pass)
        transpose_cdouble2<32,8><<<grdT, blkT, 0, stream>>>(d_c1, d_c2, n);

        // row IDFT
        dim3 blk_dft(16,16);
        dim3 grd_dft((n+blk_dft.x-1)/blk_dft.x, (n+blk_dft.y-1)/blk_dft.y);
        dft_rows_recurrence<<<grd_dft, blk_dft, 0, stream>>>(d_c2, d_c0, n, 1);

        // transpose back
        transpose_cdouble2<32,8><<<grdT, blkT, 0, stream>>>(d_c0, d_c2, n);

        // row IDFT
        dft_rows_recurrence<<<grd_dft, blk_dft, 0, stream>>>(d_c2, d_c0, n, 1);

        // final fused: scale + invshift (optimization #5)
        dim3 blk_out(16,16);
        dim3 grd_out((n+blk_out.x-1)/blk_out.x, (n+blk_out.y-1)/blk_out.y);
        complex_to_img_scale_invshift<<<grd_out, blk_out, 0, stream>>>(d_c0, d_out, n, 1);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    void download_F(std::vector<cd>& hostF) {
        hostF.resize(npx);
        std::vector<cdouble2> tmp(npx);
        CUDA_CHECK(cudaMemcpyAsync(tmp.data(), d_c0, npx*sizeof(cdouble2),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for(size_t i=0;i<npx;++i) hostF[i] = cd(tmp[i].x, tmp[i].y);
    }

    void download_img(Image& out) {
        out = Image(n,n);
        CUDA_CHECK(cudaMemcpyAsync(out.px.data(), d_out, npx*sizeof(double),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
};

// -------------------- perf+metrics record --------------------
struct PerfRow2 { std::string test; int N; std::string stage; double ms; };
struct MetricsRow2 {
    std::string test; int N;
    std::string name; std::string ref_name;
    uint64_t hash_u8=0, hash_f64=0;
    double l2=0, rmse=0, max_abs=0, psnr=0;
};

void write_perf_csv2(const std::string& path, const std::vector<PerfRow2>& rows) {
    std::ofstream out(path);
    out << "test,N,stage,ms\n";
    for (const auto& r: rows) out << r.test << "," << r.N << "," << r.stage << "," << r.ms << "\n";
}
void write_metrics_csv2(const std::string& path, const std::vector<MetricsRow2>& rows) {
    std::ofstream out(path);
    out << "test,N,name,ref,hash_u8,hash_f64,l2,rmse,max_abs,psnr\n";
    for (const auto& r: rows) {
        out << r.test << "," << r.N << "," << r.name << "," << r.ref_name << ","
            << r.hash_u8 << "," << r.hash_f64 << ","
            << r.l2 << "," << r.rmse << "," << r.max_abs << "," << r.psnr << "\n";
    }
}

// -------------------- Main --------------------
int main(int argc, char** argv) {
    std::string outdir = ".";
    int runs = 3;
    int N = 2048; // runtime N (default)

    if (argc >= 2) outdir = argv[1];
    if (argc >= 3) runs = std::stoi(argv[2]);
    if (argc >= 4) N = std::stoi(argv[3]);

    if (N <= 0) {
        std::cerr << "Invalid N: " << N << "\n";
        return 1;
    }

    std::cout << "CUDA COMMON-OPT DFT benchmark N=" << N
              << " runs=" << runs << " outdir=" << outdir << "\n";

    GpuContext gpu;
    gpu.init(N);

    std::vector<PerfRow2> perf;
    std::vector<MetricsRow2> metrics;

    auto save = [&](const std::string& fname, const Image& img){
        std::string path = outdir + "/" + fname;
        if(!write_pgm(path, img)) std::cerr << "Failed writing: " << path << "\n";
    };

    auto record_metrics = [&](const std::string& test, const std::string& name,
                              const Image& img, const std::string& ref_name,
                              const Image* ref_img) {
        MetricsRow2 r;
        r.test = test;
        r.N = N;
        r.name = name;
        r.ref_name = ref_name;
        r.hash_u8 = hash_image_bytes_u8(img);
        r.hash_f64 = hash_image_f64(img);
        if (ref_img) {
            auto st = compare_images(img, *ref_img);
            r.l2 = st.l2;
            r.rmse = st.rmse;
            r.max_abs = st.max_abs;
            r.psnr = st.psnr;
        }
        metrics.push_back(r);
    };

    auto save_spectrum_from_Fhost = [&](const std::string& fname, const std::vector<cd>& F){
        Image spec = spectrum_to_image(F, N, N, /*shift_center=*/false);
        save(fname, spec);
        record_metrics("SPECTRUM", fname, spec, "", nullptr);
    };

    // Compute DFT (GPU), download F, save spectrum (not counted in perf)
    auto save_spectrum_from_img = [&](const std::string& test, const std::string& tag, const Image& img){
        gpu.dft2_forward_shift(img);
        std::vector<cd> Fhost;
        gpu.download_F(Fhost);
        Image spec = spectrum_to_image(Fhost, N, N, /*shift_center=*/false);
        std::string fname = test + "_" + tag + "_spectrum.pgm";
        save(fname, spec);
        record_metrics(test, fname, spec, "", nullptr);
    };

    auto time_dft2 = [&](const Image& img)->double{
        // Timing includes the transpose-based two-pass row DFT (optimization #1) and tiled transpose (optimization #2),
        // plus twiddle recurrence DFT kernel (optimization #3), and uses __restrict__ (optimization #4).
        return avg_ms_cuda(runs, [&]{ gpu.dft2_forward_shift(img); }, gpu.stream);
    };
    auto time_mask = [&](const std::vector<double>& H)->double{
        return avg_ms_cuda(runs, [&]{ gpu.apply_mask_to_F(H); }, gpu.stream);
    };
    auto time_idft2 = [&]()->double{
        // Timing includes fused scale+invshift in final kernel (optimization #5).
        return avg_ms_cuda(runs, [&]{ gpu.idft2_inverse_shift_from_G(); }, gpu.stream);
    };

    // ---------------- A: Gaussian low-pass blur ----------------
    {
        std::string test = "A_gauss_lowpass_blur";
        Image src = make_scene_mix(N);
        save(test + "_0_src.pgm", src);
        record_metrics(test, test + "_0_src", src, "", nullptr);
        save_spectrum_from_img(test, "0_src", src);

        double dft_ms = time_dft2(src);
        perf.push_back({test, N, "DFT2", dft_ms});

        std::vector<cd> Fhost;
        gpu.download_F(Fhost);
        save_spectrum_from_Fhost(test + "_0_src_spectrum_fromF.pgm", Fhost);

        auto H = gaussian_lowpass(N, (double)N*0.03);
        double mask_ms = time_mask(H);
        perf.push_back({test, N, "Mask_gaussian_lowpass", mask_ms});

        double idft_ms = time_idft2();
        perf.push_back({test, N, "IDFT2", idft_ms});

        Image blur;
        gpu.download_img(blur);
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

        double dft_ms = time_dft2(src);
        perf.push_back({test, N, "DFT2", dft_ms});

        std::vector<cd> Fhost;
        gpu.download_F(Fhost);
        save_spectrum_from_Fhost(test + "_0_src_spectrum_fromF.pgm", Fhost);

        auto H = gaussian_highpass(N, (double)N*0.02);
        double mask_ms = time_mask(H);
        perf.push_back({test, N, "Mask_gaussian_highpass", mask_ms});

        double idft_ms = time_idft2();
        perf.push_back({test, N, "IDFT2", idft_ms});

        Image edges;
        gpu.download_img(edges);
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

        double dft_ms = time_dft2(src);
        perf.push_back({test, N, "DFT2", dft_ms});

        std::vector<cd> Fhost;
        gpu.download_F(Fhost);
        save_spectrum_from_Fhost(test + "_0_src_spectrum_fromF.pgm", Fhost);

        auto H = ideal_bandpass(N, (double)N*0.06, (double)N*0.15);
        double mask_ms = time_mask(H);
        perf.push_back({test, N, "Mask_bandpass_ideal", mask_ms});

        double idft_ms = time_idft2();
        perf.push_back({test, N, "IDFT2", idft_ms});

        Image band;
        gpu.download_img(band);
        Image band_vis = normalize_to_0_255(band);

        save(test + "_1_bandpass_norm.pgm", band_vis);
        record_metrics(test, test + "_1_bandpass_norm", band_vis, "", nullptr);
        save_spectrum_from_img(test, "1_bandpass_norm", band_vis);
    }

    // ---------------- D: Checkerboard stress ----------------
    {
        std::string test = "D_checkerboard";
        int block = std::max(4, N/64);
        Image src = make_checkerboard(N, block);
        save(test + "_0_src.pgm", src);
        record_metrics(test, test + "_0_src", src, "", nullptr);
        save_spectrum_from_img(test, "0_src", src);

        double dft_ms = time_dft2(src);
        perf.push_back({test, N, "DFT2", dft_ms});

        std::vector<cd> Fhost;
        gpu.download_F(Fhost);
        save_spectrum_from_Fhost(test + "_0_src_spectrum_fromF.pgm", Fhost);

        auto H = gaussian_lowpass(N, (double)N*0.03);
        double mask_ms = time_mask(H);
        perf.push_back({test, N, "Mask_gaussian_lowpass", mask_ms});

        double idft_ms = time_idft2();
        perf.push_back({test, N, "IDFT2", idft_ms});

        Image out;
        gpu.download_img(out);

        save(test + "_1_lowpass_gaussian.pgm", out);
        record_metrics(test, test + "_1_lowpass_gaussian", out, test + "_0_src", &src);
        save_spectrum_from_img(test, "1_lowpass_gaussian", out);
    }

    // ---------------- E: Periodic noise -> auto notch denoise (auto only) ----------------
    {
        std::string test = "E_periodic_noise_auto_notch";
        Image base = make_scene_mix(N);
        save(test + "_0_base_clean.pgm", base);
        record_metrics(test, test + "_0_base_clean", base, "", nullptr);
        save_spectrum_from_img(test, "0_base_clean", base);

        std::vector<std::pair<int,int>> ks = { {12,0}, {0,18}, {22,14} };
        Image noisy = add_periodic_noise_multi(base, 45.0, ks);
        save(test + "_1_noisy.pgm", noisy);
        record_metrics(test, test + "_1_noisy", noisy, test + "_0_base_clean", &base);
        save_spectrum_from_img(test, "1_noisy", noisy);

        double dft_ms = time_dft2(noisy);
        perf.push_back({test, N, "DFT2", dft_ms});

        std::vector<cd> Fhost;
        gpu.download_F(Fhost);
        save_spectrum_from_Fhost(test + "_1_noisy_spectrum_fromF.pgm", Fhost);

        // Mask build is CPU-side (baseline-matching). This is not a GPU optimization;
        // it keeps the notch detection logic identical to your CPU reference behavior.
        std::vector<double> Hnotch;
        double build_ms = 0.0;
        for(int i=0;i<runs;++i){
            auto t0 = std::chrono::high_resolution_clock::now();
            Hnotch = auto_notch_mask(Fhost, N, 28.0, 14.0, 3);
            auto t1 = std::chrono::high_resolution_clock::now();
            build_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        perf.push_back({test, N, "Build_auto_notch", build_ms/std::max(1,runs)});

        double mask_ms = time_mask(Hnotch);
        perf.push_back({test, N, "Mask_auto_notch", mask_ms});

        double idft_ms = time_idft2();
        perf.push_back({test, N, "IDFT2", idft_ms});

        Image den;
        gpu.download_img(den);

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

        double dft_ms = time_dft2(src);
        perf.push_back({test, N, "DFT2", dft_ms});

        std::vector<cd> Fhost;
        gpu.download_F(Fhost);
        save_spectrum_from_Fhost(test + "_0_src_spectrum_fromF.pgm", Fhost);

        auto H = ideal_bandstop(N, (double)N*0.08, (double)N*0.14);
        double mask_ms = time_mask(H);
        perf.push_back({test, N, "Mask_bandstop_ideal", mask_ms});

        double idft_ms = time_idft2();
        perf.push_back({test, N, "IDFT2", idft_ms});

        Image out;
        gpu.download_img(out);

        save(test + "_1_bandstop_removed_rings.pgm", out);
        record_metrics(test, test + "_1_bandstop_removed_rings", out, test + "_0_src", &src);
        save_spectrum_from_img(test, "1_bandstop_removed_rings", out);
    }

    // Write CSVs
    write_perf_csv2(outdir + "/performance.csv", perf);
    write_metrics_csv2(outdir + "/metrics.csv", metrics);

    gpu.destroy();

    std::cout << "Done. Wrote PGM images + spectrum PGM images + performance.csv + metrics.csv\n";
    return 0;
}
