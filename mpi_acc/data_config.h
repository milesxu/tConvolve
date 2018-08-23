#ifndef DATA_CONFIG_H
#define DATA_CONFIG_H

#include <cstddef>
#include <complex>

struct Sample
{
    std::complex<double> data;
    int iu;
    int iv;
    int cOffset;
};

class DataConfig
{
  public:
    DataConfig(size_t nSamples, size_t nChannels, size_t gSize, size_t baseLine);
    DataConfig(size_t input[]);
    ~DataConfig();
    void InitArray(size_t len, double *randNum);
    void InitConvolveShape();
    void InitConvolveOffset();
    void RunGrid();

    size_t n_samples;
    size_t sub_samples;
    size_t size_samples;
    size_t n_channels;
    size_t g_size;
    size_t base_line;
    size_t w_size;
    size_t support;
    size_t over_sample;
    size_t s_size;
    double w_cell_size;
    double cell_size;
    double *u = NULL;
    double *v = NULL;
    double *w = NULL;
    Sample *samples = NULL;
    double *freq = NULL;
    std::complex<double> *grid = NULL;
    std::complex<double> *grid0 = NULL;
    std::complex<double> *convolve_shape = NULL;
};

#endif
