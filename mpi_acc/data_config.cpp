#include <cmath>
#include "data_config.h"

DataConfig::DataConfig(size_t nSamples, size_t nChannels, size_t gSize,
                       size_t baseLine) : n_samples(nSamples),
                                          n_channels(nChannels),
                                          g_size(gSize),
                                          base_line(baseLine),
                                          w_size(33),
                                          cell_size(5.0)
{
}

DataConfig::DataConfig(size_t input[])
{
    n_samples = input[0];
    n_channels = input[1];
    g_size = input[2];
    base_line = input[3];
    w_size = 33;
    cell_size = 5.0;
}

void DataConfig::InitArray(size_t len, double *randNum)
{
    sub_samples = len;
    u = new double[len];
    v = new double[len];
    w = new double[len];
    samples = new Sample[len * n_channels];
    auto grid_len = g_size * g_size;
    grid = new std::complex<double>[grid_len];
    grid0 = new std::complex<double>[grid_len];
    freq = new double[n_channels];
#pragma acc enter data create(u [0:len], v [0:len], w [0:len],                 \
                              samples [0:len * n_channels], grid [0:grid_len], \
                              grid0 [0:grid_len])
#pragma acc parallel copyin(randNum [0:len])
    {
        auto offset = 0;
        for (auto i = 0; i < len; ++i)
        {
            auto rd1 = randNum[offset];
            auto rd2 = randNum[offset + 1];
            auto rd3 = randNum[offset + 2];
            auto rd4 = randNum[offset + 3];
            u[i] = base_line * rd1 - base_line / 2;
            v[i] = base_line * rd2 - base_line / 2;
            w[i] = base_line * rd3 - base_line / 2;
            for (auto j = 0; i < n_channels; ++j)
            {
                auto temp = double(j / n_channels);
                samples[i * n_channels + j].data =
                    std::complex<double>(rd4 + temp, rd4 - temp);
            }
            offset += 4;
        }
        for (auto i = 0; i < grid_len; ++i)
        {
            grid[i] = std::complex<double>(0.0);
            grid0[i] = std::complex<double>(0.0);
        }
        for (auto i = 0; i < n_channels; ++i)
        {
            freq[i] = (1.4e9 - 2.0e5 * i / n_channels) / 2.998e8;
        }

        InitConvolveShape();
        InitConvolveOffset();
    }
}

void DataConfig::InitConvolveShape()
{
    support = static_cast<int>(
        1.5 * sqrt(std::abs(double(base_line)) * static_cast<double>(cell_size) * freq[0]) / cell_size);
    over_sample = 8;
    w_cell_size = 2 * base_line * freq[0] / w_size;
    s_size = 2 * support + 1;
    const int cCenter = (s_size - 1) / 2;
    const size_t cs_len = s_size * s_size * over_sample * over_sample * w_size;
    convolve_shape = new std::complex<double>[cs_len];

#pragma acc enter data create(convolve_shape [0:cs_len])

    double rr, ri;
    for (int k = 0; k < w_size; k++)
    {
        double w = double(k - w_size / 2);
        double fScale = sqrt(std::abs(w) * w_cell_size * freq[0]) / cell_size;

        for (int osj = 0; osj < over_sample; osj++)
        {
            for (int osi = 0; osi < over_sample; osi++)
            {
                for (int j = 0; j < s_size; j++)
                {
                    double j2 = std::pow(
                        (double(j - cCenter) + double(osj) / double(over_sample)),
                        2);

                    for (int i = 0; i < s_size; i++)
                    {
                        double i2 = std::pow(
                            (double(i - cCenter) + double(osi) / double(over_sample)),
                            2);
                        double r2 = j2 + i2 + sqrt(j2 * i2);
                        long int cind =
                            i + s_size *
                                    (j + s_size *
                                             (osi + over_sample *
                                                        (osj + over_sample * k)));

                        if (w != 0.0)
                        {
                            rr = std::cos(r2 / (w * fScale));
                            ri = std::sin(r2 / (w * fScale));
                            convolve_shape[cind] =
                                static_cast<std::complex<double> >(rr, ri);
                        }
                        else
                        {
                            rr = std::exp(-r2);
                            convolve_shape[cind] =
                                static_cast<std::complex<double> >(rr);
                        }
                    }
                }
            }
        }
    }
    double sum_shape = 0.0;
#pragma acc parallel loop reduction(+ \
                                    : sum_shape)
    for (auto i = 0; i < cs_len; ++i)
        sum_shape += std::abs(convolve_shape[i]);

    for (auto i = 0; i < cs_len; ++i)
    {
        convolve_shape[i] *= (w_size * over_sample * over_sample / sum_shape);
    }
}

void DataConfig::InitConvolveOffset()
{
    for (int i = 0; i < sub_samples; i++)
    {
        for (int chan = 0; chan < n_channels; chan++)
        {

            int dind = i * n_channels + chan;

            double uScaled = freq[chan] * u[i] / cell_size;
            samples[dind].iu = int(uScaled);

            if (uScaled < double(samples[dind].iu))
            {
                samples[dind].iu -= 1;
            }

            int fracu = int(over_sample * (uScaled - double(samples[dind].iu)));
            samples[dind].iu += g_size / 2;

            double vScaled = freq[chan] * v[i] / cell_size;
            samples[dind].iv = int(vScaled);

            if (vScaled < double(samples[dind].iv))
            {
                samples[dind].iv -= 1;
            }

            int fracv = int(over_sample * (vScaled - double(samples[dind].iv)));
            samples[dind].iv += g_size / 2;

            // The beginning of the convolution function for this point
            double wScaled = freq[chan] * w[i] / w_cell_size;
            int woff = w_size / 2 + int(wScaled);
            samples[dind].cOffset = s_size * s_size *
                                    (fracu + over_sample *
                                                 (fracv + over_sample * woff));
        }
    }
}

DataConfig::~DataConfig()
{
    if (u)
        delete[] u;
    if (v)
        delete[] v;
    if (w)
        delete[] w;
    if (samples)
        delete[] samples;
    if (grid)
        delete[] grid;
    if (grid0)
        delete[] grid0;
    if (convolve_shape)
        delete[] convolve_shape;
    if (freq)
        delete[] freq;
#pragma acc exit data
}
