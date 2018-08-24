#include <cmath>
#include <iostream>
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

void DataConfig::InitArray(size_t len, double *restrict randNum)
{
    std::cout << "random num ----------- " << randNum[len * 4 - 1] << std::endl;
    sub_samples = len;
    u = new double[len];
    v = new double[len];
    w = new double[len];
    size_samples = len * n_channels;
    samples = new Sample[size_samples];
    auto grid_len = g_size * g_size;
    auto rl = len * 4;
    grid = new std::complex<double>[grid_len];
    grid0 = new std::complex<double>[grid_len];
    freq = new double[n_channels];
#pragma acc enter data create(u [0:len], v [0:len], w [0:len],             \
                              samples [0:size_samples], grid [0:grid_len], \
                              grid0 [0:grid_len], freq[0:n_channels])
#pragma acc parallel copyin(randNum [0:rl])
    {
        //auto offset = 0;
        for (auto i = 0; i < len; ++i)
        {
            auto offset = i * 4;
            auto rd1 = randNum[offset];
            auto rd2 = randNum[offset + 1];
            auto rd3 = randNum[offset + 2];
            auto rd4 = randNum[offset + 3];
            u[i] = base_line * rd1 - base_line / 2;
            v[i] = base_line * rd2 - base_line / 2;
            w[i] = base_line * rd3 - base_line / 2;
            for (auto j = 0; j < n_channels; ++j)
            {
                auto temp = double(j / n_channels);
                samples[i * n_channels + j].data =
                    std::complex<double>(rd4 + temp, rd4 - temp);
            }
            //offset += 4;
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
    }
    std::cout << "Init Array complete." << std::endl;
    InitConvolveShape();
    //InitConvolveOffset();
}

void DataConfig::InitConvolveShape()
{
    support = static_cast<int>(
        1.5 * std::sqrt(std::abs(double(base_line)) * cell_size * freq[0]) / cell_size);
    over_sample = 8;
    w_cell_size = 2 * base_line * freq[0] / w_size;
    s_size = 2 * support + 1;
    const int cCenter = (s_size - 1) / 2;
    const size_t cs_len = s_size * s_size * over_sample * over_sample * w_size;
    convolve_shape = new std::complex<double>[cs_len];

#pragma acc enter data create(convolve_shape [0:cs_len])
#pragma acc parallel
    {
        //double rr, ri;
        #pragma acc loop collapse(5)
        for (size_t k = 0; k < w_size; k++)
        {
            //double w = double(k - w_size / 2);

            for (size_t osj = 0; osj < over_sample; osj++)
            {
                for (size_t osi = 0; osi < over_sample; osi++)
                {
                    for (size_t j = 0; j < s_size; j++)
                    {

                        for (size_t i = 0; i < s_size; i++)
                        {
                            double j2 =
                                (double(j - cCenter) + double(osj) / double(over_sample));
                            j2 *= j2;
                            double i2 =
                                (double(i - cCenter) + double(osi) / double(over_sample));
                            i2 *= i2;
                            double r2 = j2 + i2 + std::sqrt(j2 * i2);
                            size_t cind =
                                i + s_size *
                                        (j + s_size *
                                                 (osi + over_sample *
                                                            (osj + over_sample * k)));

                            auto w = k - w_size / 2;
                            double fScale = std::sqrt(std::abs(w) * w_cell_size * freq[0]) / cell_size;
                            /*if (w != 0)
                            {
                                double rr = std::cos(r2 / (w * fScale));
                                double ri = std::sin(r2 / (w * fScale));
                                convolve_shape[cind] = std::complex<double>(rr, ri);
                            }
                            else
                            {
                                double rr = std::exp(-r2);
                                convolve_shape[cind] = std::complex<double>(rr);
                            }*/
                            double rr = w ? std::cos(r2/(w*fScale)) : std::exp(-r2);
                            double ri = w ? std::sin(r2/(w*fScale)) : 0;
                            //convolve_shape[cind] = w ? std::complex<double>(std::cos(r2/(w*fScale)), std::sin(r2/(w*fScale))) : std::complex<double>(std::exp(-r2));
                            convolve_shape[cind] = std::complex<double>(rr, ri);
                        }
                    }
                }
            }
        }
        double sum_shape = 0.0;
#pragma acc loop reduction(+ \
                           : sum_shape)
        for (auto i = 0; i < cs_len; ++i)
        {
            double temp_r = convolve_shape[i].real();
            double temp_i = convolve_shape[i].imag();
            double temp = std::sqrt(temp_r * temp_r + temp_i * temp_i);
            //sum_shape += std::abs(convolve_shape[i]);
            sum_shape += temp;
        }

        for (auto i = 0; i < cs_len; ++i)
        {
            convolve_shape[i] *= (w_size * over_sample * over_sample / sum_shape);
        }
    }
    std::cout << "Init convolve shape complete. " << std::endl;
}

void DataConfig::InitConvolveOffset()
{
#pragma acc parallel
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
void DataConfig::RunGrid()
{
#pragma acc parallel
    for (auto dind = 0; dind < size_samples; ++dind)
    {
        // The actual grid point from which we offset
        auto gind = samples[dind].iu + g_size * samples[dind].iv - support;

        // The Convoluton function point from which we offset
        int cind = samples[dind].cOffset;

        for (auto suppv = 0; suppv < s_size; suppv++)
        {
            std::complex<double> *gptr = grid + gind;
            const std::complex<double> *cptr = convolve_shape + cind;
            const std::complex<double> d = samples[dind].data;
            for (size_t suppu = 0; suppu < s_size; suppu++)
            {
                //*(gptr++) += d * (*(cptr++));
                gptr[suppu] += d * cptr[suppu];
            }

            gind += g_size;
            cind += s_size;
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
