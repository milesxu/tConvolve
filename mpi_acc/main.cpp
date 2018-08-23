#include <mpi.h>
#include <random>
#include <iostream>

#include "data_config.h"

#define BLOCK_LOW(id, p, n) ((id) * (n) / (p))
#define BLOCK_HIGH(id, p, n) (BLOCK_LOW((id) + 1, p, n) - 1)
#define BLOCK_SIZE(id, p, n) (BLOCK_HIGH(id, p, n) - BLOCK_LOW(id, p, n) + 1)

const size_t seed = 7715;

int main(int argc, char *argv[])
{
    auto rc = MPI_Init(&argc, &argv);
    if (rc != MPI_SUCCESS)
    {
        std::cout << "Error starting MPI program. Terminated." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, rc);
    }
    int num_tasks, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    DataConfig *config;
    size_t input[4];
    if (rank == 0)
    {
        input[0] = 3200000; //n samples
        input[1] = 1;       // n channels
        input[2] = 4096;
        input[3] = 2000;
    }
    MPI_Bcast(input, 4, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    //MPI_Bcast(&cellSize, 1, MPI_DOUBLE_PRECISION, 0, MPI_COMM_WORLD);
    config = new DataConfig(input);
    if (rank == 0)
    {
        std::mt19937 engine(seed);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        const auto total_samples = input[0] * (3 + input[1]);
        auto rand_num = new double[total_samples];
        for (auto i = 0; i < total_samples; ++i)
        {
            rand_num[i] = dist(engine);
        }
        auto nl = BLOCK_SIZE(0, num_tasks, config->n_samples);
        auto temp_ptr = rand_num + nl;
        for (auto i = 1; i < num_tasks; ++i)
        {
            auto incr = BLOCK_SIZE(i, num_tasks, config->n_samples);
            if (i == num_tasks - 1)
                incr += 1;
            MPI_Send(temp_ptr, incr, MPI_DOUBLE_PRECISION, i, 0, MPI_COMM_WORLD);
            temp_ptr += incr;
        }
        config->InitArray(nl, rand_num);
        delete[] rand_num;
    }
    else
    {
        auto ns = BLOCK_SIZE(rank, num_tasks, config->n_samples);
        size_t nl = ns;
        if (rank == num_tasks - 1)
        {
            nl += 1;
        }
        double *rand_buf = new double[nl];
        MPI_Status status;
        MPI_Recv(rand_buf, nl, MPI_DOUBLE_PRECISION, 0, 0, MPI_COMM_WORLD,
                 &status);
        config->InitArray(nl, rand_buf);
        delete[] rand_buf;
    }
    config->RunGrid();
    MPI_Reduce(config->grid, config->grid0, config->g_size * config->g_size,
               MPI_DOUBLE_COMPLEX, MPI_SUM, 0, MPI_COMM_WORLD);
    delete config;
}
