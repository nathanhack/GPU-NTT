// Copyright 2025 W. Nathan Hack <nathan.hack@gmail.com>
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: W. Nathan Hack

// This test verifies GPU 4-step INTT with multiple DISTINCT moduli (RNS-style),
// where polynomials are assigned to moduli in round-robin fashion.
// Uses 3 distinct 61-bit NTT-friendly primes of form k * 2^32 + 1

#include <cstdlib>
#include <random>

#include "ntt.cuh"
#include "ntt_4step.cuh"
#include "ntt_4step_cpu.cuh"

using namespace std;
using namespace gpuntt;

int LOGN;
int BATCH;

// typedef Data32 TestDataType; // Use for 32-bit Test
typedef Data64 TestDataType; // Use for 64-bit Test

// Number of RNS moduli to test with
constexpr int MOD_COUNT = 3;

// NTT-friendly primes of form k * 2^32 + 1 (~61 bits)
struct PrimeInfo
{
    TestDataType prime;
    TestDataType generator;
    int max_logn;
};

static PrimeInfo rns_primes[MOD_COUNT] = {
    {2305842949084151809ULL, 7, 32},
    {2305842811645198337ULL, 3, 32},
    {2305842785875394561ULL, 3, 32}
};

TestDataType compute_omega(const PrimeInfo& info, int logn)
{
    Modulus<TestDataType> mod(info.prime);
    TestDataType pm1 = info.prime - 1;
    TestDataType exp_val = pm1 >> logn;
    return OPERATOR<TestDataType>::exp(info.generator, exp_val, mod);
}

TestDataType compute_psi(const PrimeInfo& info, int logn)
{
    Modulus<TestDataType> mod(info.prime);
    TestDataType pm1 = info.prime - 1;
    TestDataType exp_val = pm1 >> (logn + 1);
    return OPERATOR<TestDataType>::exp(info.generator, exp_val, mod);
}

int main(int argc, char* argv[])
{
    CudaDevice();

    if (argc < 3)
    {
        LOGN = 12;
        BATCH = 4;
    }
    else
    {
        LOGN = atoi(argv[1]);
        BATCH = atoi(argv[2]);
    }

    for (int m = 0; m < MOD_COUNT; m++)
    {
        if (LOGN > rns_primes[m].max_logn)
        {
            cout << "LOGN=" << LOGN << " exceeds max. Using LOGN=12." << endl;
            LOGN = 12;
            break;
        }
    }

    int total_polys = BATCH * MOD_COUNT;

    cout << "Testing GPU 4-Step RNS INTT with " << MOD_COUNT
         << " DISTINCT moduli, LOGN=" << LOGN << ", BATCH=" << BATCH
         << " (total_polys=" << total_polys << ")" << endl;

    // Create NTTParameters4Step for each RNS modulus with distinct primes
    vector<NTTParameters4Step<TestDataType>> parameters_list;
    parameters_list.reserve(MOD_COUNT);

    for (int m = 0; m < MOD_COUNT; m++)
    {
        TestDataType omega = compute_omega(rns_primes[m], LOGN);
        TestDataType psi = compute_psi(rns_primes[m], LOGN);

        NTTFactors<TestDataType> factor(
            Modulus<TestDataType>(rns_primes[m].prime), omega, psi);
        parameters_list.emplace_back(LOGN, factor, ReductionPolynomial::X_N_minus);

        cout << "Modulus " << m << ": " << rns_primes[m].prime
             << " (" << rns_primes[m].prime / (1ULL << 32) << " * 2^32 + 1)" << endl;
    }

    // Create CPU NTT generators for reference
    vector<NTT_4STEP_CPU<TestDataType>> generators;
    generators.reserve(MOD_COUNT);
    for (int m = 0; m < MOD_COUNT; m++)
    {
        generators.emplace_back(parameters_list[m]);
    }

    int N = parameters_list[0].n;

    std::random_device rd;
    std::mt19937 gen(rd());

    TestDataType min_mod = rns_primes[0].prime;
    for (int m = 1; m < MOD_COUNT; m++)
    {
        min_mod = std::min(min_mod, rns_primes[m].prime);
    }
    std::uniform_int_distribution<TestDataType> dis(0, min_mod - 1);

    // Random data generation for polynomials
    vector<vector<TestDataType>> input1(total_polys);
    for (int j = 0; j < total_polys; j++)
    {
        for (int i = 0; i < N; i++)
        {
            input1[j].push_back(dis(gen));
        }
    }

    // Compute CPU INTT results for reference
    vector<vector<TestDataType>> cpu_intt_result(total_polys);
    for (int i = 0; i < total_polys; i++)
    {
        int mod_idx = i % MOD_COUNT;
        cpu_intt_result[i] = generators[mod_idx].intt(input1[i]);
    }

    ////////////////////////////////////////////////////////////////////////////
    // GPU Setup
    ////////////////////////////////////////////////////////////////////////////

    TestDataType* Input_Datas;
    GPUNTT_CUDA_CHECK(
        cudaMalloc(&Input_Datas, total_polys * N * sizeof(TestDataType)));

    TestDataType* Output_Datas;
    GPUNTT_CUDA_CHECK(
        cudaMalloc(&Output_Datas, total_polys * N * sizeof(TestDataType)));

    // For INTT, we need to apply the first transpose on CPU before upload
    for (int j = 0; j < total_polys; j++)
    {
        int mod_idx = j % MOD_COUNT;
        vector<TestDataType> cpu_intt_transposed_input =
            generators[mod_idx].intt_first_transpose(input1[j]);

        GPUNTT_CUDA_CHECK(cudaMemcpy(Input_Datas + (N * j),
                                     cpu_intt_transposed_input.data(),
                                     N * sizeof(TestDataType),
                                     cudaMemcpyHostToDevice));
    }

    //////////////////////////////////////////////////////////////////////////
    // Inverse root of unity tables - concatenated for all moduli
    //////////////////////////////////////////////////////////////////////////

    int n1 = parameters_list[0].n1;
    int n2 = parameters_list[0].n2;

    vector<Root<TestDataType>> psitable1_all;
    vector<Root<TestDataType>> psitable2_all;
    vector<Root<TestDataType>> W_table_all;

    for (int m = 0; m < MOD_COUNT; m++)
    {
        vector<Root<TestDataType>> psitable1 =
            parameters_list[m].gpu_root_of_unity_table_generator(
                parameters_list[m].n1_based_inverse_root_of_unity_table);
        vector<Root<TestDataType>> psitable2 =
            parameters_list[m].gpu_root_of_unity_table_generator(
                parameters_list[m].n2_based_inverse_root_of_unity_table);

        psitable1_all.insert(psitable1_all.end(), psitable1.begin(),
                             psitable1.end());
        psitable2_all.insert(psitable2_all.end(), psitable2.begin(),
                             psitable2.end());
        W_table_all.insert(
            W_table_all.end(),
            parameters_list[m].W_inverse_root_of_unity_table.begin(),
            parameters_list[m].W_inverse_root_of_unity_table.end());
    }

    Root<TestDataType>* psitable_device1;
    GPUNTT_CUDA_CHECK(cudaMalloc(
        &psitable_device1, MOD_COUNT * (n1 >> 1) * sizeof(Root<TestDataType>)));
    GPUNTT_CUDA_CHECK(
        cudaMemcpy(psitable_device1, psitable1_all.data(),
                   MOD_COUNT * (n1 >> 1) * sizeof(Root<TestDataType>),
                   cudaMemcpyHostToDevice));

    Root<TestDataType>* psitable_device2;
    GPUNTT_CUDA_CHECK(cudaMalloc(
        &psitable_device2, MOD_COUNT * (n2 >> 1) * sizeof(Root<TestDataType>)));
    GPUNTT_CUDA_CHECK(
        cudaMemcpy(psitable_device2, psitable2_all.data(),
                   MOD_COUNT * (n2 >> 1) * sizeof(Root<TestDataType>),
                   cudaMemcpyHostToDevice));

    Root<TestDataType>* W_Table_device;
    GPUNTT_CUDA_CHECK(
        cudaMalloc(&W_Table_device, MOD_COUNT * N * sizeof(Root<TestDataType>)));
    GPUNTT_CUDA_CHECK(cudaMemcpy(W_Table_device, W_table_all.data(),
                                 MOD_COUNT * N * sizeof(Root<TestDataType>),
                                 cudaMemcpyHostToDevice));

    //////////////////////////////////////////////////////////////////////////
    // Modulus and N_inverse arrays
    //////////////////////////////////////////////////////////////////////////

    Modulus<TestDataType>* test_modulus;
    GPUNTT_CUDA_CHECK(
        cudaMalloc(&test_modulus, MOD_COUNT * sizeof(Modulus<TestDataType>)));

    vector<Modulus<TestDataType>> modulus_array;
    for (int m = 0; m < MOD_COUNT; m++)
    {
        modulus_array.push_back(parameters_list[m].modulus);
    }
    GPUNTT_CUDA_CHECK(cudaMemcpy(test_modulus, modulus_array.data(),
                                 MOD_COUNT * sizeof(Modulus<TestDataType>),
                                 cudaMemcpyHostToDevice));

    Ninverse<TestDataType>* test_ninverse;
    GPUNTT_CUDA_CHECK(
        cudaMalloc(&test_ninverse, MOD_COUNT * sizeof(Ninverse<TestDataType>)));

    vector<Ninverse<TestDataType>> ninverse_array;
    for (int m = 0; m < MOD_COUNT; m++)
    {
        ninverse_array.push_back(parameters_list[m].n_inv);
    }
    GPUNTT_CUDA_CHECK(cudaMemcpy(test_ninverse, ninverse_array.data(),
                                 MOD_COUNT * sizeof(Ninverse<TestDataType>),
                                 cudaMemcpyHostToDevice));

    ntt4step_rns_configuration<TestDataType> cfg_intt = {.n_power = LOGN,
                                                         .ntt_type = INVERSE,
                                                         .mod_inverse =
                                                             test_ninverse,
                                                         .stream = 0};

    //////////////////////////////////////////////////////////////////////////
    // Execute GPU INTT
    //////////////////////////////////////////////////////////////////////////

    // INTT with RNS
    GPU_4STEP_NTT(Input_Datas, Output_Datas, psitable_device1, psitable_device2,
                  W_Table_device, test_modulus, cfg_intt, total_polys, MOD_COUNT);

    // Transpose back
    GPU_Transpose(Output_Datas, Input_Datas, n1, n2, LOGN, total_polys);

    vector<TestDataType> Output_Host(N * total_polys);
    cudaMemcpy(Output_Host.data(), Input_Datas,
               N * total_polys * sizeof(TestDataType), cudaMemcpyDeviceToHost);

    // Comparing GPU INTT results and CPU INTT results
    bool check = true;
    for (int i = 0; i < total_polys; i++)
    {
        check = check_result(Output_Host.data() + (i * N),
                             cpu_intt_result[i].data(), N);

        if (!check)
        {
            cout << "FAILED (in poly " << i << ", modulus " << (i % MOD_COUNT)
                 << ")" << endl;
            break;
        }

        if ((i == (total_polys - 1)) && check)
        {
            cout << "All Correct - GPU 4-Step RNS INTT with " << MOD_COUNT
                 << " DISTINCT moduli verified." << endl;
        }
    }

    // Cleanup
    cudaFree(Input_Datas);
    cudaFree(Output_Datas);
    cudaFree(psitable_device1);
    cudaFree(psitable_device2);
    cudaFree(W_Table_device);
    cudaFree(test_modulus);
    cudaFree(test_ninverse);

    return EXIT_SUCCESS;
}
