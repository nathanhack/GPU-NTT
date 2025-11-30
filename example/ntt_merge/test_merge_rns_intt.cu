// Copyright 2025 W. Nathan Hack <nathan.hack@gmail.com>
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: W. Nathan Hack <nathan.hack@gmail.com>

// This test verifies GPU INTT with multiple DISTINCT moduli (RNS-style) using
// X^n - 1 reduction polynomial, where polynomials are assigned to moduli in
// round-robin fashion. poly[i] uses modulus[i % MOD_COUNT]
//
// Uses 3 distinct 61-bit NTT-friendly primes of form k * 2^32 + 1
// This tests true RNS (Residue Number System) functionality with different
// primes, as used in lattice-based cryptographic schemes.

#include <cstdlib>
#include <random>

#include "ntt.cuh"

using namespace std;
using namespace gpuntt;

int LOGN;
int BATCH;

// typedef Data32 TestDataType; // Use for 32-bit Test
typedef Data64 TestDataType; // Use for 64-bit Test

// Number of RNS moduli to test with
constexpr int MOD_COUNT = 3;

// NTT-friendly primes of form k * 2^32 + 1 (~63 bits)
// These are DISTINCT primes for true RNS testing
// Each has a primitive generator g and supports NTT up to 2^32 points
struct PrimeInfo
{
    TestDataType prime;
    TestDataType omega_base; // Primitive generator g (used to compute omega)
    TestDataType psi_base;   // Primitive generator g (used to compute psi)
    int max_logn;            // Maximum LOGN supported (32 for these primes)
};

// Primitive generators for each prime
// g^((p-1)/2) = -1 mod p (quadratic non-residue)
// NOTE: Library Barrett reduction requires modulus <= 61 bits for Data64
//       (62-bit primes cause overflow in GPU Barrett reduction)
static PrimeInfo rns_primes[MOD_COUNT] = {
    // Prime 0: 2305842949084151809 = 536870898 * 2^32 + 1 (61 bits, g=7)
    {2305842949084151809ULL, 7, 7, 32},
    // Prime 1: 2305842811645198337 = 536870866 * 2^32 + 1 (61 bits, g=3)
    {2305842811645198337ULL, 3, 3, 32},
    // Prime 2: 2305842785875394561 = 536870860 * 2^32 + 1 (61 bits, g=3)
    {2305842785875394561ULL, 3, 3, 32}
};

// Compute omega for X^N - 1 polynomial at given LOGN
// omega is a primitive N-th root of unity: omega^N = 1
// omega = g^((p-1)/N) = g^((p-1)/2^logn)
TestDataType compute_omega(const PrimeInfo& info, int logn)
{
    Modulus<TestDataType> mod(info.prime);
    TestDataType pm1 = info.prime - 1;
    TestDataType exp_val = pm1 >> logn;  // (p-1) / 2^logn
    return OPERATOR<TestDataType>::exp(info.omega_base, exp_val, mod);
}

// Compute psi for X^N + 1 polynomial at given LOGN
// psi is a primitive 2N-th root of unity: psi^N = -1
// psi = g^((p-1)/(2N)) = g^((p-1)/2^(logn+1))
TestDataType compute_psi(const PrimeInfo& info, int logn)
{
    Modulus<TestDataType> mod(info.prime);
    TestDataType pm1 = info.prime - 1;
    TestDataType exp_val = pm1 >> (logn + 1);  // (p-1) / 2^(logn+1)
    return OPERATOR<TestDataType>::exp(info.psi_base, exp_val, mod);
}

int main(int argc, char* argv[])
{
    CudaDevice();

    int device = 0;
    cudaSetDevice(device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    std::cout << "Maximum Grid Size: " << prop.maxGridSize[0] << " x "
              << prop.maxGridSize[1] << " x " << prop.maxGridSize[2]
              << std::endl;

    // Test 1: Inverse NTT with RNS (multiple DISTINCT moduli) using X_N_minus
    {
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

        // Verify LOGN is supported
        for (int m = 0; m < MOD_COUNT; m++)
        {
            if (LOGN > rns_primes[m].max_logn)
            {
                cout << "LOGN=" << LOGN << " exceeds max supported ("
                     << rns_primes[m].max_logn << "). Using LOGN=12." << endl;
                LOGN = 12;
                break;
            }
        }

        cout << "\n=== Testing GPU RNS Inverse NTT (X^n - 1) with DISTINCT 61-bit "
                "primes ==="
             << endl;
        cout << "MOD_COUNT=" << MOD_COUNT << ", LOGN=" << LOGN
             << ", BATCH=" << BATCH << endl;

        int total_batch = BATCH * MOD_COUNT;

        // Create NTTParameters for each DISTINCT modulus using X_N_minus
        vector<NTTParameters<TestDataType>> parameters_list;
        parameters_list.reserve(MOD_COUNT);

        for (int m = 0; m < MOD_COUNT; m++)
        {
            TestDataType omega = compute_omega(rns_primes[m], LOGN);
            TestDataType psi = compute_psi(rns_primes[m], LOGN);

            NTTFactors<TestDataType> factor(
                Modulus<TestDataType>(rns_primes[m].prime), omega, psi);
            parameters_list.emplace_back(LOGN, factor,
                                         ReductionPolynomial::X_N_minus);

            cout << "Modulus " << m << ": " << rns_primes[m].prime
                 << " (" << rns_primes[m].prime / (1ULL << 32) << " * 2^32 + 1, "
                 << (int)log2(rns_primes[m].prime) + 1 << " bits)" << endl;
        }

        int N = parameters_list[0].n;
        int root_of_unity_size = parameters_list[0].root_of_unity_size;
        int root_table_stride = N;

        // Create CPU NTT generators for verification
        vector<NTTCPU<TestDataType>> generators;
        generators.reserve(MOD_COUNT);
        for (int m = 0; m < MOD_COUNT; m++)
        {
            generators.emplace_back(parameters_list[m]);
        }

        // Random data generation - use smallest modulus for safety
        std::mt19937 gen(0);
        TestDataType min_mod = rns_primes[0].prime;
        for (int m = 1; m < MOD_COUNT; m++)
        {
            min_mod = std::min(min_mod, rns_primes[m].prime);
        }
        std::uniform_int_distribution<TestDataType> dis(0, min_mod - 1);

        vector<vector<TestDataType>> input1(total_batch);
        for (int j = 0; j < total_batch; j++)
        {
            for (int i = 0; i < N; i++)
            {
                input1[j].push_back(dis(gen));
            }
        }

        // Performing CPU INTT for reference
        vector<vector<TestDataType>> intt_result(total_batch);
        for (int i = 0; i < total_batch; i++)
        {
            int mod_idx = i % MOD_COUNT;
            intt_result[i] = generators[mod_idx].intt(input1[i]);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // GPU Memory Allocation

        TestDataType* InOut_Datas;
        GPUNTT_CUDA_CHECK(
            cudaMalloc(&InOut_Datas, total_batch * N * sizeof(TestDataType)));

        for (int j = 0; j < total_batch; j++)
        {
            GPUNTT_CUDA_CHECK(cudaMemcpy(InOut_Datas + (N * j), input1[j].data(),
                                         N * sizeof(TestDataType),
                                         cudaMemcpyHostToDevice));
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Inverse Root of Unity Tables - concatenated for all DISTINCT moduli

        Root<TestDataType>* Inverse_Omega_Table_Device;
        GPUNTT_CUDA_CHECK(cudaMalloc(&Inverse_Omega_Table_Device,
                                     MOD_COUNT * root_table_stride *
                                         sizeof(Root<TestDataType>)));

        for (int m = 0; m < MOD_COUNT; m++)
        {
            vector<Root<TestDataType>> inverse_omega_table =
                parameters_list[m].gpu_root_of_unity_table_generator(
                    parameters_list[m].inverse_root_of_unity_table);

            GPUNTT_CUDA_CHECK(cudaMemcpy(
                Inverse_Omega_Table_Device + (m * root_table_stride),
                inverse_omega_table.data(),
                root_of_unity_size * sizeof(Root<TestDataType>),
                cudaMemcpyHostToDevice));
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Modulus array - DISTINCT values

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

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // N-inverse array for INTT

        Ninverse<TestDataType>* test_ninverse;
        GPUNTT_CUDA_CHECK(cudaMalloc(&test_ninverse,
                                     MOD_COUNT * sizeof(Ninverse<TestDataType>)));

        vector<Ninverse<TestDataType>> ninverse_array;
        for (int m = 0; m < MOD_COUNT; m++)
        {
            ninverse_array.push_back(parameters_list[m].n_inv);
        }

        GPUNTT_CUDA_CHECK(cudaMemcpy(test_ninverse, ninverse_array.data(),
                                     MOD_COUNT * sizeof(Ninverse<TestDataType>),
                                     cudaMemcpyHostToDevice));

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Execute GPU Inverse NTT with mod_count using X_N_minus

        ntt_rns_configuration<TestDataType> cfg_intt = {
            .n_power = LOGN,
            .ntt_type = INVERSE,
            .ntt_layout = PerPolynomial,
            .reduction_poly = ReductionPolynomial::X_N_minus,
            .zero_padding = false,
            .mod_inverse = test_ninverse,
            .stream = 0};

        GPU_INTT_Inplace(InOut_Datas, Inverse_Omega_Table_Device, test_modulus,
                         cfg_intt, total_batch, MOD_COUNT);

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Copy results back and verify

        TestDataType* Output_Host =
            (TestDataType*) malloc(total_batch * N * sizeof(TestDataType));

        GPUNTT_CUDA_CHECK(cudaMemcpy(Output_Host, InOut_Datas,
                                     total_batch * N * sizeof(TestDataType),
                                     cudaMemcpyDeviceToHost));

        bool check = true;
        for (int i = 0; i < total_batch; i++)
        {
            int mod_idx = i % MOD_COUNT;
            check =
                check_result(Output_Host + (i * N), intt_result[i].data(), N);

            if (!check)
            {
                cout << "Inverse NTT (X^n - 1) FAILED (in poly " << i
                     << ", modulus " << mod_idx << ")" << endl;
                break;
            }

            if ((i == (total_batch - 1)) && check)
            {
                cout << "All Correct - GPU RNS Inverse NTT (X^n - 1) with "
                     << MOD_COUNT << " DISTINCT 61-bit moduli." << endl;
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Cleanup

        GPUNTT_CUDA_CHECK(cudaFree(InOut_Datas));
        GPUNTT_CUDA_CHECK(cudaFree(Inverse_Omega_Table_Device));
        GPUNTT_CUDA_CHECK(cudaFree(test_modulus));
        GPUNTT_CUDA_CHECK(cudaFree(test_ninverse));
        free(Output_Host);
    }

    // Test 2: NTT -> INTT roundtrip with RNS using X_N_minus
    {
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

        cout << "\n=== Testing GPU RNS NTT->INTT Roundtrip (X^n - 1) ===" << endl;
        cout << "MOD_COUNT=" << MOD_COUNT << ", LOGN=" << LOGN
             << ", BATCH=" << BATCH << endl;

        int total_batch = BATCH * MOD_COUNT;

        // Create NTTParameters for each DISTINCT modulus using X_N_minus
        vector<NTTParameters<TestDataType>> parameters_list;
        parameters_list.reserve(MOD_COUNT);

        for (int m = 0; m < MOD_COUNT; m++)
        {
            TestDataType omega = compute_omega(rns_primes[m], LOGN);
            TestDataType psi = compute_psi(rns_primes[m], LOGN);

            NTTFactors<TestDataType> factor(
                Modulus<TestDataType>(rns_primes[m].prime), omega, psi);
            parameters_list.emplace_back(LOGN, factor,
                                         ReductionPolynomial::X_N_minus);
        }

        int N = parameters_list[0].n;
        int root_of_unity_size = parameters_list[0].root_of_unity_size;
        int root_table_stride = N;

        // Create CPU NTT generators for verification
        vector<NTTCPU<TestDataType>> generators;
        generators.reserve(MOD_COUNT);
        for (int m = 0; m < MOD_COUNT; m++)
        {
            generators.emplace_back(parameters_list[m]);
        }

        // Random data generation
        std::mt19937 gen(0);
        TestDataType min_mod = rns_primes[0].prime;
        for (int m = 1; m < MOD_COUNT; m++)
        {
            min_mod = std::min(min_mod, rns_primes[m].prime);
        }
        std::uniform_int_distribution<TestDataType> dis(0, min_mod - 1);

        vector<vector<TestDataType>> input1(total_batch);
        for (int j = 0; j < total_batch; j++)
        {
            for (int i = 0; i < N; i++)
            {
                input1[j].push_back(dis(gen));
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // GPU Memory Allocation

        TestDataType* InOut_Datas;
        GPUNTT_CUDA_CHECK(
            cudaMalloc(&InOut_Datas, total_batch * N * sizeof(TestDataType)));

        for (int j = 0; j < total_batch; j++)
        {
            GPUNTT_CUDA_CHECK(cudaMemcpy(InOut_Datas + (N * j), input1[j].data(),
                                         N * sizeof(TestDataType),
                                         cudaMemcpyHostToDevice));
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Forward Root of Unity Tables

        Root<TestDataType>* Forward_Omega_Table_Device;
        GPUNTT_CUDA_CHECK(cudaMalloc(&Forward_Omega_Table_Device,
                                     MOD_COUNT * root_table_stride *
                                         sizeof(Root<TestDataType>)));

        for (int m = 0; m < MOD_COUNT; m++)
        {
            vector<Root<TestDataType>> forward_omega_table =
                parameters_list[m].gpu_root_of_unity_table_generator(
                    parameters_list[m].forward_root_of_unity_table);

            GPUNTT_CUDA_CHECK(
                cudaMemcpy(Forward_Omega_Table_Device + (m * root_table_stride),
                           forward_omega_table.data(),
                           root_of_unity_size * sizeof(Root<TestDataType>),
                           cudaMemcpyHostToDevice));
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Inverse Root of Unity Tables

        Root<TestDataType>* Inverse_Omega_Table_Device;
        GPUNTT_CUDA_CHECK(cudaMalloc(&Inverse_Omega_Table_Device,
                                     MOD_COUNT * root_table_stride *
                                         sizeof(Root<TestDataType>)));

        for (int m = 0; m < MOD_COUNT; m++)
        {
            vector<Root<TestDataType>> inverse_omega_table =
                parameters_list[m].gpu_root_of_unity_table_generator(
                    parameters_list[m].inverse_root_of_unity_table);

            GPUNTT_CUDA_CHECK(cudaMemcpy(
                Inverse_Omega_Table_Device + (m * root_table_stride),
                inverse_omega_table.data(),
                root_of_unity_size * sizeof(Root<TestDataType>),
                cudaMemcpyHostToDevice));
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Modulus array - DISTINCT values

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

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // N-inverse array

        Ninverse<TestDataType>* test_ninverse;
        GPUNTT_CUDA_CHECK(cudaMalloc(&test_ninverse,
                                     MOD_COUNT * sizeof(Ninverse<TestDataType>)));

        vector<Ninverse<TestDataType>> ninverse_array;
        for (int m = 0; m < MOD_COUNT; m++)
        {
            ninverse_array.push_back(parameters_list[m].n_inv);
        }

        GPUNTT_CUDA_CHECK(cudaMemcpy(test_ninverse, ninverse_array.data(),
                                     MOD_COUNT * sizeof(Ninverse<TestDataType>),
                                     cudaMemcpyHostToDevice));

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Forward NTT using X_N_minus

        ntt_rns_configuration<TestDataType> cfg_ntt = {
            .n_power = LOGN,
            .ntt_type = FORWARD,
            .ntt_layout = PerPolynomial,
            .reduction_poly = ReductionPolynomial::X_N_minus,
            .zero_padding = false,
            .mod_inverse = nullptr,
            .stream = 0};

        GPU_NTT_Inplace(InOut_Datas, Forward_Omega_Table_Device, test_modulus,
                        cfg_ntt, total_batch, MOD_COUNT);

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Inverse NTT using X_N_minus

        ntt_rns_configuration<TestDataType> cfg_intt = {
            .n_power = LOGN,
            .ntt_type = INVERSE,
            .ntt_layout = PerPolynomial,
            .reduction_poly = ReductionPolynomial::X_N_minus,
            .zero_padding = false,
            .mod_inverse = test_ninverse,
            .stream = 0};

        GPU_INTT_Inplace(InOut_Datas, Inverse_Omega_Table_Device, test_modulus,
                         cfg_intt, total_batch, MOD_COUNT);

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Copy results back and verify against original input

        TestDataType* Output_Host =
            (TestDataType*) malloc(total_batch * N * sizeof(TestDataType));

        GPUNTT_CUDA_CHECK(cudaMemcpy(Output_Host, InOut_Datas,
                                     total_batch * N * sizeof(TestDataType),
                                     cudaMemcpyDeviceToHost));

        bool check = true;
        for (int i = 0; i < total_batch; i++)
        {
            int mod_idx = i % MOD_COUNT;
            check = check_result(Output_Host + (i * N), input1[i].data(), N);

            if (!check)
            {
                cout << "NTT->INTT Roundtrip (X^n - 1) FAILED (in poly " << i
                     << ", modulus " << mod_idx << ")" << endl;
                break;
            }

            if ((i == (total_batch - 1)) && check)
            {
                cout << "All Correct - GPU RNS NTT->INTT Roundtrip (X^n - 1) with "
                     << MOD_COUNT << " DISTINCT 61-bit moduli." << endl;
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Cleanup

        GPUNTT_CUDA_CHECK(cudaFree(InOut_Datas));
        GPUNTT_CUDA_CHECK(cudaFree(Forward_Omega_Table_Device));
        GPUNTT_CUDA_CHECK(cudaFree(Inverse_Omega_Table_Device));
        GPUNTT_CUDA_CHECK(cudaFree(test_modulus));
        GPUNTT_CUDA_CHECK(cudaFree(test_ninverse));
        free(Output_Host);
    }

    return EXIT_SUCCESS;
}
