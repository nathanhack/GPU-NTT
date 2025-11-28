// Copyright 2025 W. Nathan Hack <nathan.hack@gmail.com>
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: W. Nathan Hack <nathan.hack@gmail.com>

// This test verifies GPU NTT with multiple moduli (RNS-style),
// where polynomials are assigned to moduli in round-robin fashion.
// poly[i] uses modulus[i % MOD_COUNT]
//
// This tests the mod_count parameter functionality in GPU_NTT_Inplace
// and GPU_INTT_Inplace, which is used in RNS-based cryptographic schemes.

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

    // Test 1: Forward NTT with RNS (multiple moduli)
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

        cout << "\n=== Testing GPU RNS Forward NTT ===" << endl;
        cout << "MOD_COUNT=" << MOD_COUNT << ", LOGN=" << LOGN
             << ", BATCH=" << BATCH << endl;

        // Total polynomials = BATCH * MOD_COUNT
        // poly[i] uses modulus[i % MOD_COUNT]
        int total_batch = BATCH * MOD_COUNT;

        // Create NTTParameters for each modulus (using default modulus for all)
        vector<NTTParameters<TestDataType>> parameters_list;
        parameters_list.reserve(MOD_COUNT);
        for (int m = 0; m < MOD_COUNT; m++)
        {
            parameters_list.emplace_back(LOGN, ReductionPolynomial::X_N_minus);
        }

        int N = parameters_list[0].n;
        int root_of_unity_size = parameters_list[0].root_of_unity_size;

        // GPU kernels use (mod_index << N_power) = mod_index * N as the offset
        // for root table indexing, so we must use N as stride, not root_of_unity_size
        int root_table_stride = N;

        // Create CPU NTT generators for verification
        vector<NTTCPU<TestDataType>> generators;
        generators.reserve(MOD_COUNT);
        for (int m = 0; m < MOD_COUNT; m++)
        {
            generators.emplace_back(parameters_list[m]);
            cout << "Modulus " << m << ": " << parameters_list[m].modulus.value
                 << endl;
        }

        std::random_device rd;
        std::mt19937 gen(0);
        std::uint64_t minNumber = 0;
        std::uint64_t maxNumber = parameters_list[0].modulus.value - 1;
        std::uniform_int_distribution<std::uint64_t> dis(minNumber, maxNumber);

        // Random data generation for polynomials
        vector<vector<TestDataType>> input1(total_batch);
        for (int j = 0; j < total_batch; j++)
        {
            for (int i = 0; i < N; i++)
            {
                input1[j].push_back(dis(gen));
            }
        }

        // Performing CPU NTT for reference
        vector<vector<TestDataType>> ntt_result(total_batch);
        for (int i = 0; i < total_batch; i++)
        {
            int mod_idx = i % MOD_COUNT;
            ntt_result[i] = generators[mod_idx].ntt(input1[i]);
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
        // Root of Unity Tables - concatenated for all moduli
        // Must use root_table_stride (= N) as offset, matching GPU kernel's (mod_index << N_power)

        Root<TestDataType>* Forward_Omega_Table_Device;
        GPUNTT_CUDA_CHECK(cudaMalloc(&Forward_Omega_Table_Device,
                                     MOD_COUNT * root_table_stride *
                                         sizeof(Root<TestDataType>)));

        // Concatenate root tables for all moduli
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
        // Modulus array

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
        // Execute GPU Forward NTT with mod_count

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
            check = check_result(Output_Host + (i * N), ntt_result[i].data(), N);

            if (!check)
            {
                cout << "Forward NTT FAILED (in poly " << i << ", modulus "
                     << mod_idx << ")" << endl;
                break;
            }

            if ((i == (total_batch - 1)) && check)
            {
                cout << "All Correct - GPU RNS Forward NTT with " << MOD_COUNT
                     << " moduli." << endl;
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Cleanup forward NTT resources

        GPUNTT_CUDA_CHECK(cudaFree(InOut_Datas));
        GPUNTT_CUDA_CHECK(cudaFree(Forward_Omega_Table_Device));
        GPUNTT_CUDA_CHECK(cudaFree(test_modulus));
        free(Output_Host);
    }

    // Test 2: Inverse NTT with RNS (multiple moduli)
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

        cout << "\n=== Testing GPU RNS Inverse NTT ===" << endl;
        cout << "MOD_COUNT=" << MOD_COUNT << ", LOGN=" << LOGN
             << ", BATCH=" << BATCH << endl;

        int total_batch = BATCH * MOD_COUNT;

        // Create NTTParameters for each modulus
        vector<NTTParameters<TestDataType>> parameters_list;
        parameters_list.reserve(MOD_COUNT);
        for (int m = 0; m < MOD_COUNT; m++)
        {
            parameters_list.emplace_back(LOGN, ReductionPolynomial::X_N_minus);
        }

        int N = parameters_list[0].n;
        int root_of_unity_size = parameters_list[0].root_of_unity_size;
        int root_table_stride = N;  // GPU kernels use (mod_index << N_power) = mod_index * N

        // Create CPU NTT generators for verification
        vector<NTTCPU<TestDataType>> generators;
        generators.reserve(MOD_COUNT);
        for (int m = 0; m < MOD_COUNT; m++)
        {
            generators.emplace_back(parameters_list[m]);
        }

        std::random_device rd;
        std::mt19937 gen(0);
        std::uint64_t minNumber = 0;
        std::uint64_t maxNumber = parameters_list[0].modulus.value - 1;
        std::uniform_int_distribution<std::uint64_t> dis(minNumber, maxNumber);

        // Random data generation for polynomials
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
        // Inverse Root of Unity Tables - concatenated for all moduli

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
        // Modulus array

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
        // Execute GPU Inverse NTT with mod_count

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
                cout << "Inverse NTT FAILED (in poly " << i << ", modulus "
                     << mod_idx << ")" << endl;
                break;
            }

            if ((i == (total_batch - 1)) && check)
            {
                cout << "All Correct - GPU RNS Inverse NTT with " << MOD_COUNT
                     << " moduli." << endl;
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

    // Test 3: Full NTT multiplication roundtrip with RNS
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

        cout << "\n=== Testing GPU RNS NTT Multiplication Roundtrip ===" << endl;
        cout << "MOD_COUNT=" << MOD_COUNT << ", LOGN=" << LOGN
             << ", BATCH=" << BATCH << endl;

        int total_batch = BATCH * MOD_COUNT;

        // Create NTTParameters for each modulus
        vector<NTTParameters<TestDataType>> parameters_list;
        parameters_list.reserve(MOD_COUNT);
        for (int m = 0; m < MOD_COUNT; m++)
        {
            parameters_list.emplace_back(LOGN, ReductionPolynomial::X_N_minus);
        }

        int N = parameters_list[0].n;
        int root_of_unity_size = parameters_list[0].root_of_unity_size;
        int root_table_stride = N;  // GPU kernels use (mod_index << N_power) = mod_index * N

        // Create CPU NTT generators for verification
        vector<NTTCPU<TestDataType>> generators;
        generators.reserve(MOD_COUNT);
        for (int m = 0; m < MOD_COUNT; m++)
        {
            generators.emplace_back(parameters_list[m]);
        }

        std::random_device rd;
        std::mt19937 gen(0);
        std::uint64_t minNumber = 0;
        std::uint64_t maxNumber = parameters_list[0].modulus.value - 1;
        std::uniform_int_distribution<std::uint64_t> dis(minNumber, maxNumber);

        // Random data generation for two sets of polynomials
        vector<vector<TestDataType>> input1(total_batch);
        vector<vector<TestDataType>> input2(total_batch);
        for (int j = 0; j < total_batch; j++)
        {
            for (int i = 0; i < N; i++)
            {
                input1[j].push_back(dis(gen));
                input2[j].push_back(dis(gen));
            }
        }

        // CPU reference: NTT mult result
        vector<vector<TestDataType>> cpu_mult_result(total_batch);
        for (int i = 0; i < total_batch; i++)
        {
            int mod_idx = i % MOD_COUNT;
            vector<TestDataType> ntt1 = generators[mod_idx].ntt(input1[i]);
            vector<TestDataType> ntt2 = generators[mod_idx].ntt(input2[i]);
            vector<TestDataType> mult_out = generators[mod_idx].mult(ntt1, ntt2);
            cpu_mult_result[i] = generators[mod_idx].intt(mult_out);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // GPU Memory Allocation

        TestDataType* Poly1_Device;
        TestDataType* Poly2_Device;
        GPUNTT_CUDA_CHECK(
            cudaMalloc(&Poly1_Device, total_batch * N * sizeof(TestDataType)));
        GPUNTT_CUDA_CHECK(
            cudaMalloc(&Poly2_Device, total_batch * N * sizeof(TestDataType)));

        for (int j = 0; j < total_batch; j++)
        {
            GPUNTT_CUDA_CHECK(cudaMemcpy(Poly1_Device + (N * j),
                                         input1[j].data(),
                                         N * sizeof(TestDataType),
                                         cudaMemcpyHostToDevice));
            GPUNTT_CUDA_CHECK(cudaMemcpy(Poly2_Device + (N * j),
                                         input2[j].data(),
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
        // Modulus array

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
        // Forward NTT on both polynomials

        ntt_rns_configuration<TestDataType> cfg_ntt = {
            .n_power = LOGN,
            .ntt_type = FORWARD,
            .ntt_layout = PerPolynomial,
            .reduction_poly = ReductionPolynomial::X_N_minus,
            .zero_padding = false,
            .mod_inverse = nullptr,
            .stream = 0};

        GPU_NTT_Inplace(Poly1_Device, Forward_Omega_Table_Device, test_modulus,
                        cfg_ntt, total_batch, MOD_COUNT);
        GPU_NTT_Inplace(Poly2_Device, Forward_Omega_Table_Device, test_modulus,
                        cfg_ntt, total_batch, MOD_COUNT);

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Pointwise multiplication (using simple kernel or CPU for now)
        // Note: A proper implementation would use GPU_Mult, but for this test
        // we'll copy back, multiply on CPU, and copy again

        vector<TestDataType> poly1_ntt(total_batch * N);
        vector<TestDataType> poly2_ntt(total_batch * N);

        GPUNTT_CUDA_CHECK(cudaMemcpy(poly1_ntt.data(), Poly1_Device,
                                     total_batch * N * sizeof(TestDataType),
                                     cudaMemcpyDeviceToHost));
        GPUNTT_CUDA_CHECK(cudaMemcpy(poly2_ntt.data(), Poly2_Device,
                                     total_batch * N * sizeof(TestDataType),
                                     cudaMemcpyDeviceToHost));

        // Pointwise multiply
        for (int i = 0; i < total_batch; i++)
        {
            int mod_idx = i % MOD_COUNT;
            Modulus<TestDataType> mod = parameters_list[mod_idx].modulus;
            for (int j = 0; j < N; j++)
            {
                int idx = i * N + j;
                poly1_ntt[idx] =
                    OPERATOR<TestDataType>::mult(poly1_ntt[idx], poly2_ntt[idx], mod);
            }
        }

        GPUNTT_CUDA_CHECK(cudaMemcpy(Poly1_Device, poly1_ntt.data(),
                                     total_batch * N * sizeof(TestDataType),
                                     cudaMemcpyHostToDevice));

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Inverse NTT

        ntt_rns_configuration<TestDataType> cfg_intt = {
            .n_power = LOGN,
            .ntt_type = INVERSE,
            .ntt_layout = PerPolynomial,
            .reduction_poly = ReductionPolynomial::X_N_minus,
            .zero_padding = false,
            .mod_inverse = test_ninverse,
            .stream = 0};

        GPU_INTT_Inplace(Poly1_Device, Inverse_Omega_Table_Device, test_modulus,
                         cfg_intt, total_batch, MOD_COUNT);

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Copy results back and verify

        TestDataType* Output_Host =
            (TestDataType*) malloc(total_batch * N * sizeof(TestDataType));

        GPUNTT_CUDA_CHECK(cudaMemcpy(Output_Host, Poly1_Device,
                                     total_batch * N * sizeof(TestDataType),
                                     cudaMemcpyDeviceToHost));

        bool check = true;
        for (int i = 0; i < total_batch; i++)
        {
            int mod_idx = i % MOD_COUNT;
            check = check_result(Output_Host + (i * N),
                                 cpu_mult_result[i].data(), N);

            if (!check)
            {
                cout << "NTT Mult Roundtrip FAILED (in poly " << i
                     << ", modulus " << mod_idx << ")" << endl;
                break;
            }

            if ((i == (total_batch - 1)) && check)
            {
                cout << "All Correct - GPU RNS NTT Multiplication with "
                     << MOD_COUNT << " moduli." << endl;
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Cleanup

        GPUNTT_CUDA_CHECK(cudaFree(Poly1_Device));
        GPUNTT_CUDA_CHECK(cudaFree(Poly2_Device));
        GPUNTT_CUDA_CHECK(cudaFree(Forward_Omega_Table_Device));
        GPUNTT_CUDA_CHECK(cudaFree(Inverse_Omega_Table_Device));
        GPUNTT_CUDA_CHECK(cudaFree(test_modulus));
        GPUNTT_CUDA_CHECK(cudaFree(test_ninverse));
        free(Output_Host);
    }

    return EXIT_SUCCESS;
}
