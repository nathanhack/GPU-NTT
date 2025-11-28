// Copyright 2025 W. Nathan Hack <nathan.hack@gmail.com>
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: W. Nathan Hack <nathan.hack@gmail.com>

// This test verifies GPU Inverse NTT with multiple moduli (RNS-style),
// where polynomials are assigned to moduli in round-robin fashion.
// poly[i] uses modulus[i % MOD_COUNT]
//
// This tests the mod_count parameter functionality in GPU_INTT and
// GPU_INTT_Inplace, which is used in RNS-based cryptographic schemes.

#include <cstdlib>
#include <random>

#include "ntt.cuh"

using namespace std;
using namespace gpuntt;

int LOGN;
int BATCH;

// typedef Data32 TestDataType; // Use for 32-bit Test
// typedef Data32s TestDataTypeSigned; // Use for signed 32-bit Test
typedef Data64 TestDataType; // Use for 64-bit Test
typedef Data64s TestDataTypeSigned; // Use for signed 64-bit Test

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

    // Test 1: RNS INTT with GPU_INTT (separate input/output)
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

        cout << "\n=== Testing GPU RNS INTT (separate in/out) ===" << endl;
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
        // GPU kernels use (mod_index << N_power) = mod_index * N as offset
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

        // Performing CPU INTT for reference
        vector<vector<TestDataType>> intt_result(total_batch);
        for (int i = 0; i < total_batch; i++)
        {
            int mod_idx = i % MOD_COUNT;
            intt_result[i] = generators[mod_idx].intt(input1[i]);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // GPU Memory Allocation

        TestDataType* In_Datas;
        TestDataType* Out_Datas;

        GPUNTT_CUDA_CHECK(
            cudaMalloc(&In_Datas, total_batch * N * sizeof(TestDataType)));
        GPUNTT_CUDA_CHECK(
            cudaMalloc(&Out_Datas, total_batch * N * sizeof(TestDataType)));

        for (int j = 0; j < total_batch; j++)
        {
            GPUNTT_CUDA_CHECK(cudaMemcpy(In_Datas + (N * j), input1[j].data(),
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
        // Execute GPU INTT with mod_count

        ntt_rns_configuration<TestDataType> cfg_intt = {
            .n_power = LOGN,
            .ntt_type = INVERSE,
            .ntt_layout = PerPolynomial,
            .reduction_poly = ReductionPolynomial::X_N_minus,
            .zero_padding = false,
            .mod_inverse = test_ninverse,
            .stream = 0};

        GPU_INTT(In_Datas, Out_Datas, Inverse_Omega_Table_Device, test_modulus,
                 cfg_intt, total_batch, MOD_COUNT);

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Copy results back and verify

        TestDataType* Output_Host =
            (TestDataType*) malloc(total_batch * N * sizeof(TestDataType));

        GPUNTT_CUDA_CHECK(cudaMemcpy(Output_Host, Out_Datas,
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
                cout << "INTT FAILED (in poly " << i << ", modulus " << mod_idx
                     << ")" << endl;
                break;
            }

            if ((i == (total_batch - 1)) && check)
            {
                cout << "All Correct - GPU RNS INTT (separate in/out) with "
                     << MOD_COUNT << " moduli." << endl;
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Cleanup

        GPUNTT_CUDA_CHECK(cudaFree(In_Datas));
        GPUNTT_CUDA_CHECK(cudaFree(Out_Datas));
        GPUNTT_CUDA_CHECK(cudaFree(Inverse_Omega_Table_Device));
        GPUNTT_CUDA_CHECK(cudaFree(test_modulus));
        GPUNTT_CUDA_CHECK(cudaFree(test_ninverse));
        free(Output_Host);
    }

    // Test 2: RNS INTT with GPU_INTT_Inplace
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

        cout << "\n=== Testing GPU RNS INTT Inplace ===" << endl;
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
        int root_table_stride = N;

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
        // Execute GPU INTT Inplace with mod_count

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
                cout << "INTT Inplace FAILED (in poly " << i << ", modulus "
                     << mod_idx << ")" << endl;
                break;
            }

            if ((i == (total_batch - 1)) && check)
            {
                cout << "All Correct - GPU RNS INTT Inplace with " << MOD_COUNT
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

    // Test 3: RNS INTT with signed output
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

        cout << "\n=== Testing GPU RNS INTT with Signed Output ===" << endl;
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
        int root_table_stride = N;

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

        // Performing CPU INTT and convert to signed representation
        std::uint64_t mod_half = parameters_list[0].modulus.value >> 1;
        vector<vector<TestDataTypeSigned>> intt_result_signed(total_batch);
        for (int i = 0; i < total_batch; i++)
        {
            int mod_idx = i % MOD_COUNT;
            vector<TestDataType> intt_result = generators[mod_idx].intt(input1[i]);

            for (int j = 0; j < N; j++)
            {
                std::uint64_t r_modq = intt_result[j];
                if (r_modq > mod_half)
                {
                    intt_result_signed[i].push_back(
                        static_cast<std::int64_t>(r_modq) -
                        static_cast<std::int64_t>(
                            parameters_list[mod_idx].modulus.value));
                }
                else
                {
                    intt_result_signed[i].push_back(
                        static_cast<std::int64_t>(r_modq));
                }
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // GPU Memory Allocation

        TestDataType* In_Datas;
        TestDataTypeSigned* Out_Datas;

        GPUNTT_CUDA_CHECK(
            cudaMalloc(&In_Datas, total_batch * N * sizeof(TestDataType)));
        GPUNTT_CUDA_CHECK(cudaMalloc(
            &Out_Datas, total_batch * N * sizeof(TestDataTypeSigned)));

        for (int j = 0; j < total_batch; j++)
        {
            GPUNTT_CUDA_CHECK(cudaMemcpy(In_Datas + (N * j), input1[j].data(),
                                         N * sizeof(TestDataType),
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
        // Execute GPU INTT with signed output

        ntt_rns_configuration<TestDataType> cfg_intt = {
            .n_power = LOGN,
            .ntt_type = INVERSE,
            .ntt_layout = PerPolynomial,
            .reduction_poly = ReductionPolynomial::X_N_minus,
            .zero_padding = false,
            .mod_inverse = test_ninverse,
            .stream = 0};

        GPU_INTT(In_Datas, Out_Datas, Inverse_Omega_Table_Device, test_modulus,
                 cfg_intt, total_batch, MOD_COUNT);

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Copy results back and verify

        TestDataTypeSigned* Output_Host = (TestDataTypeSigned*) malloc(
            total_batch * N * sizeof(TestDataTypeSigned));

        GPUNTT_CUDA_CHECK(
            cudaMemcpy(Output_Host, Out_Datas,
                       total_batch * N * sizeof(TestDataTypeSigned),
                       cudaMemcpyDeviceToHost));

        bool check = true;
        for (int i = 0; i < total_batch; i++)
        {
            int mod_idx = i % MOD_COUNT;
            check = check_result(Output_Host + (i * N),
                                 intt_result_signed[i].data(), N);

            if (!check)
            {
                cout << "Signed INTT FAILED (in poly " << i << ", modulus "
                     << mod_idx << ")" << endl;
                break;
            }

            if ((i == (total_batch - 1)) && check)
            {
                cout << "All Correct - GPU RNS INTT Signed Output with "
                     << MOD_COUNT << " moduli." << endl;
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Cleanup

        GPUNTT_CUDA_CHECK(cudaFree(In_Datas));
        GPUNTT_CUDA_CHECK(cudaFree(Out_Datas));
        GPUNTT_CUDA_CHECK(cudaFree(Inverse_Omega_Table_Device));
        GPUNTT_CUDA_CHECK(cudaFree(test_modulus));
        GPUNTT_CUDA_CHECK(cudaFree(test_ninverse));
        free(Output_Host);
    }

    // Test 4: RNS INTT roundtrip verification (NTT then INTT should give back original)
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

        cout << "\n=== Testing GPU RNS NTT->INTT Roundtrip ===" << endl;
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
        int root_table_stride = N;

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
        // Execute Forward NTT

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
        // Execute Inverse NTT

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
                cout << "NTT->INTT Roundtrip FAILED (in poly " << i
                     << ", modulus " << mod_idx << ")" << endl;
                break;
            }

            if ((i == (total_batch - 1)) && check)
            {
                cout << "All Correct - GPU RNS NTT->INTT Roundtrip with "
                     << MOD_COUNT << " moduli." << endl;
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
