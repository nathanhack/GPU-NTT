// Copyright 2025 W. Nathan Hack <nathan.hack@gmail.com>
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: W. Nathan Hack

// This test verifies CPU 4-step NTT with multiple moduli (RNS-style),
// where polynomials are assigned to moduli in round-robin fashion.
// This pattern is used in RNS (Residue Number System) based cryptography
// where each polynomial is computed modulo a different prime.

#include <cstdlib> // For atoi or atof functions
#include <fstream>
#include <random>

#include "ntt.cuh"
#include "ntt_4step_cpu.cuh"

using namespace std;
using namespace gpuntt;

int LOGN;
int BATCH;
int N;

// typedef Data32 TestDataType; // Use for 32-bit Test
typedef Data64 TestDataType; // Use for 64-bit Test

// Number of RNS moduli to test with
constexpr int MOD_COUNT = 3;

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

    cout << "Testing CPU 4-Step RNS NTT with " << MOD_COUNT
         << " moduli, LOGN=" << LOGN << ", BATCH=" << BATCH << endl;

    // Create NTTParameters4Step for each RNS modulus
    // Using default modulus for all (tests dispatch pattern)
    vector<NTTParameters4Step<TestDataType>> parameters_list;
    parameters_list.reserve(MOD_COUNT);

    for (int m = 0; m < MOD_COUNT; m++)
    {
        parameters_list.emplace_back(LOGN, ReductionPolynomial::X_N_minus);
        cout << "Modulus " << m << ": " << parameters_list[m].modulus.value
             << endl;
    }

    // Create CPU NTT generators for each modulus
    vector<NTT_4STEP_CPU<TestDataType>> generators;
    generators.reserve(MOD_COUNT);
    for (int m = 0; m < MOD_COUNT; m++)
    {
        generators.emplace_back(parameters_list[m]);
    }

    N = parameters_list[0].n;

    // Total number of polynomials = BATCH * MOD_COUNT
    int total_polys = BATCH * MOD_COUNT;

    std::random_device rd;
    std::mt19937 gen(rd());

    // Random data generation for polynomials
    // poly[i] uses modulus[i % MOD_COUNT]
    vector<vector<TestDataType>> input1(total_polys);
    vector<vector<TestDataType>> input2(total_polys);

    for (int j = 0; j < total_polys; j++)
    {
        int mod_idx = j % MOD_COUNT;
        TestDataType minNumber = 0;
        TestDataType maxNumber = parameters_list[mod_idx].modulus.value - 1;
        std::uniform_int_distribution<TestDataType> dis(minNumber, maxNumber);

        for (int i = 0; i < N; i++)
        {
            input1[j].push_back(dis(gen));
            input2[j].push_back(dis(gen));
        }
    }

    // Performing CPU NTT for each polynomial with its corresponding modulus
    vector<vector<TestDataType>> ntt_mult_result(total_polys);
    for (int i = 0; i < total_polys; i++)
    {
        int mod_idx = i % MOD_COUNT;
        vector<TestDataType> ntt_input1 = generators[mod_idx].ntt(input1[i]);
        vector<TestDataType> ntt_input2 = generators[mod_idx].ntt(input2[i]);
        vector<TestDataType> output =
            generators[mod_idx].mult(ntt_input1, ntt_input2);
        ntt_mult_result[i] = generators[mod_idx].intt(output);
    }

    // Comparing CPU NTT multiplication results and schoolbook multiplication
    // results
    bool check = true;
    for (int i = 0; i < total_polys; i++)
    {
        int mod_idx = i % MOD_COUNT;
        std::vector<TestDataType> schoolbook_result =
            schoolbook_poly_multiplication<TestDataType>(
                input1[i], input2[i], parameters_list[mod_idx].modulus,
                ReductionPolynomial::X_N_minus);

        check = check_result(ntt_mult_result[i].data(), schoolbook_result.data(),
                             N);
        if (!check)
        {
            cout << "FAILED (in poly " << i << ", modulus " << mod_idx << ")"
                 << endl;
            break;
        }

        if ((i == (total_polys - 1)) && check)
        {
            cout << "All Correct - CPU 4-Step RNS NTT with " << MOD_COUNT
                 << " moduli verified." << endl;
        }
    }

    return EXIT_SUCCESS;
}
