// Copyright 2025 W. Nathan Hack <nathan.hack@gmail.com>
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: W. Nathan Hack

// This test verifies CPU 4-step INTT (Inverse NTT) functionality,
// ensuring that INTT correctly inverts the NTT operation.

#include <cstdlib> // For atoi or atof functions
#include <fstream>
#include <random>

#include "ntt.cuh"
#include "ntt_4step_cpu.cuh"

#define DEFAULT_MODULUS

using namespace std;
using namespace gpuntt;

int LOGN;
int BATCH;
int N;

// typedef Data32 TestDataType; // Use for 32-bit Test
typedef Data64 TestDataType; // Use for 64-bit Test

int main(int argc, char* argv[])
{
    CudaDevice();

    if (argc < 3)
    {
        LOGN = 12;
        BATCH = 1;
    }
    else
    {
        LOGN = atoi(argv[1]);
        BATCH = atoi(argv[2]);
    }

    cout << "=== Testing CPU 4-Step INTT ===" << endl;
    cout << "LOGN=" << LOGN << ", BATCH=" << BATCH << endl;

    NTTParameters4Step<TestDataType> parameters(LOGN,
                                                ReductionPolynomial::X_N_minus);

    // NTT generator with certain modulus and root of unity
    NTT_4STEP_CPU<TestDataType> generator(parameters);

    N = parameters.n;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uint64_t minNumber = 0;
    std::uint64_t maxNumber = parameters.modulus.value - 1;
    std::uniform_int_distribution<std::uint64_t> dis(minNumber, maxNumber);

    // Random data generation for polynomials (per-polynomial vectors)
    vector<vector<TestDataType>> input1(BATCH);
    vector<vector<TestDataType>> input2(BATCH);
    for (int j = 0; j < BATCH; j++)
    {
        for (int i = 0; i < N; i++)
        {
            input1[j].push_back(dis(gen));
            input2[j].push_back(dis(gen));
        }
    }

    // Test 1: NTT -> INTT roundtrip (per-polynomial processing)
    cout << "\nTest 1: NTT -> INTT Roundtrip" << endl;
    bool check = true;
    for (int b = 0; b < BATCH; b++)
    {
        vector<TestDataType> ntt_result = generator.ntt(input1[b]);
        vector<TestDataType> intt_result = generator.intt(ntt_result);

        check = check_result(intt_result.data(), input1[b].data(), N);
        if (!check)
        {
            cout << "FAILED (in batch " << b << ")" << endl;
            break;
        }
    }
    if (check)
    {
        cout << "All Correct - NTT->INTT roundtrip verified." << endl;
    }

    // Test 2: Verify INTT via multiplication
    cout << "\nTest 2: INTT via Multiplication Verification" << endl;
    check = true;
    for (int b = 0; b < BATCH; b++)
    {
        vector<TestDataType> ntt_input1 = generator.ntt(input1[b]);
        vector<TestDataType> ntt_input2 = generator.ntt(input2[b]);
        vector<TestDataType> output = generator.mult(ntt_input1, ntt_input2);
        vector<TestDataType> ntt_mult_result = generator.intt(output);

        std::vector<TestDataType> schoolbook_result =
            schoolbook_poly_multiplication(input1[b], input2[b],
                                           parameters.modulus,
                                           ReductionPolynomial::X_N_minus);

        check = check_result(ntt_mult_result.data(), schoolbook_result.data(), N);

        if (!check)
        {
            cout << "FAILED (in batch " << b << ")" << endl;
            break;
        }
    }
    if (check)
    {
        cout << "All Correct - INTT multiplication verified." << endl;
    }

    return EXIT_SUCCESS;
}
