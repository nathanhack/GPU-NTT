// Copyright 2025 W. Nathan Hack <nathan.hack@gmail.com>
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: W. Nathan Hack <nathan.hack@gmail.com>

// This test verifies CPU INTT (Inverse NTT) functionality,
// ensuring that INTT correctly inverts the NTT operation.

#include <cstdlib> // For atoi or atof functions
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

#ifdef DEFAULT_MODULUS
    NTTParameters<TestDataType> parameters(LOGN,
                                           ReductionPolynomial::X_N_minus);
#else
    NTTFactors factor((Modulus) 576460752303415297, 288482366111684746,
                      238394956950829);
    NTTParameters parameters(LOGN, factor, ReductionPolynomial::X_N_minus);
#endif

    // NTT generator with certain modulus and root of unity
    NTTCPU<TestDataType> generator(parameters);

    N = parameters.n;

    std::random_device rd;
    std::mt19937 gen(rd());
    TestDataType minNumber = 0;
    TestDataType maxNumber = parameters.modulus.value - 1;
    std::uniform_int_distribution<TestDataType> dis(minNumber, maxNumber);

    cout << "=== Testing CPU INTT ===" << endl;
    cout << "LOGN=" << LOGN << ", BATCH=" << BATCH << endl;

    // Random data generation for polynomials
    vector<vector<TestDataType>> input(BATCH);
    for (int j = 0; j < BATCH; j++)
    {
        for (int i = 0; i < N; i++)
        {
            input[j].push_back(dis(gen));
        }
    }

    // Test 1: NTT -> INTT roundtrip (should recover original)
    cout << "\nTest 1: NTT -> INTT Roundtrip" << endl;
    bool check = true;
    for (int i = 0; i < BATCH; i++)
    {
        vector<TestDataType> ntt_result = generator.ntt(input[i]);
        vector<TestDataType> intt_result = generator.intt(ntt_result);

        check = check_result(intt_result.data(), input[i].data(), N);
        if (!check)
        {
            cout << "FAILED (in " << i << ". Poly.)" << endl;
            break;
        }

        if ((i == (BATCH - 1)) && check)
        {
            cout << "All Correct - NTT->INTT roundtrip verified." << endl;
        }
    }

    // Test 2: Verify INTT via multiplication (NTT mult should equal schoolbook)
    cout << "\nTest 2: INTT via Multiplication Verification" << endl;

    // Generate second set of polynomials for multiplication
    vector<vector<TestDataType>> input2(BATCH);
    for (int j = 0; j < BATCH; j++)
    {
        for (int i = 0; i < N; i++)
        {
            input2[j].push_back(dis(gen));
        }
    }

    // Performing CPU NTT multiplication
    vector<vector<TestDataType>> ntt_mult_result(BATCH);
    for (int i = 0; i < BATCH; i++)
    {
        vector<TestDataType> ntt_input1 = generator.ntt(input[i]);
        vector<TestDataType> ntt_input2 = generator.ntt(input2[i]);
        vector<TestDataType> output = generator.mult(ntt_input1, ntt_input2);
        ntt_mult_result[i] = generator.intt(output);
    }

    // Comparing CPU NTT multiplication results and schoolbook multiplication
    // results
    check = true;
    for (int i = 0; i < BATCH; i++)
    {
        std::vector<TestDataType> schoolbook_result =
            schoolbook_poly_multiplication<TestDataType>(
                input[i], input2[i], parameters.modulus,
                ReductionPolynomial::X_N_minus);

        check = check_result(ntt_mult_result[i].data(),
                             schoolbook_result.data(), N);
        if (!check)
        {
            cout << "FAILED (in " << i << ". Poly.)" << endl;
            break;
        }

        if ((i == (BATCH - 1)) && check)
        {
            cout << "All Correct - INTT multiplication verified." << endl;
        }
    }

    return EXIT_SUCCESS;
}
