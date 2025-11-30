// Copyright 2025 W. Nathan Hack <nathan.hack@gmail.com>
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: W. Nathan Hack

// This test verifies CPU 4-step INTT with multiple DISTINCT moduli (RNS-style),
// where polynomials are assigned to moduli in round-robin fashion.
// Uses 3 distinct 61-bit NTT-friendly primes of form k * 2^32 + 1

#include <cstdlib>
#include <random>

#include "ntt.cuh"
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

    cout << "Testing CPU 4-Step RNS INTT with " << MOD_COUNT
         << " DISTINCT moduli, LOGN=" << LOGN << ", BATCH=" << BATCH << endl;

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

    // Create CPU NTT generators for each modulus
    vector<NTT_4STEP_CPU<TestDataType>> generators;
    generators.reserve(MOD_COUNT);
    for (int m = 0; m < MOD_COUNT; m++)
    {
        generators.emplace_back(parameters_list[m]);
    }

    int N = parameters_list[0].n;
    int total_polys = BATCH * MOD_COUNT;

    std::random_device rd;
    std::mt19937 gen(rd());

    TestDataType min_mod = rns_primes[0].prime;
    for (int m = 1; m < MOD_COUNT; m++)
    {
        min_mod = std::min(min_mod, rns_primes[m].prime);
    }
    std::uniform_int_distribution<TestDataType> dis(0, min_mod - 1);

    // Random data generation for polynomials
    vector<vector<TestDataType>> input(total_polys);
    for (int j = 0; j < total_polys; j++)
    {
        for (int i = 0; i < N; i++)
        {
            input[j].push_back(dis(gen));
        }
    }

    // Test 1: NTT -> INTT roundtrip
    cout << "\n=== Test 1: NTT -> INTT Roundtrip ===" << endl;
    bool check = true;
    for (int i = 0; i < total_polys; i++)
    {
        int mod_idx = i % MOD_COUNT;
        vector<TestDataType> ntt_result = generators[mod_idx].ntt(input[i]);
        vector<TestDataType> intt_result = generators[mod_idx].intt(ntt_result);

        check = check_result(intt_result.data(), input[i].data(), N);
        if (!check)
        {
            cout << "FAILED (in poly " << i << ", modulus " << mod_idx << ")"
                 << endl;
            break;
        }

        if ((i == (total_polys - 1)) && check)
        {
            cout << "All Correct - CPU 4-Step RNS NTT->INTT roundtrip with "
                 << MOD_COUNT << " DISTINCT moduli verified." << endl;
        }
    }

    // Test 2: INTT via multiplication verification
    cout << "\n=== Test 2: INTT via Multiplication ===" << endl;

    // Generate second set of polynomials for multiplication
    vector<vector<TestDataType>> input2(total_polys);
    for (int j = 0; j < total_polys; j++)
    {
        for (int i = 0; i < N; i++)
        {
            input2[j].push_back(dis(gen));
        }
    }

    // Performing CPU NTT multiplication for each polynomial
    vector<vector<TestDataType>> ntt_mult_result(total_polys);
    for (int i = 0; i < total_polys; i++)
    {
        int mod_idx = i % MOD_COUNT;
        vector<TestDataType> ntt_input1 = generators[mod_idx].ntt(input[i]);
        vector<TestDataType> ntt_input2 = generators[mod_idx].ntt(input2[i]);
        vector<TestDataType> output =
            generators[mod_idx].mult(ntt_input1, ntt_input2);
        ntt_mult_result[i] = generators[mod_idx].intt(output);
    }

    // Comparing CPU NTT multiplication results and schoolbook multiplication
    check = true;
    for (int i = 0; i < total_polys; i++)
    {
        int mod_idx = i % MOD_COUNT;
        std::vector<TestDataType> schoolbook_result =
            schoolbook_poly_multiplication<TestDataType>(
                input[i], input2[i], parameters_list[mod_idx].modulus,
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
            cout << "All Correct - CPU 4-Step RNS INTT multiplication with "
                 << MOD_COUNT << " DISTINCT moduli verified." << endl;
        }
    }

    return EXIT_SUCCESS;
}
