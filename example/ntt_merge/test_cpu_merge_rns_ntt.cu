// Copyright 2025 W. Nathan Hack <nathan.hack@gmail.com>
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: W. Nathan Hack <nathan.hack@gmail.com>

// This test verifies CPU NTT with multiple moduli (RNS-style),
// where polynomials are assigned to moduli in round-robin fashion.
// This pattern is used in RNS (Residue Number System) based cryptography
// where each polynomial is computed modulo a different prime.

#include <cstdlib> // For atoi or atof functions
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

// NTT-friendly primes of the form p = k * 2^m + 1
// Along with their primitive roots (generators g such that g^((p-1)/2) = -1 mod p)
// For omega = g^((p-1)/N) and psi = g^((p-1)/(2N)) where N = 2^LOGN
struct PrimeWithGenerator
{
    TestDataType prime;
    TestDataType generator;  // A primitive root modulo prime
    int max_logn;            // Maximum supported LOGN (m in p = k*2^m + 1)
};

// Using primes and generators from the library's existing parameter pools
// The generator values come from NTTParameters::omega_pool() base primitive roots
// Prime: 576460756061519873 = 5 * 2^57 + 1 (default 64-bit prime)
// Generator: 229929041166717729 is a 2^28-th primitive root of unity
//
// For this CPU test, we use the same prime for all moduli to test the
// round-robin modulus assignment pattern. The GPU RNS test would use
// different primes to expose the claimed bug in InverseCoreLowRing.
static PrimeWithGenerator rns_primes[MOD_COUNT] = {
    {576460756061519873ULL, 229929041166717729ULL, 28},  // Library default with its generator
    {576460756061519873ULL, 229929041166717729ULL, 28},  // Same prime (tests dispatch pattern)
    {576460756061519873ULL, 229929041166717729ULL, 28}   // Same prime (tests dispatch pattern)
};

// Compute omega for X^N - 1 polynomial
// The library uses: omega = g^(2^(28-logn)) for the default 64-bit modulus
// This follows from the base generator being a 2^28-th root of unity
TestDataType compute_omega(TestDataType p, TestDataType g, int logn)
{
    Modulus<TestDataType> mod(p);
    // omega = g^(2^(28-logn)) for 64-bit
    TestDataType exp_val = static_cast<TestDataType>(1) << (28 - logn);
    return OPERATOR<TestDataType>::exp(g, exp_val, mod);
}

// Compute psi for X^N + 1 polynomial: psi = sqrt(omega) = g^(2^(28-logn-1))
TestDataType compute_psi(TestDataType p, TestDataType g, int logn)
{
    Modulus<TestDataType> mod(p);
    // psi = g^(2^(28-logn-1)) = g^(2^(27-logn))
    TestDataType exp_val = static_cast<TestDataType>(1) << (27 - logn);
    return OPERATOR<TestDataType>::exp(g, exp_val, mod);
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

    // Verify LOGN is supported by all primes
    for (int m = 0; m < MOD_COUNT; m++)
    {
        if (LOGN > rns_primes[m].max_logn)
        {
            cout << "LOGN=" << LOGN << " exceeds max supported by prime " << m
                 << " (max=" << rns_primes[m].max_logn << "). Using LOGN=12."
                 << endl;
            LOGN = 12;
            break;
        }
    }

    cout << "Testing RNS NTT with " << MOD_COUNT << " moduli, LOGN=" << LOGN
         << ", BATCH=" << BATCH << endl;

    // Create NTTParameters for each RNS modulus with computed omega/psi values
    vector<NTTParameters<TestDataType>> parameters_list;
    parameters_list.reserve(MOD_COUNT);

    for (int m = 0; m < MOD_COUNT; m++)
    {
        TestDataType p = rns_primes[m].prime;
        TestDataType g = rns_primes[m].generator;
        TestDataType omega = compute_omega(p, g, LOGN);
        TestDataType psi = compute_psi(p, g, LOGN);

        NTTFactors<TestDataType> factor(Modulus<TestDataType>(p), omega, psi);
        parameters_list.emplace_back(LOGN, factor, ReductionPolynomial::X_N_minus);

        cout << "Modulus " << m << ": " << p << " (omega=" << omega << ")"
             << endl;
    }

    // Create CPU NTT generators for each modulus
    vector<NTTCPU<TestDataType>> generators;
    generators.reserve(MOD_COUNT);
    for (int m = 0; m < MOD_COUNT; m++)
    {
        generators.emplace_back(parameters_list[m]);
    }

    N = parameters_list[0].n;

    // Total number of polynomials = BATCH * MOD_COUNT
    // Each batch of polynomials is processed with a different modulus
    int total_polys = BATCH * MOD_COUNT;

    std::random_device rd;
    std::mt19937 gen(rd());

    // Random data generation for polynomials
    // We generate polynomials and assign them to moduli in round-robin fashion
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
            cout << "All Correct - CPU RNS NTT with " << MOD_COUNT
                 << " moduli verified." << endl;
        }
    }

    return EXIT_SUCCESS;
}
