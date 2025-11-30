// Copyright 2025 W. Nathan Hack <nathan.hack@gmail.com>
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: W. Nathan Hack <nathan.hack@gmail.com>

// This test verifies CPU NTT with multiple DISTINCT moduli (RNS-style),
// where polynomials are assigned to moduli in round-robin fashion.
// This pattern is used in RNS (Residue Number System) based cryptography
// where each polynomial is computed modulo a different prime.
//
// Uses 3 distinct 61-bit NTT-friendly primes of form k * 2^32 + 1

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

// NTT-friendly primes of form k * 2^32 + 1 (~61 bits)
// These are DISTINCT primes for true RNS testing
// Each has a primitive generator g and supports NTT up to 2^32 points
struct PrimeInfo
{
    TestDataType prime;
    TestDataType generator;  // Primitive generator g
    int max_logn;            // Maximum LOGN supported (32 for these primes)
};

// Primitive generators for each prime
// g^((p-1)/2) = -1 mod p (quadratic non-residue)
// NOTE: Library Barrett reduction requires modulus <= 61 bits for Data64
//       (62-bit primes cause overflow in GPU Barrett reduction)
static PrimeInfo rns_primes[MOD_COUNT] = {
    // Prime 0: 2305842949084151809 = 536870898 * 2^32 + 1 (61 bits, g=7)
    {2305842949084151809ULL, 7, 32},
    // Prime 1: 2305842811645198337 = 536870866 * 2^32 + 1 (61 bits, g=3)
    {2305842811645198337ULL, 3, 32},
    // Prime 2: 2305842785875394561 = 536870860 * 2^32 + 1 (61 bits, g=3)
    {2305842785875394561ULL, 3, 32}
};

// Compute omega for X^N - 1 polynomial at given LOGN
// omega is a primitive N-th root of unity: omega^N = 1
// omega = g^((p-1)/N) = g^((p-1)/2^logn)
TestDataType compute_omega(const PrimeInfo& info, int logn)
{
    Modulus<TestDataType> mod(info.prime);
    TestDataType pm1 = info.prime - 1;
    TestDataType exp_val = pm1 >> logn;  // (p-1) / 2^logn
    return OPERATOR<TestDataType>::exp(info.generator, exp_val, mod);
}

// Compute psi for X^N + 1 polynomial at given LOGN
// psi is a primitive 2N-th root of unity: psi^N = -1
// psi = g^((p-1)/(2N)) = g^((p-1)/2^(logn+1))
TestDataType compute_psi(const PrimeInfo& info, int logn)
{
    Modulus<TestDataType> mod(info.prime);
    TestDataType pm1 = info.prime - 1;
    TestDataType exp_val = pm1 >> (logn + 1);  // (p-1) / 2^(logn+1)
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

    // Verify LOGN is supported by all primes
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

    cout << "Testing CPU RNS NTT with " << MOD_COUNT
         << " DISTINCT moduli, LOGN=" << LOGN << ", BATCH=" << BATCH << endl;

    // Create NTTParameters for each RNS modulus with computed omega/psi values
    vector<NTTParameters<TestDataType>> parameters_list;
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

    int N = parameters_list[0].n;

    // Create CPU NTT generators for each modulus
    vector<NTTCPU<TestDataType>> generators;
    generators.reserve(MOD_COUNT);
    for (int m = 0; m < MOD_COUNT; m++)
    {
        generators.emplace_back(parameters_list[m]);
    }

    // Total number of polynomials = BATCH * MOD_COUNT
    int total_polys = BATCH * MOD_COUNT;

    std::random_device rd;
    std::mt19937 gen(rd());

    // Use smallest modulus for random range to ensure valid values for all
    TestDataType min_mod = rns_primes[0].prime;
    for (int m = 1; m < MOD_COUNT; m++)
    {
        min_mod = std::min(min_mod, rns_primes[m].prime);
    }
    std::uniform_int_distribution<TestDataType> dis(0, min_mod - 1);

    // Random data generation for polynomials
    // poly[i] uses modulus[i % MOD_COUNT]
    vector<vector<TestDataType>> input1(total_polys);
    vector<vector<TestDataType>> input2(total_polys);

    for (int j = 0; j < total_polys; j++)
    {
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
                 << " DISTINCT moduli verified." << endl;
        }
    }

    return EXIT_SUCCESS;
}
