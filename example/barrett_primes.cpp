// Copyright 2025 W. Nathan Hack <nathan.hack@gmail.com>
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: W. Nathan Hack <nathan.hack@gmail.com>
//
// Utility to find NTT-friendly primes for CPU or GPU Barrett reduction.
//
// Usage: barrett_primes <CPU|GPU> <count>
//
// GPU Barrett reduction is limited to 61-bit primes (not 62 as documented)
// CPU Barrett reduction supports up to 62-bit primes
//
// Primes are of the form k * 2^32 + 1, supporting NTT up to 2^32 points.
// For each prime, outputs: prime value, primitive generator g, and notes on
// computing omega (for X^N-1) and psi (for X^N+1).

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <random>

using namespace std;

typedef uint64_t u64;
typedef __uint128_t u128;

// Miller-Rabin primality test
bool is_prime(u64 n) {
    if (n < 2) return false;
    if (n == 2 || n == 3) return true;
    if (n % 2 == 0) return false;

    // Write n-1 as 2^r * d
    u64 d = n - 1;
    int r = 0;
    while ((d & 1) == 0) {
        d >>= 1;
        r++;
    }

    // Witnesses to test
    vector<u64> witnesses = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};

    for (u64 a : witnesses) {
        if (a >= n) continue;

        // Compute a^d mod n
        u128 x = 1;
        u128 base = a;
        u64 exp = d;
        while (exp > 0) {
            if (exp & 1) x = (x * base) % n;
            base = (base * base) % n;
            exp >>= 1;
        }

        if (x == 1 || x == n - 1) continue;

        bool composite = true;
        for (int i = 0; i < r - 1; i++) {
            x = (x * x) % n;
            if (x == n - 1) {
                composite = false;
                break;
            }
        }

        if (composite) return false;
    }

    return true;
}

// Modular exponentiation
u64 mod_exp(u64 base, u64 exp, u64 mod) {
    u128 result = 1;
    u128 b = base;
    while (exp > 0) {
        if (exp & 1) result = (result * b) % mod;
        b = (b * b) % mod;
        exp >>= 1;
    }
    return (u64)result;
}

// Find a primitive root (generator) for prime p
// g is a primitive root if g^((p-1)/2) = -1 (mod p)
int find_generator(u64 p) {
    for (int g = 2; g < 100; g++) {
        if (mod_exp(g, (p - 1) / 2, p) == p - 1) {
            return g;
        }
    }
    return -1;
}

// Count trailing zeros (power of 2 dividing n)
int count_trailing_zeros(u64 n) {
    int count = 0;
    while ((n & 1) == 0) {
        n >>= 1;
        count++;
    }
    return count;
}

struct PrimeInfo {
    u64 prime;
    int generator;
    u64 k;          // prime = k * 2^32 + 1
    int max_logn;   // Maximum supported LOGN
    int bits;       // Bit size of prime
};

// Find primes of form k * 2^32 + 1 in descending order
vector<PrimeInfo> find_primes(int max_bits, int count) {
    vector<PrimeInfo> results;

    u64 base = 1ULL << 32;
    u64 max_prime = (1ULL << max_bits) - 1;
    u64 min_prime = 1ULL << (max_bits - 1);

    // Start from largest k that gives a prime <= max_prime
    u64 max_k = max_prime / base;
    u64 min_k = min_prime / base;

    // Search odd k values (even k gives even prime-1, but we need high power of 2)
    for (u64 k = max_k | 1; k >= min_k && results.size() < (size_t)count; k -= 2) {
        u64 p = k * base + 1;

        if (p > max_prime) continue;
        if (p < min_prime) break;

        if (is_prime(p)) {
            int g = find_generator(p);
            if (g > 0) {
                PrimeInfo info;
                info.prime = p;
                info.generator = g;
                info.k = k;
                info.max_logn = count_trailing_zeros(p - 1);
                info.bits = (int)(log2(p)) + 1;
                results.push_back(info);
            }
        }
    }

    return results;
}

void print_usage() {
    cerr << "Usage: barrett_primes <CPU|GPU> <count>" << endl;
    cerr << endl;
    cerr << "  CPU - Find primes for CPU Barrett reduction (up to 62 bits)" << endl;
    cerr << "  GPU - Find primes for GPU Barrett reduction (up to 61 bits)" << endl;
    cerr << "  count - Number of primes to find" << endl;
    cerr << endl;
    cerr << "Example: barrett_primes GPU 5" << endl;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        print_usage();
        return 1;
    }

    string mode = argv[1];
    int count = atoi(argv[2]);

    if (count <= 0 || count > 100) {
        cerr << "Error: count must be between 1 and 100" << endl;
        return 1;
    }

    int max_bits;
    string mode_desc;

    if (mode == "CPU" || mode == "cpu") {
        max_bits = 62;
        mode_desc = "CPU Barrett reduction (62-bit limit)";
    } else if (mode == "GPU" || mode == "gpu") {
        max_bits = 61;
        mode_desc = "GPU Barrett reduction (61-bit limit)";
    } else {
        cerr << "Error: mode must be CPU or GPU" << endl;
        print_usage();
        return 1;
    }

    cout << "Finding " << count << " NTT-friendly primes for " << mode_desc << endl;
    cout << "Primes are of form k * 2^32 + 1, supporting NTT up to 2^32 points" << endl;
    cout << endl;

    vector<PrimeInfo> primes = find_primes(max_bits, count);

    if (primes.empty()) {
        cerr << "No primes found!" << endl;
        return 1;
    }

    // Print header
    cout << "=== " << primes.size() << " Primes Found ===" << endl;
    cout << endl;

    // Print for X^N + 1 (needs psi, a 2N-th root of unity)
    cout << "For X^N + 1 reduction polynomial:" << endl;
    cout << "  psi = g^((p-1)/(2N)) = g^((p-1)/2^(logn+1))" << endl;
    cout << "  psi^N = -1 (mod p)" << endl;
    cout << endl;

    // Print for X^N - 1 (needs omega, an N-th root of unity)
    cout << "For X^N - 1 reduction polynomial:" << endl;
    cout << "  omega = g^((p-1)/N) = g^((p-1)/2^logn)" << endl;
    cout << "  omega^N = 1 (mod p)" << endl;
    cout << endl;

    // Print C++ style definitions
    cout << "=== C++ Definitions ===" << endl;
    cout << endl;

    for (size_t i = 0; i < primes.size(); i++) {
        const auto& p = primes[i];
        cout << "// Prime " << i << ": " << p.prime << " = " << p.k << " * 2^32 + 1" << endl;
        cout << "// " << p.bits << " bits, generator g=" << p.generator
             << ", max LOGN=" << p.max_logn << endl;
        cout << "{ " << p.prime << "ULL, " << p.generator << " }," << endl;
        cout << endl;
    }

    // Print detailed table
    cout << "=== Detailed Table ===" << endl;
    cout << endl;
    cout << "| # | Prime | Bits | k | Generator | Max LOGN |" << endl;
    cout << "|---|-------|------|---|-----------|----------|" << endl;

    for (size_t i = 0; i < primes.size(); i++) {
        const auto& p = primes[i];
        cout << "| " << i << " | " << p.prime << " | " << p.bits
             << " | " << p.k << " | " << p.generator
             << " | " << p.max_logn << " |" << endl;
    }

    cout << endl;
    cout << "Note: " << mode_desc << endl;
    if (mode == "GPU" || mode == "gpu") {
        cout << "Warning: GPU Barrett reduction does NOT reliably support 62-bit primes" << endl;
        cout << "         despite what the code comments say. Use 61-bit or smaller." << endl;
    }

    return 0;
}
