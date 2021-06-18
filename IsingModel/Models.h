//
// Created by Lukasz on 26.04.2021.
//

#ifndef ISING2021_MODELS_H
#define ISING2021_MODELS_H

#include "Utils.h"
#include <fstream>
#include <array>
#include <vector>
#include <cmath>


void writeSingleConfiguration(const std::vector<int> &spins, const std::string &fileName);
void writeConfigurations(const std::vector<int> &spins, double T, std::ofstream &file, const std::string &separator);

void writeData(const std::vector<int> &spins,
               double magnetization,
               double T,
               std::ofstream &file,
               const std::string &separator);

void initNeighbors(std::vector<int> &Right,
                   std::vector<int> &Left,
                   std::vector<int> &Up,
                   std::vector<int> &Down,
                   int L);


namespace MetropolisRSU {
    std::array<double, 5> calculateBoltzmannCoeff(double T);
    double getBoltzmannCoeff(const std::array<double, 5> &boltzmanCoeffs, int dE);
    void initState(std::vector<int> &spins, pcg64 &rng, std::uniform_int_distribution<int> &choice);

    void updateSpin(int position,
                    int dE, std::vector<int> &spins,
                    const std::vector<int> &next,
                    const std::vector<int> &previous,
                    const std::vector<int> &up,
                    const std::vector<int> &down,
                    pcg64 &rng,
                    std::uniform_real_distribution<double> &realDist,
                    const std::array<double, 5> &boltzmannCoeffs);

    void monteCarloStep(int size,
                        std::vector<int> &spins,
                        const std::vector<int> &next,
                        const std::vector<int> &previous,
                        const std::vector<int> &up,
                        const std::vector<int> &down,
                        pcg64 &rng,
                        std::uniform_real_distribution<double> &realDist,
                        std::uniform_int_distribution<int> &intDist,
                        const std::array<double, 5> &boltzmannCoeffs,
                        double &m);

    // This one does not calculate magnetizations
    void
    monteCarloStep(int size,
                   std::vector<int> &spins,
                   const std::vector<int> &next,
                   const std::vector<int> &previous,
                   const std::vector<int> &up,
                   const std::vector<int> &down,
                   pcg64 &rng,
                   std::uniform_real_distribution<double> &realDist,
                   std::uniform_int_distribution<int> &intDist,
                   const std::array<double, 5> &boltzmannCoeffs);

    void thermalize(std::vector<int> &spins,
                    const std::vector<int> &next,
                    const std::vector<int> &previous,
                    const std::vector<int> &up,
                    const std::vector<int> &down,
                    int warmingTime,
                    pcg64 &rng,
                    std::uniform_real_distribution<double> &realDist,
                    std::uniform_int_distribution<int> &intDist,
                    const std::array<double, 5> &boltzmannCoeffs);
    double
    simulate(int size,
             std::vector<int> &spins,
             const std::vector<int> &next,
             const std::vector<int> &previous,
             const std::vector<int> &up,
             const std::vector<int> &down,
             int MCS,
             int warmingTime,
             int takeEvery,
             std::uniform_real_distribution<double> &realDist,
             const std::array<double, 5> &boltzmannCoeffs,
             std::uniform_int_distribution<int> &choice,
             std::uniform_int_distribution<int> &intDist,
             pcg64 &rng);

    // This one used only for generating configurations / without calculating m
    void
    simulate(int size, std::vector<int> &spins,
             const std::vector<int> &next,
             const std::vector<int> &previous,
             const std::vector<int> &up,
             const std::vector<int> &down,
             int MCS,
             int warmingTime,
             std::uniform_real_distribution<double> &realDist,
             const std::array<double, 5> &boltzmannCoeffs,
             std::uniform_int_distribution<int> &choice,
             std::uniform_int_distribution<int> &intDist,
             pcg64 &rng);

    void
    simulate(int size,
             std::vector<int> &spins,
             const std::vector<int> &next,
             const std::vector<int> &previous,
             const std::vector<int> &up,
             const std::vector<int> &down,
             int MCS,
             int warmingTime,
             int takeEvery,
             double T,
             std::uniform_real_distribution<double> &realDist,
             const std::array<double, 5> &boltzmannCoeffs,
             std::uniform_int_distribution<int> &choice,
             std::uniform_int_distribution<int> &intDist,
             pcg64 &rng,
             std::ofstream &file,
             const std::string &separator);

}

namespace BoolSpinConfigurations {
    std::array<double, 5> calculateBoltzmannCoeff(double T);
    double getBoltzmannCoeff(const std::array<double, 5> &boltzmanCoeffs, int neighborEnergySum);
    void initState(std::vector<bool> &spins, pcg64 &rng, std::uniform_int_distribution<int> &choice);

    void updateSpin(int position,
                    int sumE,
                    std::vector<bool> &spins,
                    const std::vector<int> &next,
                    const std::vector<int> &previous,
                    const std::vector<int> &up,
                    const std::vector<int> &down,
                    pcg64 &rng,
                    std::uniform_real_distribution<double> &realDist,
                    std::uniform_int_distribution<int> &intDist,
                    const std::array<double, 5> &boltzmannCoeffs);

    void
    monteCarloStep(int size,
                   std::vector<bool> &spins,
                   const std::vector<int> &next,
                   const std::vector<int> &previous,
                   const std::vector<int> &up,
                   const std::vector<int> &down,
                   pcg64 &rng,
                   std::uniform_real_distribution<double> &realDist,
                   std::uniform_int_distribution<int> &intDist,
                   const std::array<double, 5> &boltzmannCoeffs,
                   double &m);

    // This one is only for generating configurations
    void
    monteCarloStep(int size,
                   std::vector<bool> &spins,
                   const std::vector<int> &next,
                   const std::vector<int> &previous,
                   const std::vector<int> &up,
                   const std::vector<int> &down,
                   pcg64 &rng,
                   std::uniform_real_distribution<double> &realDist,
                   std::uniform_int_distribution<int> &intDist,
                   const std::array<double, 5> &boltzmannCoeffs);

    void
    thermalize(int warmingTime,
               std::vector<bool> &spins,
               const std::vector<int> &next,
               const std::vector<int> &previous,
               const std::vector<int> &up,
               const std::vector<int> &down,
               pcg64 &rng,
               std::uniform_real_distribution<double> &realDist,
               std::uniform_int_distribution<int> &intDist,
               const std::array<double, 5> &boltzmannCoeffs);
    double
    simulate(std::vector<bool> &spins,
             const std::vector<int> &next,
             const std::vector<int> &previous,
             const std::vector<int> &up,
             const std::vector<int> &down,
             pcg64 &rng,
             std::uniform_real_distribution<double> &realDist,
             std::uniform_int_distribution<int> &choice,
             std::uniform_int_distribution<int> &intDist,
             const std::array<double, 5> &boltzmannCoeffs,
             int size,
             int MCS,
             int warmingTime,
             int takeEvery);

    // This one is only for calculating configurations
    void
    simulate(std::vector<bool> &spins,
             const std::vector<int> &next,
             const std::vector<int> &previous,
             const std::vector<int> &up,
             const std::vector<int> &down,
             pcg64 &rng,
             std::uniform_real_distribution<double> &realDist,
             std::uniform_int_distribution<int> &choice,
             std::uniform_int_distribution<int> &intDist,
             const std::array<double, 5> &boltzmannCoeffs,
             int size,
             int MCS,
             int warmingTime);

    // This one is only for calculating configurations and writing them every some Monte carlo steps
    void
    simulate(std::vector<bool> &spins,
             const std::vector<int> &next,
             const std::vector<int> &previous,
             const std::vector<int> &up,
             const std::vector<int> &down,
             pcg64 &rng,
             std::uniform_real_distribution<double> &realDist,
             std::uniform_int_distribution<int> &choice,
             std::uniform_int_distribution<int> &intDist,
             const std::array<double, 5> &boltzmannCoeffs,
             int size,
             int MCS,
             int warmingTime,
             int takeEvery,
             double T,
             std::ofstream &file,
             const std::string &separator);

    void writeData(const std::vector<bool> &spins,
                   double magnetization,
                   double T,
                   std::ofstream &file,
                   const std::string &separator);

    void writeConfigurations(const std::vector<bool> &spins,
                             double T,
                             std::ofstream &file,
                             const std::string &separator);
}


#endif //ISING2021_MODELS_H
