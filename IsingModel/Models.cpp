//
// Created by Lukasz on 26.04.2021.
//

#include "Models.h"



void writeSingleConfiguration(const std::vector<int> &spins, const std::string &fileName) {
    /**
     * Helper for tests
     */
    std::ofstream file{fileName, std::ios::app};
    if (!file)
        std::cerr <<  fileName << "could not be opened for writing!\n";

    for (const auto &spin : spins)
        file << spin << "\n";
    file.close();
}

void writeConfigurations(const std::vector<int> &spins, double T, std::ofstream &file, const std::string &separator){
    /**
     * Write only spin configurations for given temperature in one row
     */
    file << T << separator;
    for (const auto &spin : spins)
        file << spin << separator;
    file << "\n";
}

void writeData(const std::vector<int> &spins,
               double magnetization,
               double T,
               std::ofstream &file,
               const std::string &separator){
    /**
     * write data in the following configuration:
     * T <separator> M <separator> spin[i]...spin[size] \n
     */
    file << T << separator << magnetization << separator;
    for (const auto &spin : spins)
        file << spin << separator;
    file << "\n";
}


void initNeighbors(std::vector<int> &Right,
                   std::vector<int> &Left,
                   std::vector<int> &Up,
                   std::vector<int> &Down,
                   int L) {
    /**
     * Initialize four arrays with indices for the neighbors of every element in the matrix.
     * Everything is adjusted to the location on the 2D matrix, but squeezed to separated 1D arrays
     * (reason: possibly improving performance!)
     * Implemented for periodic boundary conditions
     *
     * Example:
     * For 3 x 3 matrix:
     * | 0 | 1 | 2 |
     * -------------
     * | 3 | i | 5 |
     * -------------
     * | 6 | 7 | j |
     *
     * numbers = indices in the squeezed matrix (array)
     * the element with the index "i" has:
     * -> array[1] -- Upper neighbor Up[i] = array[1]
     * -> array[7] -- Lower neighbor Down[i] = array[7]
     * -> array[3] -- Left neighbor Left[i] = array[3]
     * -> array[5] -- Right neighbor Right[i] = array[5]
     *
     * According to periodic boundaries, the element array[j] has:
     * -> Up[j] = array[5]
     * -> Down[j] = array[2]
     * -> Left[j] = array[7]
     * -> Right[j] = array[6]
     *
     */
     int size = L*L;

    for (int i = 0; i < size; ++i) {
        Right[i] = i + 1;
        Left[i] = i - 1;
        Up[i] = i - L;
        Down[i] = i + L;
    }
    // correct the boundaries
    for (int i = 0; i < size; ++i) {
        if ((i % L - (L - 1)) == 0)
            Right[i] = i - (L - 1); // right border
        if ((i % L) == 0)
            Left[i] = i + (L - 1); // left border
        if (i < L)
            Up[i] = i + L * (L - 1); // top border
        if (i >= ((L - 1) * L))
            Down[i] = i - L * (L - 1); // bottom border
    }
}

namespace MetropolisRSU {
/** ************************************************************************
 *
 * Model implementation metropolis according to Random Sequential Updating
 *
 * *************************************************************************
 * */

    void initState(std::vector<int> &spins, pcg64 &rng, std::uniform_int_distribution<int> &choice) {
        /** Randomly initialize array of spins with the values {-1,1} **/
        for (int i{0}; i < spins.size(); ++i)
            spins[i] = getRandomChoice(rng, choice);
    }

    std::array<double, 5> calculateBoltzmannCoeff(double T) {
        /**
         * Calculate Boltzmann coefficient: w = exp(-dE / T) for every possible
         * value of energy change <delta> E in 2D Ising model
         */
        std::array<double, 5> states{std::exp(8.0 / T),
                                     std::exp(4.0 / T),
                                     1.0,                   // for dE=0
                                     std::exp(-4.0 / T),
                                     std::exp(-8.0 / T)};
        return states;
    }

    double getBoltzmannCoeff(const std::array<double, 5> &boltzmanCoeffs, int dE) {
        switch (dE) {
            case -8:
                return boltzmanCoeffs[0];
            case -4:
                return boltzmanCoeffs[1];
            case 0:
                return boltzmanCoeffs[2];
            case 4:
                return boltzmanCoeffs[3];
            case 8:
                return boltzmanCoeffs[4];
            default:
                throw std::exception("Something's wrong with calculating energy of the system!");
        }
    }

    void updateSpin(int position,
                    int dE,
                    std::vector<int> &spins,
                    const std::vector<int> &next,
                    const std::vector<int> &previous,
                    const std::vector<int> &up,
                    const std::vector<int> &down,
                    pcg64 &rng,
                    std::uniform_real_distribution<double> &realDist,
                    const std::array<double, 5> &boltzmannCoeffs) {
        /**
         *  make update of the spin
         */

        dE = 2 * spins[position] * (spins[previous[position]] + spins[next[position]] + spins[up[position]] + spins[down[position]]);
        if (dE <= 0) {
            spins[position] = -spins[position];
        } else {
            if (realDist(rng) < getBoltzmannCoeff(boltzmannCoeffs, dE))
                spins[position] = -spins[position];
        }
    }

    void
    monteCarloStep(int size,
                   std::vector<int> &spins,
                   const std::vector<int> &next,
                   const std::vector<int> &previous,
                   const std::vector<int> &up,
                   const std::vector<int> &down,
                   pcg64 &rng, std::uniform_real_distribution<double> &realDist,
                   std::uniform_int_distribution<int> &intDist,
                   const std::array<double, 5> &boltzmannCoeffs,
                   double &m) {
        /**
         * The metropolis algorithm version 2: Random sequential update
         * Calculate magnetization of the system
         */
        int dE{};  // the change of energy of the system
        m = 0.0; // magnetization (average spin)
        int position;

        for (int i = 0; i < size; ++i) {
            position = intDist(rng);
            updateSpin(position, dE, spins, next, previous, up, down, rng, realDist, boltzmannCoeffs);
            m += spins[i];
        }
        m = m/size;
    }

    void thermalize(std::vector<int> &spins,
                    const std::vector<int> &next,
                    const std::vector<int> &previous,
                    const std::vector<int> &up,
                    const std::vector<int> &down,
                    int warmingTime,
                    pcg64 &rng,
                    std::uniform_real_distribution<double> &realDist,
                    std::uniform_int_distribution<int> &intDist,
                    const std::array<double, 5> &boltzmannCoeffs) {
        /**
          * Thermalization is the monte-carlo step repeated over warmingTime value (without calculating quantites)
          * for stabilization of the system.
          *
          * The metropolis algorithm version 2: Random sequential updating
          */
        int dE{};
        int position;

        for (int k{0}; k < warmingTime; ++k) {
            position = intDist(rng);
            updateSpin(position, dE, spins, next, previous, up, down, rng, realDist, boltzmannCoeffs);

        }
    }

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
                   const std::array<double, 5> &boltzmannCoeffs) {
        /**
         * The metropolis algorithm version 2: Random sequential update
         * Overloaded to calculate nothing from averages
         */
        int dE{};  // the change of energy of the system
        int position;

        for (int i = 0; i < size; ++i) {
            position = intDist(rng);
            updateSpin(position, dE, spins, next, previous, up, down, rng, realDist, boltzmannCoeffs);
        }
    }

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
             pcg64 &rng) {
        /**
         * The overloaded function for collecting average magnetization for given Temperature --> algorithm ver 2
         * returns average magnetization for given temperature.
         */
        double m;
        double magnetizations = 0.0;

        // init
        initState(spins, rng, choice);

        // Prepare equilibrium - thermalize the model
        thermalize(spins, next, previous, up, down, warmingTime, rng, realDist, intDist, boltzmannCoeffs);

        for (int i = 0; i <= MCS; ++i) {
            monteCarloStep(size, spins, next, previous, up, down, rng, realDist, intDist, boltzmannCoeffs, m);
            if (i % takeEvery == 0)
                magnetizations += std::abs(m);
        }
        return magnetizations / (MCS/takeEvery);
    }

    void
    simulate(int size,
             std::vector<int> &spins,
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
             pcg64 &rng) {
        /**
         * The overloaded function for generating only configurations --> algorithm ver 2
         * Writes only one configuration after all monte carlo steps
         *
         */
        // init
        initState(spins, rng, choice);

        // Prepare equilibrium - thermalize the model
        thermalize(spins, next, previous, up, down, warmingTime, rng, realDist, intDist, boltzmannCoeffs);

        for (int i = 0; i <= MCS; ++i)
            monteCarloStep(size, spins, next, previous, up, down, rng, realDist, intDist, boltzmannCoeffs);
    }

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
             const std::string &separator) {
        /**
         * The overloaded function for generating only configurations --> algorithm ver 2
         * Writes configurations sampled by monte carlo steps
         *
         */
        // init
        initState(spins, rng, choice);

        // Prepare equilibrium - thermalize the model
        thermalize(spins, next, previous, up, down, warmingTime, rng, realDist, intDist, boltzmannCoeffs);

        for (int i = 0; i <= MCS; ++i){
            monteCarloStep(size, spins, next, previous, up, down, rng, realDist, intDist, boltzmannCoeffs);
            if (i % takeEvery == 0) {
                writeConfigurations(spins, T, file, separator);
            }
        }
    }
}

namespace BoolSpinConfigurations {
/** ************************************************************************
 *
 * Version of the simulation where boolean vector is used
 * Only for returning spin configurations
 *
 * *************************************************************************
 * */
    std::array<double, 5> calculateBoltzmannCoeff(double T) {
        /**
         * Calculate Boltzmann coefficient: w = exp(-dE / T) for every possible
         * value of energy change <delta> E in 2D Boolean model
         */
        std::array<double, 5> states{std::min(1.0, std::exp(8.0 / T)),
                                     std::min(1.0, std::exp(4.0 / T)),
                                     1.0,
                                     std::min(1.0, std::exp(-4.0 / T)),
                                     std::min(1.0, std::exp(-8.0 / T)),
        };
        return states;
    }

    double getBoltzmannCoeff(const std::array<double, 5> &boltzmanCoeffs, int change) {
        switch (change) {
            case 4:
                return boltzmanCoeffs[4];
            case 3:
                return boltzmanCoeffs[3];
            case 2:
                return boltzmanCoeffs[2];
            case 1:
                return boltzmanCoeffs[1];
            case 0:
                return boltzmanCoeffs[0];
            default:
                throw std::exception("Something's wrong with calculating energy of the system!");
        }
    }

    void initState(std::vector<bool> &spins, pcg64 &rng, std::uniform_int_distribution<int> &choice) {
        /** Randomly initialize array of spins with the values {0,1} **/
        for (int i{0}; i < spins.size(); ++i)
            spins[i] = getRandomChoice(rng, choice);
    }

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
                    const std::array<double, 5> &boltzmannCoeffs) {

        sumE = spins[previous[position]] + spins[next[position]] + spins[up[position]] + spins[down[position]];
        int change = spins[position] ? 2 * sumE - 4 : 4 - 2*sumE;
        if (realDist(rng) < getBoltzmannCoeff(boltzmannCoeffs, (change+4)/2)) {
            spins[position] = !spins[position];
        }

    }

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
                   double &m) {
        /**
         * The metropolis algorithm version 1: Iterate over all elements
         * Calculate magnetization of the system
         */
        int neighborEnergySum{};  // the change of energy of the system
        int position;
        m = 0.0; // magnetization (average spin)

        for (position = 0; position < size; ++position) {
            updateSpin(position, neighborEnergySum, spins, next, previous, up, down,rng, realDist,intDist,
            boltzmannCoeffs);

            m += spins[position];
        }
        m = (2*m-size)/size;
    }

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
                   const std::array<double, 5> &boltzmannCoeffs) {
        /**
         * The metropolis algorithm version 2: Random Sequential updating
         * Does not calculate magnetization!!
         */
        int neighborEnergySum{};  // the change of energy of the system
        int position;
        int i;

        for (i = 0; i < size; ++i) {
            position = intDist(rng);
            updateSpin(position, neighborEnergySum, spins, next, previous, up, down,rng, realDist,intDist,
                       boltzmannCoeffs);
        }
    }

    void
    thermalize(int warmingTime, std::vector<bool> &spins,
                   const std::vector<int> &next,
                   const std::vector<int> &previous,
                   const std::vector<int> &up,
                   const std::vector<int> &down,
                   pcg64 &rng,
                   std::uniform_real_distribution<double> &realDist,
                   std::uniform_int_distribution<int> &intDist,
                   const std::array<double, 5> &boltzmannCoeffs) {
        /**
          * Thermalization is the monte-carlo step repeated over warmingTime value
          * (without calculating quantites) for stabilization of the system.
          *
          * The metropolis algorithm version 1
          */
        int neighborEnergySum{};  // the change of energy of the system
        int position;

        for (int i = 0; i < warmingTime; ++i) {
            position = intDist(rng);
            updateSpin(position, neighborEnergySum, spins, next, previous, up, down,rng, realDist,intDist,
                       boltzmannCoeffs);
        }
    }

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
             int takeEvery) {
        /**
         * The overloaded function for collecting average magnetization for given Temperature --> algorithm ver 1
         */
        double m;
        double magnetizations = 0.0;


        // init
        initState(spins, rng, choice);

        // Prepare equilibrium - warmup of the matrix
        thermalize(warmingTime, spins, next, previous, up, down, rng, realDist, intDist, boltzmannCoeffs);
            for (int i = 0; i <= MCS; ++i) {
                monteCarloStep(size, spins, next, previous, up, down, rng, realDist, intDist, boltzmannCoeffs, m);
                if (i % takeEvery == 0)
                    magnetizations += std::abs(m);
            }
            return magnetizations / (MCS/takeEvery); // no need explicit casting if MCS and takeEvery are correct
    }

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
             int warmingTime) {
        /**
         * The overloaded function that doesnt calculate magnetization --> algorithm ver 2
         */
        // init
        initState(spins, rng, choice);

        // Prepare equilibrium - warmup of the matrix
        thermalize(warmingTime, spins, next, previous, up, down, rng, realDist, intDist, boltzmannCoeffs);

        for (int i = 0; i <= MCS; ++i)
            monteCarloStep(size, spins, next, previous, up, down, rng, realDist, intDist, boltzmannCoeffs);
    }

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
             const std::string &separator) {
        /**
         * The overloaded function that doesnt calculate magnetization --> algorithm ver 2
         * Writes only configrations sampled by MCS
         */
        // init
        initState(spins, rng, choice);
//        std::fill(spins.begin(), spins.end(), 0); // choose this for fixed initial state

        // Prepare equilibrium - warmup of the matrix
        thermalize(warmingTime, spins, next, previous, up, down, rng, realDist, intDist, boltzmannCoeffs);

        for (int i = 0; i <= MCS; ++i) {
            monteCarloStep(size, spins, next, previous, up, down, rng, realDist, intDist, boltzmannCoeffs);
            if (i % takeEvery == 0)
                writeConfigurations(spins, T, file, separator);
        }
    }



    void writeData(const std::vector<bool> &spins,
                   double magnetization,
                   double T,
                   std::ofstream &file,
                   const std::string &separator){
        /**
         * write data in the following configuration:
         * T <separator> M <separator> spin[i]...spin[size] \n
         */
        file << T << separator << magnetization << separator;
        for (const auto &spin : spins)
            file << spin << separator;
        file << "\n";
    }

    void writeConfigurations(const std::vector<bool> &spins,
                             double T,
                             std::ofstream &file,
                             const std::string &separator){
        /**
         * Write only spin configurations for given temperature in one row
         */
        file << T << separator;
        for (const auto &spin : spins)
            file << spin << separator;
        file << "\n";
    }
}

