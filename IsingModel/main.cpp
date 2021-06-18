#include "Models.h"
#include "Timer.h"


namespace RandomGenerator {

    // Seed with a real random value, if available
    pcg_extras::seed_seq_from<std::random_device> seed_source;
    // Make a random number engine
//    pcg32 rng(seed_source);
    pcg64 rng(seed_source);
}

int main(int argc, char **argv) {
    double magnetization;
    int takeEvery{};
    int MCS;
    int warmingTime{};
    int L;
    int size;
    double Tmin;
    double Tmax;
    double Tstar;
    double dT;
    int mode;
    int saveData;

    if (argc !=10){
        std::cout<<"Try again. Type in the following order: \n"
                   " 1) L \n"
                   " 2) MCS \n"
                   " 3) warmingTime \n"
                   " 4) takeEvery \n"
                   " 5) Tmin \n"
                   " 6) Tmax \n"
                   " 7) dT \n"
                   " 8) mode \n"
                   " 9) saveData\n";

        std::cout<<"Recommended ranges: L>=10, MCS>=1e5, takeEvery>=0, T=[1.0, 5.0], mode=[0,1], saveData=[0,1] \n"
                   "-----------------------------------------------------------------------------------"
                   "\n mode=0 for bool configuration (0,1), mode=1 for standard Ising (-1,1)\n"
                   "saveData=0 for saving only configurations, saveData=1 for all data"<<std::endl;

        return 0;
    } else {
        std::cout<<"Correct number of values!"<<std::endl;
    }

     //Assign input values
    std::istringstream (argv[1]) >> L;
    std::istringstream (argv[2]) >> MCS;
    std::istringstream (argv[3]) >> warmingTime;
    std::istringstream (argv[4]) >> takeEvery;
    std::istringstream (argv[5]) >> Tmin;
    std::istringstream (argv[6]) >> Tmax;
    std::istringstream (argv[7]) >> dT;
    std::istringstream (argv[8]) >> mode;
    std::istringstream (argv[9]) >> saveData;

    // Set default values
    if(warmingTime == 0) warmingTime = 20000;
    if (MCS == 0) MCS = 200000;
    if (takeEvery == 0) takeEvery = 100;
    if ((mode > 1) || (mode < 0)) mode = 0;
    if ((saveData > 1) || (saveData < 0)) saveData = 0;
    if (Tmin < 0) Tmin = 0.5;
    if (Tmax < 0) Tmax = 4.02;
    if (dT < 0) dT = 0.02;

    std::cout<<"Start simulation for: \n";
    std::cout<<"L = "<<L<<"\n";
    std::cout<<"MCS = "<<MCS<<"\n";
    std::cout<<"warmingTime = "<<warmingTime<<"\n";
    std::cout<<"takeEvery = "<<takeEvery<<"\n";
    std::cout<<"Tmin = "<<Tmin<<"\n";
    std::cout<<"Tmax = "<<Tmax<<"\n";
    std::cout<<"dT = "<<dT<<"\n";
    mode? std::cout<<"mode = Standard Ising\n" : std::cout<<"mode = Binary Spins\n";
    saveData? std::cout<<"saveData = save all data\n" : std::cout<<"saveData = save only spins\n";

    size = L*L;
//    Tmin = 1.02;
//    Tmax = 3.50;
//    Tstar = 2.3;
//    dT = 0.02;

    std::vector<double> Temperatures{};
    std::array<double, 5> boltzmannCoeff{};
    // define TNN
    std::vector<int> previous(size, 0);
    std::vector<int> next(size, 0);
    std::vector<int> up(size, 0);
    std::vector<int> down(size, 0);

    std::uniform_real_distribution<double> realDist{0.0, 1.0};
    std::uniform_int_distribution<int> choices{0, 1};
    std::uniform_int_distribution<int> intDist{0, size-1};

    // initialize neighbors
    initNeighbors(next, previous, up, down, L);

    // fill temperature vector
//    for (double t=Tmax; t > Tstar+0.3; t -= dT) Temperatures.push_back(t);
//    // little densify
//    dT = 0.01;
//    for (double t=Tstar+0.3; t > Tstar-0.3; t -= dT) Temperatures.push_back(t);
//    dT = 0.02;
//    for (double t=Tstar-0.3; t >= Tmin; t -= dT) Temperatures.push_back(t);
    for (double t=Tmax; t > Tmin; t -= dT) Temperatures.push_back(t);
    std::string fileName;


    if (!mode) {
        /***************************************************************
         *  Bool Spin simulation
         *  ************************************************************
         */

        std::vector<bool> spins(size, false); // for bool {0,1} configs

        fileName = generateFileName("DataBool", L, MCS, warmingTime, saveData, 0.0, ".txt");
        std::string separator = " ";
        std::ofstream file{fileName, std::ios::app}; //appending mode
        if (!file)
            std::cerr << "Uh oh, The file could not be opened for writing!\n";

        if (saveData) {  //
            /*************************************************************************************
             *  save magnetization and configurations for given temperature in BoolSpin simulation
             *  **********************************************************************************
             */
            Timer timer;
            for (const auto &T : Temperatures) {
                boltzmannCoeff = BoolSpinConfigurations::calculateBoltzmannCoeff(T);
                magnetization = BoolSpinConfigurations::simulate(spins, next, previous, up, down,
                                                                 RandomGenerator::rng,
                                                                 realDist,
                                                                 choices,
                                                                 intDist,
                                                                 boltzmannCoeff,
                                                                 size,
                                                                 MCS,
                                                                 warmingTime,
                                                                 takeEvery);
                BoolSpinConfigurations::writeData(spins, magnetization, T, file, separator);
                std::cout<<"T="<<T<<" M="<<magnetization<<"\n";
            }
            file.close();
            std::cout<<"Simulations done! Time elapsed: " << timer.elapsed() << " seconds\n";
        }
        else {
            /***********************************************
             *  Save only configurations sampled by MCS
             *  ********************************************
             */
            Timer timer;
            for (const auto &T : Temperatures) {
                boltzmannCoeff = BoolSpinConfigurations::calculateBoltzmannCoeff(T);
                BoolSpinConfigurations::simulate(spins, next, previous, up, down,
                                                                 RandomGenerator::rng,
                                                                 realDist,
                                                                 choices,
                                                                 intDist,
                                                                 boltzmannCoeff,
                                                                 size,
                                                                 MCS,
                                                                 warmingTime,
                                                                 takeEvery,
                                                                 T,
                                                                 file,
                                                                 separator);
//                BoolSpinConfigurations::writeConfigurations(spins, T, file, separator);
                std::cout<<"T="<<T<<"\n";
            }
            file.close();
            std::cout<<"Simulations done! Time elapsed: " << timer.elapsed() << " seconds\n";
        }
    }
    else {
        /*************************************************************************************
          *  save magnetization and configurations for given temperature in IntegerSpin {-1,1} simulation
          *  **********************************************************************************
        */
        std::vector<int> spins(size, 0); // for integer {-1,1} configs

        fileName = generateFileName("Data", L, MCS, warmingTime, saveData, 0.0, ".txt");
        std::string separator = " ";
        std::ofstream file{fileName, std::ios::app}; //appending mode
        if (!file)
            std::cerr << "Uh oh, The file could not be opened for writing!\n";

        if (saveData) { // save magnetization and configurations for given temperature
            /*************************************************************************************
             *  save magnetization and configurations for given temperature
             *  **********************************************************************************
             */
            Timer timer;
            for (const auto &T : Temperatures) {
                boltzmannCoeff = MetropolisRSU::calculateBoltzmannCoeff(T);
                magnetization = MetropolisRSU::simulate(size,
                                                        spins,next, previous, up, down,
                                                        MCS,
                                                        warmingTime,
                                                        takeEvery,
                                                        realDist,
                                                        boltzmannCoeff,
                                                        choices,
                                                        intDist,
                                                        RandomGenerator::rng
                                                        );
                writeData(spins, magnetization, T, file, separator);
                std::cout<<"T="<<T<<" M="<<magnetization<<"\n";
            }
            file.close();
            std::cout<<"Simulations done! Time elapsed: " << timer.elapsed() << " seconds\n";
        }
        else { // save only spin configurations // --> for my master thesis
            /***********************************************
             *  Save only configurations sampled by MCS
             *  ********************************************
             */
            Timer timer;
            for (const auto &T : Temperatures) {
                boltzmannCoeff = MetropolisRSU::calculateBoltzmannCoeff(T);
                MetropolisRSU::simulate(size,
                                        spins,next, previous, up, down,
                                        MCS,
                                        warmingTime,
                                        takeEvery,
                                        T,
                                        realDist,
                                        boltzmannCoeff,
                                        choices,
                                        intDist,
                                        RandomGenerator::rng,
                                        file,
                                        separator
                                        );
//                writeConfigurations(spins, T, file, separator);
                std::cout<<"T="<<T<<"\n";
            }
            file.close();
            std::cout<<"Simulations done! Time elapsed: " << timer.elapsed() << " seconds\n";
        }
    }








    return 0;
}
