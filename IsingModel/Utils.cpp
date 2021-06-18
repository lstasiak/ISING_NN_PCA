//
// Created by Lukasz on 26.04.2021.
//

#include "Utils.h"

int getRandomChoice(pcg64 &rng, std::uniform_int_distribution<int> &dist) {
    /** Get random integer from ~U{-1,1} */
    static const std::vector<int> choices = {-1, 1};
    return choices[dist(rng)];
}


std::string generateFileName(const std::string& Quantity, int L, int MCS, int warmingTime, int saveMode, double T = 0.0, const std::string& format = ".csv") {
    /** Helper fcn for creating name of the data file */
    std::ostringstream name;

    if (T != 0.0) {
        if (!saveMode)
            name << Quantity << "_C_" << "L" << L << "_" << "T" << T << std::setprecision(3) << "_" << "MCS"
            << std::to_string(MCS) << "_WT" << std::to_string(warmingTime) << format;
        else
            name << Quantity << "_A_" << "L" << L << "_" << "T" << T << std::setprecision(3) << "_" << "MCS"
                 << std::to_string(MCS) << "_WT" << std::to_string(warmingTime) << format;
    } else {
        if (!saveMode)
            name << Quantity << "_C_" << "L" << L << "_" << "MCS" << std::to_string(MCS) << "_WT" << std::to_string(warmingTime)<< format;
        else
            name << Quantity << "_A_" << "L" << L << "_" << "MCS" << std::to_string(MCS) << "_WT" << std::to_string(warmingTime)<< format;
    }

    return name.str();
}

