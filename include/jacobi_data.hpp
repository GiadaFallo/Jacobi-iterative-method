#ifndef JACOBIDATA_H
#define JACOBIDATA_H

#include <ostream>

class JacobiData{
public:

    int iteration_numbers;
    float error_on_computation;
    float precision;
    int threads;
    int repetitions = 20;
    float* completion_times = new float [repetitions];

    friend std::ostream &operator<<(std::ostream &os, const JacobiData &data){
        for (int i = 0; i < 20; i++) 
            os << data.completion_times[i] << "\t";
        os << data.iteration_numbers << "\t" << data.precision << "\t" << data.error_on_computation << "\t" << data.threads;
        return os;
    }
};

#endif