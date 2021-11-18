// Copyright (c) 2019 CSC Training
// Copyright (c) 2021 ENCCS
// Main routine for heat equation solver in 2D.

#include <stdio.h>
#include <omp.h>

#include "heat.h"

int main(int argc, char **argv)
{
    // Image output interval
    int image_interval = 1500;

    // Number of time steps
    int nsteps;
    // Current and previous temperature fields
    field current, previous;
    initialize(argc, argv, &current, &previous, &nsteps);

    // Output the initial field 
    write_field(&current, 0);

    double average_temp = average(&current);
    printf("Average temperature at start: %f\n", average_temp);

    // Diffusion constant
    double a = 0.5;

    // Compute the largest stable time step
    double dx2 = current.dx * current.dx;
    double dy2 = current.dy * current.dy;
    // Time step
    double dt = dx2 * dy2 / (2.0 * a * (dx2 + dy2));

    // Get the start time stamp
    double start_clock = omp_get_wtime();

    // Time evolution
    for (int iter = 1; iter <= nsteps; iter++) {
        evolve(&current, &previous, a, dt);
        if (iter % image_interval == 0) {
            write_field(&current, iter);
        }
        // Swap current field so that it will be used
        // as previous for next iteration step
        swap_fields(&current, &previous);
    }

    double stop_clock = omp_get_wtime();

    // Average temperature for reference
    average_temp = average(&previous);

    // Determine the CPU time used for all the iterations
    printf("Iterations took %.3f seconds.\n", (stop_clock - start_clock));
    printf("Average temperature: %f\n", average_temp);
    if (argc == 1) {
        printf("Reference value with default arguments: 59.281239\n");
    }

    // Output the final field
    write_field(&previous, nsteps);

    return 0;
}
