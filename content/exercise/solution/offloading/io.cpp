// I/O related functions for heat equation solver

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "heat.h"
#include "pngwriter.h"

// Output routine that prints out a picture of the temperature
// distribution.
void write_field(field *temperature, int iter)
{
    char filename[64];

    // The actual write routine takes only the actual data
    // (without boundary layers) so we need to copy an array with that.
    std::vector<double> inner_data(temperature->nx * temperature->ny);
    auto inner_data_iterator = inner_data.begin();
    auto beginning_of_row = temperature->data.begin() + (temperature->ny + 2) + 1;
    for (int i = 0; i < temperature->nx; i++) {
        auto end_of_row = beginning_of_row + temperature->ny;
        std::copy(beginning_of_row, end_of_row, inner_data_iterator);
        inner_data_iterator += temperature->ny;
        beginning_of_row = end_of_row + 2;
    }

    // Write out the data to a png file
    sprintf(filename, "%s_%04d.png", "heat", iter);
    save_png(inner_data.data(), temperature->nx, temperature->ny, filename, 'c');
}

// Read the initial temperature distribution from a file and
// initialize the temperature fields temperature1 and
// temperature2 to the same initial state.
void read_field(field *temperature1, field *temperature2, char *filename)
{
    FILE *fp;
    int nx, ny, ind;

    int nx_local, ny_local, count;

    fp = fopen(filename, "r");
    // Read the header
    count = fscanf(fp, "# %d %d \n", &nx, &ny);
    if (count < 2) {
        fprintf(stderr, "Error while reading the input file!\n");
	exit(-1);
    }

    set_field_dimensions(temperature1, nx, ny);
    set_field_dimensions(temperature2, nx, ny);

    // Allocate arrays (including boundary layers)
    int newSize = (temperature1->nx + 2) * (temperature1->ny + 2);
    temperature1->data.resize(newSize, 0.0);
    temperature2->data.resize(newSize, 0.0);

    // Array from file
    std::vector<double> file_data(nx * ny, 0.0);

    // Read the actual data
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            ind = i * ny + j;
            count = fscanf(fp, "%lf", &file_data[ind]);
        }
    }

    nx_local = temperature1->nx;
    ny_local = temperature1->ny;

    // Copy to the inner part of the full temperature field
    auto temperature_data_iterator = temperature1->data.begin() + (ny_local + 2) + 1;
    auto beginning_of_row = file_data.begin();
    for (int i = 0; i < nx_local; i++) {
        auto end_of_row = beginning_of_row + ny_local;
        std::copy(beginning_of_row, end_of_row, temperature_data_iterator);
        temperature_data_iterator += ny_local + 2;
        beginning_of_row = end_of_row;
    }

    // Set the boundary values
    for (int i = 1; i < nx_local + 1; i++) {
        temperature1->data[i * (ny_local + 2)] = temperature1->data[i * (ny_local + 2) + 1];
        temperature1->data[i * (ny_local + 2) + ny + 1] = temperature1->data[i * (ny_local + 2) + ny];
    }
    for (int j = 0; j < ny + 2; j++) {
        temperature1->data[j] = temperature1->data[ny_local + j];
        temperature1->data[(nx_local + 1) * (ny_local + 2) + j] =
            temperature1->data[nx_local * (ny_local + 2) + j];
    }

    copy_field(temperature1, temperature2);

    fclose(fp);
}
