/* Copyright (c) 2019 CSC Training */
/* Copyright (c) 2021 ENCCS */
#ifndef PNGWRITER_H_
#define PNGWRITER_H_

#if __cplusplus
  extern "C" {
#endif

int save_png(double *data, const int nx, const int ny, const char *fname,
             const char lang);

#if __cplusplus
  }
#endif
#endif
