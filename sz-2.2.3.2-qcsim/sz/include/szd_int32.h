/**
 *  @file szd_int32.h
 *  @author Sheng Di
 *  @date July, 2017
 *  @brief Header file for the szd_int32.c.
 *  (C) 2016 by Mathematics and Computer Science (MCS), Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#ifndef _SZD_Int32_H
#define _SZD_Int32_H

#ifdef __cplusplus
extern "C" {
#endif

#include "TightDataPointStorageI.h"

#define SZ_INT32_MIN -2147483648
#define SZ_INT32_MAX 2147483647

void decompressDataSeries_int32_1D(int32_t** data, size_t dataSeriesLength, TightDataPointStorageI* tdps);
void decompressDataSeries_int32_2D(int32_t** data, size_t r1, size_t r2, TightDataPointStorageI* tdps);
void decompressDataSeries_int32_3D(int32_t** data, size_t r1, size_t r2, size_t r3, TightDataPointStorageI* tdps);
void decompressDataSeries_int32_4D(int32_t** data, size_t r1, size_t r2, size_t r3, size_t r4, TightDataPointStorageI* tdps);

void getSnapshotData_int32_1D(int32_t** data, size_t dataSeriesLength, TightDataPointStorageI* tdps, int errBoundMode);
void getSnapshotData_int32_2D(int32_t** data, size_t r1, size_t r2, TightDataPointStorageI* tdps, int errBoundMode);
void getSnapshotData_int32_3D(int32_t** data, size_t r1, size_t r2, size_t r3, TightDataPointStorageI* tdps, int errBoundMode);
void getSnapshotData_int32_4D(int32_t** data, size_t r1, size_t r2, size_t r3, size_t r4, TightDataPointStorageI* tdps, int errBoundMode);

int SZ_decompress_args_int32(int32_t** newData, size_t r5, size_t r4, size_t r3, size_t r2, size_t r1, unsigned char* cmpBytes, size_t cmpSize);

#ifdef __cplusplus
}
#endif

#endif /* ----- #ifndef _SZD_Int32_H  ----- */
