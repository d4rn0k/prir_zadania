#include <stdio.h>

#ifndef HANDLEERROR_H
#define HANDLEERROR_H

#include "mpi.h"
#include "gauss_mpi.h"

static void HandleError( int errorCode, const char * file, unsigned line) {

    if (errorCode != MPI_SUCCESS) {

        char errorString[BUFSIZ];
        int lengthErrorString, errorClass;

        MPI_Error_class(errorCode, &errorClass);
        MPI_Error_string(errorClass, errorString, &lengthErrorString);
        MPI_Error_string(errorCode, errorString, &lengthErrorString);

        fprintf(stdout, "Error! Task: [%d] Class: %s Code: %d at: %s:%d\n", rank, errorString, errorCode, file, line);
        printf("Wyjebało się, a widzimy to przy pomocy printf!\n");
        puts("Wyjebalo się a widzimy to przy pomocy puts!\n");
        MPI_Abort(MPI_COMM_WORLD, errorCode);
    }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


//static void checkCudaError (const char *, unsigned, const char *, cudaError_t);
//#define CUDA_CHECK_RETURN(value) checkCudaError(__FILE__,__LINE__, #value, value)


#endif
