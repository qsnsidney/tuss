#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

// Comment out this line to enable debug mode
// #define NDEBUG

/* time stamp function in milliseconds */
__host__ double getTimeStamp()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_usec / 1000000 + tv.tv_sec;
}

int main(int argc, char *argv[])
{
    int error = 0;

    /* Get Dimension */
    /// TODO: Add more arguments for input and output
    if (argc != 2)
    {
        printf("Error: The number of arguments is not exactly 1\n");
        return 0;
    }
    int nBody = atoi(argv[1]);

    return 0;
}