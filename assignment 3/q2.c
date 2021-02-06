/* File:    omp_trap1.c
 * Purpose: Estimate definite integral (or area under curve) using trapezoidal 
 *          rule.
 *
 * Input:   a, b, n
 * Output:  estimate of integral from a to b of f(x)
 *          using n trapezoids.
 *
 * Compile: gcc -g -Wall -fopenmp -o omp_trap1 omp_trap1.c
 * Usage:   ./omp_trap1 <number of threads>
 *
 * Notes:   
 *   1.  The function f(x) is hardwired.
 *   2.  In this version, each thread explicitly computes the integral
 *       over its assigned subinterval, a critical directive is used
 *       for the global sum.
 *   3.  This version assumes that n is evenly divisible by the 
 *       number of threads
 *
 * IPP:  Section 5.2.1 (pp. 216 and ff.)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

void Usage(char* prog_name);
double f(double x);    /* Function we're integrating */
void Trap(double a, double b, int n, double* global_result_p);
double Trap_manual(double a, double b, int n, double global_result);

int main(int argc, char* argv[]) {
   double  global_result = 0.0;  /* Store result in global_result */
   double  a, b;                 /* Left and right endpoints      */
   double  start, end;           /* Start and end time */
   int     n;                    /* Total number of trapezoids    */
   int     thread_count;

   if (argc != 2) Usage(argv[0]);
   thread_count = strtol(argv[1], NULL, 10);
   printf("Enter a, b, and n\n"); fflush(stdout);
   scanf("%lf %lf %d", &a, &b, &n);
   if (n % thread_count != 0) Usage(argv[0]);

	start = omp_get_wtime();
   /* Original implementation */
#  pragma omp parallel num_threads(thread_count) 
   Trap(a, b, n, &global_result);

	end = omp_get_wtime();

   printf("Original implementation took %f ms\n", 1000 * (end - start));
   printf("With n = %d trapezoids, our estimate\n", n);
   printf("of the integral from %f to %f = %.14e\n\n",
      a, b, global_result);

   /* Manual reduction */
   global_result = 0.0;
	start = omp_get_wtime();
#  pragma omp parallel num_threads(thread_count) 
   {
      double local_result;
      local_result = Trap_manual(a, b, n, global_result);
      global_result += local_result;
   }
   end = omp_get_wtime();

   printf("Manual reduction took %f ms\n", 1000 * (end - start));
   printf("With n = %d trapezoids, our estimate\n", n);
   printf("of the integral from %f to %f = %.14e\n\n",
      a, b, global_result);

   /* Reduction cluse */
   global_result = 0.0;
	start = omp_get_wtime();
#  pragma omp parallel num_threads(thread_count) reduction(+:global_result)
   {
      global_result = Trap_manual(a, b, n, global_result);
   }
   end = omp_get_wtime();

   printf("Reduction cluse took %f ms\n", 1000 * (end - start));
   printf("With n = %d trapezoids, our estimate\n", n);
   printf("of the integral from %f to %f = %.14e\n",
      a, b, global_result);

   return 0;
}  /* main */

/*--------------------------------------------------------------------
 * Function:    Usage
 * Purpose:     Print command line for function and terminate
 * In arg:      prog_name
 */
void Usage(char* prog_name) {

   fprintf(stderr, "usage: %s <number of threads>\n", prog_name);
   fprintf(stderr, "   number of trapezoids must be evenly divisible by\n");
   fprintf(stderr, "   number of threads\n");
   exit(0);
}  /* Usage */

/*------------------------------------------------------------------
 * Function:    f
 * Purpose:     Compute value of function to be integrated
 * Input arg:   x
 * Return val:  f(x)
 */
double f(double x) {
   double return_val;

   return_val = x*x;
   return return_val;
}  /* f */

/*------------------------------------------------------------------
 * Function:    Trap
 * Purpose:     Use trapezoidal rule to estimate definite integral
 * Input args:  
 *    a: left endpoint
 *    b: right endpoint
 *    n: number of trapezoids
 * Output arg:
 *    integral:  estimate of integral from a to b of f(x)
 */
void Trap(double a, double b, int n, double* global_result_p) {
   double  h, x, my_result;
   double  local_a, local_b;
   int  i, local_n;
   int my_rank = omp_get_thread_num();
   int thread_count = omp_get_num_threads();

   h = (b-a)/n; 
   local_n = n/thread_count;  
   local_a = a + my_rank*local_n*h; 
   local_b = local_a + local_n*h; 
   my_result = (f(local_a) + f(local_b))/2.0; 
   for (i = 1; i <= local_n-1; i++) {
     x = local_a + i*h;
     my_result += f(x);
   }
   my_result = my_result*h; 

#  pragma omp critical 
   *global_result_p += my_result; 
}  /* Trap */

double Trap_manual(double a, double b, int n, double global_result) {
   double  h, x, my_result;
   double  local_a, local_b;
   int  i, local_n;
   int my_rank = omp_get_thread_num();
   int thread_count = omp_get_num_threads();

   h = (b-a)/n; 
   local_n = n/thread_count;  
   local_a = a + my_rank*local_n*h; 
   local_b = local_a + local_n*h; 
   my_result = (f(local_a) + f(local_b))/2.0; 
   for (i = 1; i <= local_n-1; i++) {
     x = local_a + i*h;
     my_result += f(x);
   }
   my_result = my_result*h; 

   return my_result;
}  /* Trap */

/*
Original implementation took 0.115428 ms
With n = 1000 trapezoids, our estimate
of the integral from 1.000000 to 100.000000 = 3.33333161716500e+05

Manual reduction took 0.002450 ms
With n = 1000 trapezoids, our estimate
of the integral from 1.000000 to 100.000000 = 3.33333161716500e+05

Reduction cluse took 0.002400 ms
With n = 1000 trapezoids, our estimate
of the integral from 1.000000 to 100.000000 = 3.33333161716500e+05
*/