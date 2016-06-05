 /* squint.i */

%module squint
%{
	#include "squint.h"
%}


const double pi = 3.141592653589793238462643383279502884L;

double lnerfc(double x);

double lnerfd(double l, double r);

double lnevidence(double R, double V);