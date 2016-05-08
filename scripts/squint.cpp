#include "squint.h"

  long double lnerfc(long double x) {
    assert(x >= 0); // prevent avoidable loss of precision

    if (x <= 20) return log(erfc(x));

    long double x2 = x*x;

    long double cfcs[][2] = {
      // coefficients of continued fraction expansion of
      // \ln(\erfc(x)) + ln(x) around x=\infty
      {-0.5723649429247000870717136756765293558236474064576557858L, -1.0000000000000000000000000000000000000000000000000000000L},
      {-2.5000000000000000000000000000000000000000000000000000000L, -2.0000000000000000000000000000000000000000000000000000000L},
      { 1.1405516982548320510414711953462188027772565209232501407L,  0.32876712328767123287671232876712328767123287671232876712L},
      {-3.3177066788052613518460846864156780048255173002016145217L, -0.6058273370987464749864979602108747639868362120035158253L},
      { 1.1752725769625413012681791551552393251684853690728194177L,  0.15710817791140841439381619395123241498865641242166093227L}
    };

    long double r = INFINITY;
    for (int i = sizeof(cfcs)/sizeof(*cfcs)-1; i>=0; --i) {
      r = cfcs[i][0] + cfcs[i][1]*x2 + 1/r;
    }

    return r - log(x);
  }


  // ln(erf(r) - erf(l))
  long double lnerfd(long double l, long double r) {
    assert(l <= r);

    if (r < 0) {
      return lnerfd(-r, -l);
    } else if (l > 0) {
      // ln(erf(r) - erf(l))
      // = ln(erfc(l) - erfc(r))
      long double hi = lnerfc(l);
      long double lo = lnerfc(r);
      assert(hi >= lo);
      return hi + log1p(-exp(lo-hi));
    } else {
      // ln(2 - erfc(-l) - erfc(r))
      long double hi = lnerfc(-l);
      long double lo = lnerfc(r);
      return log(2 - exp(hi) - exp(lo));
    }
  }

  // \ln \int_0^{1/2} e^{\eta R - \eta^2 V} \dif \eta

  long double lnevidence(long double R, long double V) {
    assert(V >= 0);

    // Special-case small V
    if (V < 1e-10) {
	// First order Taylor expansion around V=0 is
	// log((exp(R/2-V/4)-1)/R)
	//
	// but that is unstable for small R. A better approximation is
	// log((exp(x)-1)/(2*x)) where x = R/2-V/4
      long double x = R/2-V/4;

      if (abs(x) < 1e-10) {
	return x/2 - log(2);
      } else {
	return x < 0
	  ?     log(expm1(x)/(2*x))
	  : x + log(-expm1(-x)/(2*x));
      }
    }

    // okay, R and V are reasonable.
    long double sqV = sqrt(V);
 
    return R*R/(4*V) + log(pi/(4*V))/2 + lnerfd((R-V)/(2*sqV), R/(2*sqV));
  }

int fact(int n) {
    if (n < 0){ /* This should probably return an error, but this is simpler */
        return 0;
    }
    if (n == 0) {
        return 1;
    }
    else {
        /* testing for overflow would be a good idea here */
        return n * fact(n-1);
    }
}
