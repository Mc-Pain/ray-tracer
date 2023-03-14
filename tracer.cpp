#include <math.h>
#include <stdio.h>
#include <stdlib.h> // card > pixar.ppm
#include <omp.h>
#define R return
#define O operator
typedef float F;
typedef int I;
struct V {
  F x, y, z;
  V(F v = 0) { x = y = z = v; }
  V(F a, F b, F c = 0) {
    x = a;
    y = b;
    z = c;
  }
  V O + (V r) { R V(x + r.x, y + r.y, z + r.z); }
  V O *(V r) { R V(x * r.x, y * r.y, z * r.z); }
  F O % (V r) { R x *r.x + y *r.y + z *r.z; }
  V O !() { R *this *(1 / sqrtf(*this % *this)); }
};
F L(F l, F r) { R l < r ? l : r; }
F U() { R(F) rand() / RAND_MAX; }
F B(V p, V l, V h) {
  l = p + l * -1;
  h = h + p * -1;
  R - L(L(L(l.x, h.x), L(l.y, h.y)), L(l.z, h.z));
}
F S(V p, I &m) {
  F d = 1e9;
  V f = p;
  f.z = 0;
  char l[] = "5O5_" "5O=O"        // L
             "COC_" "AOEO" "A_E_" // I
             "IOI_" "I_QO" "QOQ_" // N
             "USU_" "]S]_"        // U
             "aOi_" "a_iO";       // X
  for (I i = 0; i < 48; i += 4) {
    V b = V(l[i] - 79, l[i + 1] - 79) * .5,
      e = V(l[i + 2] - 79, l[i + 3] - 79) * .5 + b * -1,
      o = f + (b + e * L(-L((b + f * -1) % e / (e % e), 0), 1)) * -1;
    d = L(d, o % o);
  }
  d = sqrtf(d);
  V a[] = {V(5, 2)};
  for (I i = 2; i--;) {
    V o = f + a[i] * -1;
    d = L(d, o.y < 0 ? fabsf(sqrtf(o % o) - 2)
                     : (o.x += o.x > 0 ? -2 : 2, sqrtf(o % o)));
  }
  d = powf(powf(d, 8) + powf(p.z, 8), .125) - .5;
  m = 1;
  F r = L(
      -L(B(p, V(-30, -.5, -30), V(30, 18, 30)),
         B(p, V(-25, 17, -25), V(25, 20, 25))),
      B(V(fmodf(fabsf(p.x), 8), p.y, p.z), V(1.5, 18.5, -25), V(6.5, 20, 25)));
  if (r < d)
    d = r, m = 2;
  F s = 19.9 - p.y;
  if (s < d)
    d = s, m = 3;
  R d;
}
I M(V o, V d, V &h, V &n) {
  I m, s = 0;
  F t = 0, c;
  for (; t < 100; t += c)
    if ((c = S(h = o + d * t, m)) < .01 || ++s > 99)
      R n = !V(S(h + V(.01, 0), s) - c, S(h + V(0, .01), s) - c,
               S(h + V(0, 0, .01), s) - c),
        m;
  R 0;
}

// sky light
bool solveQuadratic(F a, F b, F c, F& x1, F& x2)
{
    if (b == 0) {
        // Handle special case where the the two vector ray.dir and V are perpendicular
        // with V = ray.orig - sphere.centre
        if (a == 0) return false;
        x1 = 0; x2 = sqrtf(-c / a);
        return true;
    }
    F discr = b * b - 4 * a * c;

    if (discr < 0) return false;

    F q = (b < 0.f) ? -0.5f * (b - sqrtf(discr)) : -0.5f * (b + sqrtf(discr));
    x1 = q / a;
    x2 = c / q;

    return true;
}
bool raySphereIntersect(const V& orig, const V& dir, const F& radius, F& t0, F& t1)
{
    // They ray dir is normalized so A = 1
    F A = dir.x * dir.x + dir.y * dir.y + dir.z * dir.z;
    F B = 2 * (dir.x * orig.x + dir.y * orig.y + dir.z * orig.z);
    F C = orig.x * orig.x + orig.y * orig.y + orig.z * orig.z - radius * radius;

    if (!solveQuadratic(A, B, C, t0, t1)) return false;

    if (t0 > t1) std::swap(t0, t1);

    return true;
}
V computeIncidentLight(V sunDirection, V& orig, V& dir, F tmin, F tmax)
{
    sunDirection = !sunDirection;
    V betaR(3.8e-6f, 13.5e-6f, 33.1e-6f);
    V betaM(21e-6f);
    F earthRadius = 6360e3,
      atmosphereRadius = 6420e3,
      Hr = 7994,
      Hm = 1200;

    orig.y += earthRadius;
    F t0, t1;
    if (!raySphereIntersect(orig, dir, atmosphereRadius, t0, t1) || t1 < 0) return 0;
    if (t0 > tmin && t0 > 0) tmin = t0;
    if (t1 < tmax) tmax = t1;
    I numSamples = 16;
    I numSamplesLight = 8;
    F segmentLength = (tmax - tmin) / numSamples;
    F tCurrent = tmin;
    V sumR(0), sumM(0); // mie and rayleigh contribution
    F opticalDepthR = 0, opticalDepthM = 0;
    F mu = dir % sunDirection; // mu in the paper which is the cosine of the angle between the sun direction and the ray direction
    F phaseR = 3.f / (16.f * 3.141592) * (1 + mu * mu);
    F g = 0.76f;
    F phaseM = 3.f / (8.f * 3.141592) * ((1.f - g * g) * (1.f + mu * mu)) / ((2.f + g * g) * pow(1.f + g * g - 2.f * g * mu, 1.5f));
    for (I i = 0; i < numSamples; ++i) {
        V samplePosition = orig + V(tCurrent + segmentLength * 0.5f) * dir;
        F height = sqrtf(samplePosition % samplePosition) - earthRadius;
        // compute optical depth for light
        F hr = exp(-height / Hr) * segmentLength;
        F hm = exp(-height / Hm) * segmentLength;
        opticalDepthR += hr;
        opticalDepthM += hm;
        // light optical depth
        F t0Light, t1Light;
        raySphereIntersect(samplePosition, sunDirection, atmosphereRadius, t0Light, t1Light);
        F segmentLengthLight = t1Light / numSamplesLight, tCurrentLight = 0;
        F opticalDepthLightR = 0, opticalDepthLightM = 0;
        I j;
        for (j = 0; j < numSamplesLight; ++j) {
            V samplePositionLight = samplePosition + V(tCurrentLight + segmentLengthLight * 0.5f) * sunDirection;
            F heightLight = sqrtf(samplePositionLight % samplePositionLight) - earthRadius;
            if (heightLight < 0) break;
            opticalDepthLightR += exp(-heightLight / Hr) * segmentLengthLight;
            opticalDepthLightM += exp(-heightLight / Hm) * segmentLengthLight;
            tCurrentLight += segmentLengthLight;
        }
        if (j == numSamplesLight) {
            V tau = betaR * (opticalDepthR + opticalDepthLightR) + betaM * 1.1f * (opticalDepthM + opticalDepthLightM);
            V attenuation(exp(-tau.x), exp(-tau.y), exp(-tau.z));
            sumR = sumR + attenuation * hr;
            sumM = sumM + attenuation * hm;
        }
        tCurrent += segmentLength;
    }

    // [comment]
    // We use a magic number here for the intensity of the sun (20). We will make it more
    // scientific in a future revision of this lesson/code
    // [/comment]
    V ret = (sumR * betaR * phaseR + sumM * betaM * phaseM);
    R ret;
}

V T(V o, V d) {
  V h, n, r, t = 1;
  for (I b = 3; b--;) {
    I m = M(o, d, h, n);
    if (!m)
      break;
    if (m == 1) {
      d = d + n * (n % d * -2);
      o = h + d * .1;
      t = t * .2;
    }
    if (m == 2) {
      if (U() < 0.05) {
        // specular reflection
        d = d + n * (n % d * -2);
        o = h + d * .1;
        t = t * .2;
      } else {
        // diffuse reflection
        F p = 6.283185 * U(), c = U(), s = sqrtf(1 - c),
          g = n.z < 0 ? -1 : 1, u = -1 / (g + n.z), v = n.x * n.y * u;
        d = V(v, g + n.y * n.y * u, -n.y) * (cosf(p) * s) +
            V(1 + g * n.x * n.x * u, g * v, -g * n.x) * (sinf(p) * s) +
            n * sqrtf(c);
        o = h + d * .1;
        t = t * .2;

        V ls[] = {V(.6, .6, 1)};
        V cs[] = {V(500, 400, 100)};
        for (I a = 0; a < 1; a++) {
          V l = !ls[a], hc = h, nc = n;
          F i = n % l;
          if (i > 0 && M(h + n * .1, l, hc, nc) == 3)
            r = r + t * cs[a] * i;
        }
      }
    }
    if (m == 3) {
      r = r + t * computeIncidentLight(V(.6, .6, 1), o, d, 0, 1e9) * V(50, 80, 100);
      break;
    }
  }
  R r;
}
I main() {
  FILE *f;
  f = fopen("pixar.ppm", "wb");

  I w = 1920, h = 1080, s = 4096;
  V e(-22, 5, 25),
      g = !(V(-3, 4, 0) + e * -1), l = !V(g.z, 0, -g.x) * (1. / w),
      u(g.y * l.z - g.z * l.y, g.z * l.x - g.x * l.z, g.x * l.y - g.y * l.x);
  fprintf(f, "P6 %d %d 255 ",
          w, h);
  for (I y = h; y--;) {
    printf("Rendered %d rows out of %d\n", y, h);
    for (I x = w; x--;) {
      V c;
      I p;
#pragma omp parallel for private(p) shared(s, c, e, g, l, x, w, u, y, h)
      for (p = s; p > 0; p--)
        c = c + T(e, !(g + l * (x - w / 2 + U()) + u * (y - h / 2 + U())));
      c = c * (1. / s) + 14. / 241;
      V o = c + 1;
      c = V(c.x / o.x, c.y / o.y, c.z / o.z) * 255;
      fprintf(f, "%c%c%c", (I)c.x, (I)c.y, (I)c.z);
    }
  }
  fclose(f);
} // Andrew Kensler
