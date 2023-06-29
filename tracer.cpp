//gcc -o card2 -O3 tracer.cpp -lm -fopenmp

#include <math.h>
#include <stdio.h>
#include <stdlib.h> // card > pixar.ppm
#include <time.h>
#define R return
#define O operator
typedef float F;
typedef int I;

static long int s_time;
static struct tm m_time;

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

static V position = V(0);

bool sevenSegTest(int digit, int segment) {
  switch (segment) {
    case 0:
      return !(digit == 1 || digit == 4);
      break;
    case 1:
      return !(digit == 5 || digit == 6);
      break;
    case 2:
      return digit != 2;
      break;
    case 3:
      return !(digit == 1 || digit == 4 || digit == 7);
      break;
    case 4:
      return digit == 0 || digit == 2 || digit == 6 || digit == 8;
      break;
    case 5:
      return !(digit == 1 || digit == 2 || digit == 3 || digit == 7);
      break;
    case 6:
      return !(digit == 0 || digit == 1 || digit == 7);
      break;
    default:
      return false;
  }
}

F S(V p, I &m) {
  F d = 1e9;
  V f = p;
  f.z = 0;

  char l[] =
             "5_=_"  // A
             "=W=_"  // B
             "=O=W"  // C
             "5O=O"  // D
             "5O5W"  // E
             "5W5_"  // F
             "5W=W"  // G segment
             ;

  int offsets[] = {0, 12, 28, 40};
  for (I j = 0; j < 4; j++) {
    I digit;
    switch (j) {
      case 0:
        digit = m_time.tm_hour / 10;
        break;
      case 1:
        digit = m_time.tm_hour % 10;
        break;
      case 2:
        digit = m_time.tm_min / 10;
        break;
      case 3:
        digit = m_time.tm_min % 10;
        break;
    }
    for (I i = 0; i < 28; i += 4) {
      if (sevenSegTest(digit, i / 4)) {
        V b = V(l[i] - 79 + offsets[j], l[i + 1] - 79) * .5,
          e = V(l[i + 2] - 79 + offsets[j], l[i + 3] - 79) * .5 + b * -1,
          o = f + (b + e * L(-L((b + f * -1) % e / (e % e), 0), 1)) * -1;
        d = L(d, o % o);
      }
    }
  }
  char ll[] = "MSMS" "M[M["; // hhmm separator
  for (I i = 0; i < 8; i += 4) {
    V b = V(ll[i] - 79, ll[i + 1] - 79) * .5,
      e = V(ll[i + 2] - 79, ll[i + 3] - 79) * .5 + b * -1,
      o = f + (b + e * L(-L((b + f * -1) % e / (e % e), 0), 1)) * -1;
    d = L(d, o % o);
  }
  d = sqrtf(d);
  d = powf(powf(d, 8) + powf(p.z, 8), .125) - .5;
  m = 1;
  F r = L(
      -L(B(p, V(-30, -.5, -30), V(30, 18, 30)),
         B(p, V(-25, 17, -25), V(25, 20, 25))),
      B(V(fmodf(fabsf(p.x), 8), p.y, p.z), V(1.5, 18.5, -25), V(6.5, 20, 25)));
  if (r < d)
    d = r, m = 2;
  F s = 9 - p.y;
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
V computeIncidentLight(V sunDirection, V orig, V dir, F tmin, F tmax)
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

void calculateSunPosition() {
  position = V(0, 0, 1); // x+: west, y+: above horizon, z+: north, initial is south, looking at SW

  F x1, y1, z1;

  // apply seasonal daylength
  F declination = m_time.tm_yday + 10; // winter solstice = 22 Dec
  declination /= (365 / 6.283185);
  declination = -23.5 * cosf(declination);
  declination /= (360 / 6.283185);
  y1 = position.y * cosf(declination) + position.z * sinf(declination);
  z1 = -position.y * sinf(declination) + position.z * cosf(declination);
  position.y = y1;
  position.z = z1;

  // calculate sun rotation (degrees)
  F time = (m_time.tm_hour * 15 + m_time.tm_min / 4.0);

  time -= 3 * 15;   // subtract timezone
  time += 36.6;     // add longitude
  time /= (360 / 6.283185); // to radians

  // rotate along y
  x1 = position.x * cosf(time) - position.z * sinf(time);
  z1 = position.x * sinf(time) + position.z * cosf(time);
  position.x = x1;
  position.z = z1;

  // rotate along x
  F angle = 55.1 - 90; // latitude (55.1 N)
  angle /= (360 / 6.283185);

  y1 = position.y * cosf(angle) + position.z * sinf(angle);
  z1 = -position.y * sinf(angle) + position.z * cosf(angle);
  position.y = y1;
  position.z = z1;
}


V T(V o, V d) {
  V h, n, r, t = 1;

  // light sources
  V ls[] = {
    position,
  };
  V cs[] = {
    V(500, 500, 500),
  };
  I w = 1;

  for (I b = 3; b--;) {
    I m = M(o, d, h, n);
    if (!m)
      break;
    if (m == 1) {
      d = d + n * (n % d * -2);
      o = h + d * .1;
      t = t * .2;

      // glowing letters?
      V letter_color;
      I season = ((m_time.tm_mon + 1) % 12) / 3; // 0-11 -> 0-3

      switch (season) {
        case 0: // winter
          letter_color = V(0, 5, 5);
          break;
        case 1: // spring
          letter_color = V(0, 5, 0);
          break;
        case 2: // summer
          letter_color = V(5, 5, 0);
          break;
        case 3: // fall
          letter_color = V(5, 1, 0);
          break;
      }
      r = r + letter_color;
    }
    if (m == 2) {
      if (U() < 0) {
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

        for (I a = 0; a < w; a++) {
          V l = !ls[a], hc = h, nc = n;
          F i = n % l;
          if (i > 0 && M(h + n * .1, l, hc, nc) == 3) {
            V incident = computeIncidentLight(l, o, l, 0, 1e9);
            r = r + t * incident * cs[a] * i;
          }
        }
      }
    }
    if (m == 3) {
      for (I a = 0; a < w; a++) {
        V l = !ls[a];
        r = r + t * computeIncidentLight(l, o, d, 0, 1e9) * cs[a];
      }
      break;
    }
  }
  R r;
}
I main() {
  FILE *f;
  f = fopen("/tmp/pixar.ppm.tmp", "wbx");

  if (!f) return 0;

  s_time = time(NULL);
  localtime_r (&s_time, &m_time);
  calculateSunPosition();

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
      for (p = s; p > 0; p--)
        c = c + T(e, !(g + l * (x - w / 2 + U()) + u * (y - h / 2 + U())));
      c = c * (1. / s) + 14. / 241;
      V o = c + 1;
      c = V(c.x / o.x, c.y / o.y, c.z / o.z);

      // gamma correction
      F gamma = 2.2;
      c = V(powf(c.x, gamma), powf(c.y, gamma), powf(c.z, gamma)) * 255;

      fprintf(f, "%c%c%c", (I)c.x, (I)c.y, (I)c.z);
    }
  }
  fclose(f);

  rename("/tmp/pixar.ppm.tmp", "/tmp/pixar.ppm");
} // Andrew Kensler
