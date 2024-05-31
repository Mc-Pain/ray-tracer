//gcc -o card2 -O3 tracer.cpp -lm -fopenmp

#include <math.h>
#include <stdio.h>
#include <stdlib.h> // card > pixar.ppm
#include <time.h>

static long int s_time;
static struct tm m_time;

#define GLM_SWIZZLE

#include <glm/glm.hpp>
#include <glm/gtx/rotate_vector.hpp> // for rotating vectors

#define CLOUD_ROWS 16
#define CLOUD_COLS CLOUD_ROWS
#define CLOUD_RESOLUTION (CLOUD_COLS * CLOUD_ROWS)
int clouds[CLOUD_RESOLUTION];

// clouds are 10 km high, each square is 1 km across
float cloudHeight = 1000;
glm::vec3 cloud_color = glm::vec3(0);
bool clouds_initalized = false;
bool inside_letter = false;
bool dispersed = false;

float operator% (glm::vec3 a, glm::vec3 b) {
  return glm::dot(a, b);
}
glm::vec3 operator! (glm::vec3 a) {
  return glm::normalize(a);
}

float L(float l, float r) { return l < r ? l : r; }
float U() { return(float) rand() / RAND_MAX; }
int UI() { return rand(); }
float B(glm::vec3 p, glm::vec3 l, glm::vec3 h) {
  l = p - l;
  h = h - p;
  return -L(L(L(l.x, h.x), L(l.y, h.y)), L(l.z, h.z));
}

static glm::vec3 position = glm::vec3(0);

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

void draw_clock(float &d, glm::vec3 p) {
  glm::vec3 f = p;
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
  for (int j = 0; j < 4; j++) {
    int digit;
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
    for (int i = 0; i < 28; i += 4) {
      if (sevenSegTest(digit, i / 4)) {
        glm::vec3 b = glm::vec3(l[i] - 79 + offsets[j], l[i + 1] - 79, 0) * .5f,
          e = glm::vec3(l[i + 2] - 79 + offsets[j], l[i + 3] - 79, 0) * .5f - b,
          o = f - (b + e * L(-L((b - f) % e / (e % e), 0), 1));
        d = L(d, o % o);
      }
    }
  }
  char ll[] = "MSMS" "M[M["; // hhmm separator
  for (int i = 0; i < 8; i += 4) {
    glm::vec3 b = glm::vec3(ll[i] - 79, ll[i + 1] - 79, 0) * .5f,
      e = glm::vec3(ll[i + 2] - 79, ll[i + 3] - 79, 0) * .5f - b,
      o = f - (b + e * L(-L((b - f) % e / (e % e), 0), 1));
    d = L(d, o % o);
  }
  d = sqrtf(d);
  d = powf(powf(d, 8) + powf(p.z, 8), .125) - .5;
}

void draw_sphere(float &d, glm::vec3 p) {
  glm::vec3 o = glm::vec3(-1, 4, 0) - p;
  float r = sqrtf(o % o) - 4.f;

  // cut the sphere
  if (p.x < -1) {
    glm::vec3 p1 = glm::vec3(0, p.y - 4, p.z);
    float cylinder_distance = sqrtf(p1 % p1) - 4.f;

    glm::vec3 p2 = glm::vec3(p.x + 1, -L(0, -cylinder_distance), 0);
    float semisphere_distance = sqrtf(p2 % p2);

    r = semisphere_distance;
  }

  d = L(d, r);
}

void draw_prism(float &d, glm::vec3 p) {
  float unit = sqrtf(3.f) / 6;

  glm::vec4 plane = glm::vec4(0.f, 0.f, -1.f, -20*unit);
  float d1 = (plane.w + plane.xyz() % p) / sqrtf(plane.xyz() % plane.xyz());

  plane = glm::vec4(6*unit, 0.f, 1.f, -20*unit);
  float d2 = (plane.w + plane.xyz() % p) / sqrtf(plane.xyz() % plane.xyz());

  plane = glm::vec4(-6*unit, 0.f, 1.f, -20*unit);
  float d3 = (plane.w + plane.xyz() % p) / sqrtf(plane.xyz() % plane.xyz());
  float r = -L(-d1, L(-d2, -d3));

  float dh = p.y - 7;
  float dl = 1 - p.y;
  r = -L(-r, L(-dh, -dl));

  d = L(d, r);
}

float S(glm::vec3 p, int &m) {
  float d = 1e9;

  draw_clock(d, p);
  //draw_sphere(d, p);
  //draw_prism(d, p);
  m = 1;

  if (inside_letter)
    d = -d;


  float r = L(
          -L(B(p, glm::vec3(-30, -.5, -30), glm::vec3(30, 18, 30)),
             B(p, glm::vec3(-25, 17, -25), glm::vec3(25, 20, 25))),
          B(glm::vec3(fmodf(fabsf(p.x), 8), p.y, p.z), glm::vec3(1.5, 18.5, -25), glm::vec3(6.5, 20, 25)));
  if (r < d)
    d = r, m = 2;
  float s = 9 - p.y;
  if (s < d)
    d = s, m = 3;
  return d;
}
int M(glm::vec3 o, glm::vec3 d, glm::vec3 &h, glm::vec3 &n) {
  int m, s = 0;
  float t = 0, c;
  for (; t < 100; t += c)
    if ((c = S(h = o + d * t, m)) < .00001 || ++s > 99)
      return n = !glm::vec3(S(h + glm::vec3(.01, 0, 0), s) - c, S(h + glm::vec3(0, .01, 0), s) - c,
                    S(h + glm::vec3(0, 0, .01), s) - c),
             m;
  return 0;
}

// sky light
bool solveQuadratic(float a, float b, float c, float& x1, float& x2)
{
    if (b == 0) {
        // Handle special case where the the two vector ray.dir and glm::vec3 are perpendicular
        // with glm::vec3 = ray.orig - sphere.centre
        if (a == 0) return false;
        x1 = 0; x2 = sqrtf(-c / a);
        return true;
    }
    float discr = b * b - 4 * a * c;

    if (discr < 0) return false;

    float q = (b < 0.f) ? -0.5f * (b - sqrtf(discr)) : -0.5f * (b + sqrtf(discr));
    x1 = q / a;
    x2 = c / q;

    return true;
}
bool raySphereIntersect(const glm::vec3& orig, const glm::vec3& dir, const float& radius, float& t0, float& t1)
{
    // They ray dir is normalized so A = 1
    float A = dir.x * dir.x + dir.y * dir.y + dir.z * dir.z;
    float B = 2 * (dir.x * orig.x + dir.y * orig.y + dir.z * orig.z);
    float C = orig.x * orig.x + orig.y * orig.y + orig.z * orig.z - radius * radius;

    if (!solveQuadratic(A, B, C, t0, t1)) return false;

    if (t0 > t1) std::swap(t0, t1);

    return true;
}
glm::vec3 computeIncidentLight(glm::vec3 sunDirection, glm::vec3 orig, glm::vec3 dir, float tmin, float tmax)
{
    sunDirection = !sunDirection;
    glm::vec3 betaR(3.8e-6f, 13.5e-6f, 33.1e-6f);
    glm::vec3 betaM(21e-6f);
    float earthRadius = 6360e3,
      atmosphereRadius = 6420e3,
      Hr = 7994,
      Hm = 1200;

    orig.y += earthRadius;
    float t0, t1;
    if (!raySphereIntersect(orig, dir, atmosphereRadius, t0, t1) || t1 < 0) return glm::vec3(0.f);
    if (t0 > tmin && t0 > 0) tmin = t0;
    if (t1 < tmax) tmax = t1;
    int numSamples = 16;
    int numSamplesLight = 8;
    float segmentLength = (tmax - tmin) / numSamples;
    float tCurrent = tmin;
    glm::vec3 sumR(0), sumM(0); // mie and rayleigh contribution
    float opticalDepthR = 0, opticalDepthM = 0;
    float mu = dir % sunDirection; // mu in the paper which is the cosine of the angle between the sun direction and the ray direction
    float phaseR = 3.f / (16.f * 3.141592) * (1 + mu * mu);
    float g = 0.76f;
    float phaseM = 3.f / (8.f * 3.141592) * ((1.f - g * g) * (1.f + mu * mu)) / ((2.f + g * g) * pow(1.f + g * g - 2.f * g * mu, 1.5f));
    for (int i = 0; i < numSamples; ++i) {
        glm::vec3 samplePosition = orig + glm::vec3(tCurrent + segmentLength * 0.5f) * dir;
        float height = sqrtf(samplePosition % samplePosition) - earthRadius;
        // compute optical depth for light
        float hr = exp(-height / Hr) * segmentLength;
        float hm = exp(-height / Hm) * segmentLength;
        opticalDepthR += hr;
        opticalDepthM += hm;
        // light optical depth
        float t0Light, t1Light;
        raySphereIntersect(samplePosition, sunDirection, atmosphereRadius, t0Light, t1Light);
        float segmentLengthLight = t1Light / numSamplesLight, tCurrentLight = 0;
        float opticalDepthLightR = 0, opticalDepthLightM = 0;
        int j;
        for (j = 0; j < numSamplesLight; ++j) {
            glm::vec3 samplePositionLight = samplePosition + glm::vec3(tCurrentLight + segmentLengthLight * 0.5f) * sunDirection;
            float heightLight = sqrtf(samplePositionLight % samplePositionLight) - earthRadius;
            if (heightLight < 0) break;
            opticalDepthLightR += exp(-heightLight / Hr) * segmentLengthLight;
            opticalDepthLightM += exp(-heightLight / Hm) * segmentLengthLight;
            tCurrentLight += segmentLengthLight;
        }
        if (j == numSamplesLight) {
            glm::vec3 tau = betaR * (opticalDepthR + opticalDepthLightR) + betaM * 1.1f * (opticalDepthM + opticalDepthLightM);
            glm::vec3 attenuation(exp(-tau.x), exp(-tau.y), exp(-tau.z));
            sumR = sumR + attenuation * hr;
            sumM = sumM + attenuation * hm;
        }
        tCurrent += segmentLength;
    }

    // [comment]
    // We use a magic number here for the intensity of the sun (20). We will make it more
    // scientific in a future revision of this lesson/code
    // [/comment]
    glm::vec3 ret = (sumR * betaR * phaseR + sumM * betaM * phaseM);
    return ret;
}

void calculateSunPosition() {
  const float latitude  = 55.1; // N is positive, S is negative
  const float longitude = 36.6; // E is positive, W is negative
  const float timezone  = 3;    // local difference with UTC, in hours

  position = glm::vec3(0, 0, 1); // x+: west, y+: above horizon, z+: north, initial is south, looking at SW

  float x1, y1, z1;

  // apply seasonal daylength
  float declination = m_time.tm_yday + 10; // winter solstice = 22 Dec
  declination /= (365 / 6.283185);
  declination = -23.5 * cosf(declination);
  declination /= (360 / 6.283185);
  y1 = position.y * cosf(declination) + position.z * sinf(declination);
  z1 = -position.y * sinf(declination) + position.z * cosf(declination);
  position.y = y1;
  position.z = z1;

  // calculate sun rotation (degrees)
  float time = (m_time.tm_hour * 15 + m_time.tm_min / 4.0);

  time -= timezone * 15;    // subtract timezone
  time += longitude;        // add longitude
  time /= (360 / 6.283185); // to radians

  // rotate along y
  x1 = position.x * cosf(time) - position.z * sinf(time);
  z1 = position.x * sinf(time) + position.z * cosf(time);
  position.x = x1;
  position.z = z1;

  // rotate along x
  float angle = latitude - 90;
  angle /= (360 / 6.283185);

  y1 = position.y * cosf(angle) + position.z * sinf(angle);
  z1 = -position.y * sinf(angle) + position.z * cosf(angle);
  position.y = y1;
  position.z = z1;
}

void generateClouds() {
  bool checkerboard = false;
  for (long i = 0; i < CLOUD_RESOLUTION; i++) {
    if (checkerboard) {
      long col = i / CLOUD_COLS;
      long row = i % CLOUD_ROWS;
      clouds[i] = (col + row) % 2;
    } else {
      clouds[i] = UI() % 8 < 1;
    }
  }
}

bool traceClouds(glm::vec3 orig, glm::vec3 dir) {
  //float cloudThickness = 10;

  if (fabsf(dir.y) < 1e-6) {
    // no intersect, assuming dir is parallel to horizon
    return false;
  }

  // where ray intersects cloud level?
  float coeff = (cloudHeight - orig.y) / dir.y;
  if (coeff < 0) {
    // negative distance, no intersect
    return false;
  }

  // wind, clouds are moving 0.1m/s from west to east
  const float seconds = m_time.tm_hour * 60 * 60 + m_time.tm_min * 60;
  glm::vec3 shift = seconds * glm::vec3(0.1f, 0.f, 0.f);

  // point where intersection happens
  glm::vec3 p = shift + orig + (dir * coeff);

  long col = p.x / 100;
  long row = p.z / 100;
  if (p.x < 0) {
    col = CLOUD_COLS - (-col % CLOUD_COLS) - 1;
  }
  if (p.z < 0) {
    row = CLOUD_ROWS - (-row % CLOUD_ROWS) - 1;
  }

  col %= CLOUD_COLS;
  row %= CLOUD_ROWS;

  long i = col * CLOUD_COLS + row;

  if (clouds[i] == 1) {
    return true;
  }

  return false;
}

glm::vec3 generateCloudColor(glm::vec3 sunDirection, glm::vec3 sunColor) {
  int cloud_picks = 256;
  glm::vec3 ret = glm::vec3(0);
  for (int i = 0; i < cloud_picks; i++) {
    // pick a random uniform point from sphere
    float theta = 6.283185 * U();
    float cosphi = 1 - 2*U();
    float sinphi = 1 - cosphi * cosphi;

    float x = sinphi * cos(theta);
    float y = sinphi * sin(theta);
    float z = cosphi;

    glm::vec3 dir = glm::vec3(x, y, z);
    glm::vec3 orig = glm::vec3(0, cloudHeight, 0);

    /*
     * incident light, times:
     * 1) normal
     * 2) sun color
     * 3) 1 / cloud_picks
     * 4) cloud albedo = 0.9
     */
    float normal = glm::vec3(0, 1, 0) % sunDirection;
    glm::vec3 incident = computeIncidentLight(sunDirection, orig, dir, 0, 1e9) + computeIncidentLight(sunDirection, orig, sunDirection, 0, 1e9);
    ret += incident * normal * sunColor / (1.f * cloud_picks);
  }
  return ret;
}

glm::vec3 specular(glm::vec3 direction, glm::vec3 normal) {
  return direction + normal * (normal % direction * -2);
}

glm::vec3 diffuse(glm::vec3 direction, glm::vec3 n) {
  float p = 6.283185 * U(),
        c = U(),
        s = sqrtf(1 - c),
        g = n.z < 0 ? -1 : 1,
        u = -1 / (g + n.z),
        v = n.x * n.y * u;
  return glm::vec3(v, g + n.y * n.y * u, -n.y) * (cosf(p) * s) +
         glm::vec3(1 + g * n.x * n.x * u, g * v, -g * n.x) * (sinf(p) * s) +
         n * sqrtf(c);
}

// n1, n2 indices of refraction of incoming and refracted media
// if n1 > n2, total internal reflection can occur
bool refract(glm::vec3 &direction, glm::vec3 normal, float n1, float n2) {
  if (inside_letter) {
    float temp = n1;
    n2 = n1;
    n1 = temp;
  }
  float r = n1 / n2;
  float cosine1 = -(normal % direction);

  // cosine of refraction angle, squared
  // if negative, total internal reflection occurs, means there is no refraction
  float radicand = 1 - r*r * (1 - cosine1 * cosine1);
  if (radicand < 0) {
    direction = specular(direction, normal);
    return false;
  } else {
    float cosine2 = sqrtf(radicand);

    // apply Fresnel formulas
    float n1cosi = n1 * cosine1;
    float n2cosi = n2 * cosine1;
    float n1cost = n1 * cosine2;
    float n2cost = n2 * cosine2;

    // reflect ratios, s-, p- polarisation
    float rs = (n1cosi - n2cost) / (n1cosi + n2cost);
    float rp = (n2cosi - n1cost) / (n2cosi + n1cost);

    // transmit ratios, s-, p- polarisation
    float ts = 2 * n1cosi / (n1cosi + n2cost);
    float tp = 2 * n1cosi / (n2cosi + n1cost);

    float total = rs + rp + ts + tp;
    if (total * U() < (rs + rp)) {
      direction = specular(direction, normal);
      return false;
    }

    direction = r * direction + (r * cosine1 - cosine2) * normal;
    return true;
  }
}

glm::vec3 calculateAmbientLight(glm::vec3 sunColor, glm::vec3 l, glm::vec3 o, glm::vec3 n, glm::vec3 h) {
  glm::vec3 r = glm::vec3(0);
  float i = n % l;

  if (i > 0) {
    glm::vec3 incident = computeIncidentLight(l, o, l, 0, 1e9);

    float cosTheta = glm::dot(glm::vec3(0, 0, 1), l);
    float sinTheta = 1 - (cosTheta * cosTheta);

    int sun_picks = 16;
    for (int j = 0; j < sun_picks; j++) {
      float radii = sqrt(U());
      float phi = 6.283185 * U();

      // angular diameter of Sun is 30', cos(15') ~= 0.99999, sin(15') ~= 0.00436
      float x = .00436 * cosf(phi);
      float y = .00436 * sinf(phi);
      glm::vec3 dir;

      // time to transform (0, 0, 1) -> (x_l, y_l, z_l)
      if (fabsf(sinTheta) < 0.00001) {
        // l is near zenith, (x, y, 1) will work
        dir = glm::vec3(x, y, 1);
      } else {
        glm::vec3 axis = !glm::vec3(-l.y, l.x, 0);
        dir = glm::rotate(glm::vec3(x, y, 1), acosf(cosTheta), axis);
      }

      glm::vec3 hc = h, nc = n;
      if (M(h + n * .1f, dir, hc, nc) == 3 && !traceClouds(o, dir)) {
        r += incident * sunColor * i / (1.f * sun_picks);
      }
    }
  }
  return r;
}

glm::vec3 T(glm::vec3 o,
            glm::vec3 d,
            glm::vec3 t = glm::vec3(1.f),
            int depth = 3
) {
  glm::vec3 h, n, r = glm::vec3(0.f);

  if (depth < 0)
    return r;

  bool rainbow = false;
  int w;
  glm::vec3 ls[11];
  glm::vec3 cs[11];
  if (rainbow) {
    w = 11;
    glm::vec3 css[] = {
      glm::vec3(500, 0, 0),
      glm::vec3(500, 200, 0),
      glm::vec3(500, 400, 0),
      glm::vec3(400, 500, 0),
      glm::vec3(200, 500, 0),
      glm::vec3(0, 500, 0),
      glm::vec3(0, 500, 200),
      glm::vec3(0, 500, 400),
      glm::vec3(0, 400, 500),
      glm::vec3(0, 200, 500),
      glm::vec3(0, 0, 500),
    };
    for (int i = -5; i <= 5; i++) {
      ls[i+5] = position + (glm::vec3(0, 0.01, 0) * glm::vec3(i));
      cs[i+5] = css[i+5] / 10.f;
    }
  } else {
    // light sources
    ls[0] = {
      position,
    };
    cs[0] = {
      glm::vec3(500, 500, 500),
    };
    w = 1;
  }

  // initialize cloud color
  if (!clouds_initalized) {
    for (int a = 0; a < w; a++) {
      glm::vec3 l = !ls[a];
      cloud_color += generateCloudColor(l, cs[a]);
    }
    clouds_initalized = true;
  }

  int m = M(o, d, h, n);
  if (!m)
    return glm::vec3(0.f);
  if (m == 1) {
    float dispersion = U(); // pick a wavelength: 0 = 760 nm (red), 1 = 380 nm (violet)
    float refractive_index_violet = 1.5337f;
    float refractive_index_red = 1.5116f;
    float refractive_index = refractive_index_red * (1.f - dispersion) + refractive_index_violet * dispersion;

    bool refracted = refract(d, n, 1.f, refractive_index);
    o = h + d * .1f;

    // apply color
    if (refracted && !dispersed) {
      float wavelength = 760.f - 380 * dispersion;
      float red = 0.f;
      float green = 0.f;
      float blue = 0.f;
      if (wavelength > 580) { // red-yellow
        red = 1.f;
        green = (760 - wavelength) / (760 - 580);
      } else if (wavelength > 530) { // yellow-green
        red = 1 - (580 - wavelength) / (580 - 530);
        green = 1.f;
      } else if (wavelength > 470) { // green-cyan
        green = 1.f;
        blue = (530 - wavelength) / (530 - 470);
      } else if (wavelength > 450) { //cyan-blue
        green = 1 - (470 - wavelength) / (470 - 450);
        blue = 1.f;
      } else { // blue-violet
        blue = 1.f;
        red = 0.5 * (450 - wavelength) / (450 - 380);
      }
      glm::vec3 refracted_ray = glm::vec3(red, green, blue);
      t *= refracted_ray;
      dispersed = true;
    }

    // glowing letters?
    glm::vec3 letter_color;
    int season = ((m_time.tm_mon + 1) % 12) / 3; // 0-11 -> 0-3

    switch (season) {
      case 0: // winter
        letter_color = glm::vec3(0, 5, 5);
        break;
      case 1: // spring
        letter_color = glm::vec3(0, 5, 0);
        break;
      case 2: // summer
        letter_color = glm::vec3(5, 5, 0);
        break;
      case 3: // fall
        letter_color = glm::vec3(5, 1, 0);
        break;
    }
    if (refracted) {
      inside_letter = !inside_letter;
      depth += 1;
      //t *= !letter_color;
    }
    //r = r + !letter_color;
    for (int a = 0; a < w; a++) {
      glm::vec3 l = !ls[a];
      r += t * calculateAmbientLight(cs[a], l, o, n, h);
    }
  }
  if (m == 2) {
    if (U() < 0) {
      // specular reflection
      d = specular(d, n);
      o = h + d * .1f;
      t = t * .2f;
    } else {
      // diffuse reflection
      d = diffuse(d, n);
      o = h + d * .1f;
      t = t * .2f;

      for (int a = 0; a < w; a++) {
        glm::vec3 l = !ls[a];
        r += t * calculateAmbientLight(cs[a], l, o, n, h);
      }
    }
  }
  if (m == 3) {
    inside_letter = false;
    dispersed = false;
    for (int a = 0; a < w; a++) {
      glm::vec3 l = !ls[a];
      if (traceClouds(o, d)) {
        r = r + t * cloud_color;
      } else {
        r = r + t * computeIncidentLight(l, o, d, 0, 1e9) * cs[a];
      }
    }
    return r;
  }
  r += T(o, d, t, depth - 1);
  return r;
}
int main() {
  FILE *f;
  f = fopen("/tmp/pixar.ppm.tmp", "wbx");

  if (!f) return 0;

  s_time = time(NULL);
  localtime_r (&s_time, &m_time);
  calculateSunPosition();
  generateClouds();

  int w = 1920, h = 1080, s = 1;
  glm::vec3 e(-22, 5, 25),
      g = !(glm::vec3(-3, 4, 0) - e), l = !glm::vec3(g.z, 0, -g.x) * (1.f / w),
      u(g.y * l.z - g.z * l.y, g.z * l.x - g.x * l.z, g.x * l.y - g.y * l.x);
  fprintf(f, "P6 %d %d 255 ",
          w, h);
  for (int y = h; y--;) {
    printf("Rendered %d rows out of %d\n", y, h);
    for (int x = w; x--;) {
      glm::vec3 c(0.f);
      int p;
      for (p = s; p > 0; p--)
        c = c + T(e, !(g + l * (x - w / 2 + U()) + u * (y - h / 2 + U())));
      c = c * (1.f / s) + 14.f / 241;
      glm::vec3 o = c + 1.f;
      c = glm::vec3(c.x / o.x, c.y / o.y, c.z / o.z);

      // gamma correction
      float gamma = 2.2;
      c = glm::vec3(powf(c.x, gamma), powf(c.y, gamma), powf(c.z, gamma)) * 255.f;

      fprintf(f, "%c%c%c", (int)c.x, (int)c.y, (int)c.z);
    }
  }
  fclose(f);

  rename("/tmp/pixar.ppm.tmp", "/tmp/pixar.ppm");
} // Andrew Kensler
