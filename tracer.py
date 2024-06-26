import numpy as np
from PIL import Image

from numba import njit

@njit
def normalize(vector):
    return vector / np.sqrt(np.sum(vector**2))

@njit
def U():
    return np.random.rand(1)[0]

@njit
def L(l, r):
    return min(l, r)

@njit
def B(p, l, h):
    l = p - l
    h = h - p
    return -L(L(L(l[0], h[0]), L(l[1], h[1])), L(l[2], h[2]))

@njit
def draw_sphere(d, p):
  o = np.array([-1, 4, 0]) - p
  r = np.linalg.norm(o) - 4.0
  return L(d, r)

@njit
def S(p, insideLetter):
    d = 1e9
    m = 0

    d = draw_sphere(d, p)
    m = 1

    if insideLetter:
        d = -d

    r = -B(p, np.array([-30, -.5, -30]), np.array([30, 18, 30]))

    if r < d:
        d = r
        m = 2

    s = 9 - p[1]
    if s < d:
        d = s
        m = 3
    return d, m

@njit
def march(cameraPosition, rayDirection, insideLetter):
    m = 0
    h = 0 # hit
    s = 0 # steps
    t = 0.0 # distance
    c = None # step

    while t < 100.0:
        h = cameraPosition + rayDirection * t
        (c, m) = S(h, insideLetter)
        s += 1

        if c < .00001 or s > 99:
            (c1, _) = S(h + np.array([.01, 0, 0]), insideLetter)
            (c2, _) = S(h + np.array([0, .01, 0]), insideLetter)
            (c3, _) = S(h + np.array([0, 0, .01]), insideLetter)
            n = normalize(np.array([c1, c2, c3]) - c)
            return (m, h, n)
        t += c
    return (0, None, None)

@njit
def specular(direction, normal):
    return direction + normal * (np.dot(normal, direction) * -2)

@njit
def diffuse(direction, normal):
    p = U() * 6.283185
    c = U()
    s = np.sqrt(1 - c)
    g = -1 if normal[2] < 0 else 1
    u = -1 / (g + normal[2])
    v = normal[0] * normal[1] * u

    ret = np.array([v, g + normal[1] * normal[1] * u, -normal[1]]) * (np.cos(p) * s)
    ret += np.array([1 + g * normal[0] * normal[0] * u, g * v, -g * normal[0]]) * (np.sin(p) * s)
    ret += normal * np.sqrt(c)
    return ret

@njit
def refract(direction, normal, n1, n2):
    cosine1 = -np.dot(normal, direction)
    ret = 0

    if cosine1 < 0:
        n2, n1 = n1, n2

    r = n1 / n2
    #cosine of refraction angle, squared
    #if negative, total internal reflection occurs, means there is no refraction

    # cos^2 + sin^2 = 1, cos1 is known
    sine1_squared = (1 - cosine1 * cosine1)
    sine1 = np.sqrt(sine1_squared)

    # Snell's law
    sine2 = r * sine1
    cosine2_squared = 1 - sine2 * sine2
    if cosine2_squared < 0:
        # sine2 > 1, total internal reflection
        ret = specular(direction, normal)
        return (False, ret)
    else:
        cosine2 = np.sqrt(cosine2_squared)
        # refract outside, keep sign
        if cosine1 < 0:
            cosine2 = -cosine2

        #apply Fresnel formulas
        n1cosi = n1 * cosine1
        n2cosi = n2 * cosine1
        n1cost = n1 * cosine2
        n2cost = n2 * cosine2

        #reflect ratios, s-, p- polarisation
        rs = (n1cosi - n2cost) / (n1cosi + n2cost)
        rp = (n2cosi - n1cost) / (n2cosi + n1cost)

        #transmit ratios, s-, p- polarisation
        ts = 2 * n1cosi / (n1cosi + n2cost)
        tp = 2 * n1cosi / (n2cosi + n1cost)

        total = rs + rp + ts + tp
        if total * U() < (rs + rp):
            ret = specular(direction, normal)
            return (False, ret)
        else:
            ret = r * direction + (r * cosine1 - cosine2) * normal
            return (True, ret)

@njit
def solveQuadratic(a, b, c):
    if b == 0:
        if a == 0:
            return (False, 0, 0)
        x1 = 0
        x2 = np.sqrt(-c / a)
        return (True, x1, x2)

    discr = b * b - 4 * a * c;

    if discr < 0:
        return (False, 0, 0)

    q = 0
    if b < 0:
        q = -0.5 * (b - np.sqrt(discr))
    else:
        q = -0.5 * (b + np.sqrt(discr))
    x1 = q / a
    x2 = c / q

    return (True, x1, x2)

@njit
def raySphereIntersect(orig, direction, radius):
    # They ray dir is normalized so A = 1

    A = np.dot(direction, direction)
    B = direction[0] * orig[0] + direction[1] * orig[1] + direction[2] * orig[2] * 2.0
    C = orig[0] * orig[0] + orig[1] * orig[1] + orig[2] * orig[2] - radius * radius

    ret, t0, t1 = solveQuadratic(A, B, C)

    if t0 > t1:
        t1, t0 = t0, t1

    return ret, t0, t1


# rayStart is mutable, pass a copy() when invoking
@njit
def incidentLight(sunPosition, rayStart, rayDirection, tmin, tmax):
    betaR = np.array([3.8e-6, 13.5e-6, 33.1e-6])
    betaM = np.array([21e-6, 21e-6, 21e-6])
    earthRadius = 6360e3
    atmosphereRadius = 6420e3
    Hr = 7994
    Hm = 1200

    rayStart[1] += earthRadius
    hasIntersect, t0, t1 = raySphereIntersect(rayStart, rayDirection, atmosphereRadius)
    if not hasIntersect or t1 < 0:
        return np.array([0., 0., 0.])
    if t0 > tmin and t0 > 0:
        tmin = t0
    if t1 < tmax:
        tmax = t1
    numSamples = 16
    numSamplesLight = 8
    segmentLength = (tmax - tmin) / numSamples
    tCurrent = tmin
    sumR = np.array([0., 0., 0.]) # rayleigh contribution
    sumM = np.array([0., 0., 0.]) # mie contribution
    opticalDepthR = 0.0
    opticalDepthM = 0.0
    mu = np.dot(rayDirection, sunPosition) # mu in the paper which is the cosine of the angle between the sun direction and the ray direction
    phaseR = 3. / (16. * 3.141592) * (1 + mu * mu)
    g = 0.76
    phaseM = 3. / (8. * 3.141592) * ((1. - g * g) * (1. + mu * mu)) / ((2. + g * g) * np.power(1. + g * g - 2. * g * mu, 1.5))
    for i in range(numSamples):
        samplePosition = rayStart + (tCurrent + segmentLength * 0.5) * rayDirection
        height = np.linalg.norm(samplePosition) - earthRadius
        # compute optical depth for light
        hr = np.exp(-height / Hr) * segmentLength
        hm = np.exp(-height / Hm) * segmentLength
        opticalDepthR += hr
        opticalDepthM += hm
        # light optical depth
        (_, t0Light, t1Light) = raySphereIntersect(samplePosition, sunPosition, atmosphereRadius)
        segmentLengthLight = t1Light / numSamplesLight
        tCurrentLight = 0.0
        opticalDepthLightR = 0.0
        opticalDepthLightM = 0.0
        for j in range(numSamplesLight):
            samplePositionLight = samplePosition + (tCurrentLight + segmentLengthLight * 0.5) * sunPosition
            heightLight = np.linalg.norm(samplePositionLight) - earthRadius
            if (heightLight < 0):
                break
            opticalDepthLightR += np.exp(-heightLight / Hr) * segmentLengthLight
            opticalDepthLightM += np.exp(-heightLight / Hm) * segmentLengthLight
            tCurrentLight += segmentLengthLight
        else:
            tau = betaR * (opticalDepthR + opticalDepthLightR) + betaM * 1.1 * (opticalDepthM + opticalDepthLightM)
            attenuation = np.exp(-tau)
            sumR += attenuation * hr
            sumM += attenuation * hm

        tCurrent += segmentLength

    # [comment]
    # We use a magic number here for the intensity of the sun (20). We will make it more
    # scientific in a future revision of this lesson/code
    # [/comment]
    ret = (sumR * betaR * phaseR + sumM * betaM * phaseM)
    return ret

@njit
def trace(rayStart, rayDirection, attenuation = 1.0, depth = 3, insideLetter = False):
    if depth < 0:
        return np.array([0.0, 0.0, 0.0])

    hit = normal = ret = np.array([0.0, 0.0, 0.0])

    sunColor = np.array([500.0, 500.0, 500.0])
    sunPosition = normalize(np.array([1.0, 0.05, 1.0]))

    (hitType, hit, normal) = march(rayStart, rayDirection, insideLetter)
    if hitType == 1:
        refracted, rayDirection = refract(rayDirection.copy(), normal.copy(), 1.0, 1.5)
        rayStart = hit + rayDirection * 0.1

        # let the ray out of translucent object
        if refracted == True:
            insideLetter = not insideLetter
            depth += 1

    elif hitType == 2:
        rayDirection = diffuse(rayDirection, normal)
        rayStart = hit + rayDirection * 0.1
        attenuation *= 0.2

    elif hitType == 3:
        sunPositionNorm = normalize(sunPosition)
        ret += sunColor * attenuation * incidentLight(sunPositionNorm, rayStart.copy(), rayDirection, 0.0, 1e9)

    else:
        return np.array([0.0, 0.0, 0.0])

    ret += trace(rayStart, rayDirection, attenuation, depth - 1, insideLetter)
    return ret

@njit(parallel=True)
def render(width, height, steps, outArray, randArray):
    cameraPosition = np.array([-22., 5., 25.])
    cameraCenter = np.array([-3, 4, 0])
    centerVector = normalize(cameraCenter - cameraPosition)
    viewX = normalize(np.array([centerVector[2], 0, -centerVector[0]])) / width
    viewY = np.cross(centerVector, viewX)

    for y in range(height):
        print(f"Rendered {y} rows out of {height}")
        for x in range(width):
            point = np.array([0.0, 0.0, 0.0])
            for step in range(steps):
                rayVector = centerVector + viewX * (width / 2 - x + randArray[y, x, 0])
                rayVector += viewY * (height / 2 - y + randArray[y, x, 1])
                rayDirection = normalize(rayVector)
                point += trace(cameraPosition, rayDirection)

            outArray[y, x] += point

if __name__ == '__main__':
    width = 1920
    height = 1080
    steps = 1

    outArray = np.zeros([height, width, 3])
    #randArray = np.random.rand(width, height, 2)
    randArray = np.zeros([height, width, 2])

    render(width, height, steps, outArray, randArray)

    # post processing
    outArray /= steps
    outArray += 14 / 241
    outArrayCopy = outArray + 1
    outArray /= outArrayCopy
    outArray = (outArray ** 2.2) * 255

    im = Image.fromarray(outArray.astype(np.uint8), 'RGB')
    im.save("pixar.png")
