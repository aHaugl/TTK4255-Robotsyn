import numpy as np
import numba as nb


@nb.njit('float64[:,:](float64[:])', cache=True, parallel=True)
def rotors_t(array):
    """
    base_x, base_y, base_z, base_p, base_q, base_r, hinge_x, hinge_y, hinge_z, hinge_p, hinge_q, hinge_r, rotors_x, rotors_y, rotors_z, rotors_p, rotors_q, rotors_r
    """

    base_x = array[0]
    base_y = array[1]
    base_z = array[2]
    base_p = array[3]
    base_q = array[4]
    base_r = array[5]
    hinge_x = array[6]
    hinge_y = array[7]
    hinge_z = array[8]
    hinge_p = array[9]
    hinge_q = array[10]
    hinge_r = array[11]
    rotors_x = array[12]
    rotors_y = array[13]
    rotors_z = array[14]
    rotors_p = array[15]
    rotors_q = array[16]
    rotors_r = array[17]

    x0 = np.cos(base_r)
    x1 = np.sin(base_q)
    x2 = 0.0146748733049962*x1
    x3 = np.sin(base_r)
    x4 = np.cos(base_p)
    x5 = 0.448564787567025*x4
    x6 = np.cos(base_q)
    x7 = 0.893629833571167*x6
    x8 = np.sin(base_p)
    x9 = x3*x8
    x10 = 0.893629833571167*x1
    x11 = 0.0146748733049962*x6
    x12 = x0*x8
    x13 = x1*x4
    x14 = x4*x6
    x15 = 0.97250713111939*x1
    x16 = 0.214136510870782*x4
    x17 = 0.0915174007171296*x6
    x18 = 0.0915174007171296*x1
    x19 = 0.97250713111939*x6
    x20 = 0.232410257982334*x1
    x21 = 0.867718379468268*x4
    x22 = 0.43937488084499*x6
    x23 = 0.43937488084499*x1
    x24 = 0.232410257982334*x6

    return np.array([[-x0*x2 + x0*x7 + x10*x9 + x11*x9 - x3*x5, -x0*x5 + x10*x12 + x11*x12 + x2*x3 - x3*x7, 0.893629833571167*x13 + 0.0146748733049962*x14 + 0.448564787567025*x8, 0.893629833571167*base_x - 0.448564787567025*base_y + 0.0146748733049962*base_z - 0.258257036698329],
                     [x0*x15 - x0*x17 - x16*x3 - x18*x9 - x19*x9, -x0*x16 - x12*x18 - x12*x19 - x15*x3 + x17*x3, -0.0915174007171296*x13 - 0.97250713111939 *
                         x14 + 0.214136510870782*x8, -0.0915174007171296*base_x - 0.214136510870782*base_y - 0.97250713111939*base_z + 0.116344254058977],
                     [x0*x20 + x0*x22 + x21*x3 + x23*x9 - x24*x9, x0*x21 + x12*x23 - x12*x24 - x20*x3 - x22*x3, 0.43937488084499*x13 - 0.232410257982334 *
                         x14 - 0.867718379468268*x8, 0.43937488084499*base_x + 0.867718379468268*base_y - 0.232410257982334*base_z + 0.790231880961314],
                     [0., 0., 0., 1.]])


@nb.njit('float64[:,:](float64[:])', cache=True, parallel=True)
def rotors_t_diff(array):
    """
    base_x, base_y, base_z, base_p, base_q, base_r, hinge_x, hinge_y, hinge_z, hinge_p, hinge_q, hinge_r, rotors_x, rotors_y, rotors_z, rotors_p, rotors_q, rotors_r
    """

    base_x = array[0]
    base_y = array[1]
    base_z = array[2]
    base_p = array[3]
    base_q = array[4]
    base_r = array[5]
    hinge_x = array[6]
    hinge_y = array[7]
    hinge_z = array[8]
    hinge_p = array[9]
    hinge_q = array[10]
    hinge_r = array[11]
    rotors_x = array[12]
    rotors_y = array[13]
    rotors_z = array[14]
    rotors_p = array[15]
    rotors_q = array[16]
    rotors_r = array[17]

    x0 = np.sin(base_r)
    x1 = np.sin(base_p)
    x2 = np.cos(base_p)
    x3 = np.sin(base_q)
    x4 = 0.893629833571167*x3
    x5 = np.cos(base_q)
    x6 = 0.0146748733049962*x5
    x7 = 0.448564787567025*x1 + x2*x4 + x2*x6
    x8 = np.cos(base_r)
    x9 = x4*x8
    x10 = x6*x8
    x11 = 0.0146748733049962*x3
    x12 = x0*x11
    x13 = 0.893629833571167*x5
    x14 = x0*x13
    x15 = 0.448564787567025*x2
    x16 = x0*x4
    x17 = x0*x6
    x18 = x11*x8
    x19 = x13*x8
    x20 = 0.0915174007171296*x3
    x21 = 0.97250713111939*x5
    x22 = 0.214136510870782*x1 - x2*x20 - x2*x21
    x23 = x20*x8
    x24 = x21*x8
    x25 = 0.97250713111939*x3
    x26 = x0*x25
    x27 = 0.0915174007171296*x5
    x28 = x0*x27
    x29 = 0.214136510870782*x2
    x30 = x0*x20
    x31 = x0*x21
    x32 = x25*x8
    x33 = x27*x8
    x34 = 0.43937488084499*x3
    x35 = 0.232410257982334*x5
    x36 = -0.867718379468268*x1 + x2*x34 - x2*x35
    x37 = x34*x8
    x38 = x35*x8
    x39 = 0.232410257982334*x3
    x40 = x0*x39
    x41 = 0.43937488084499*x5
    x42 = x0*x41
    x43 = 0.867718379468268*x2
    x44 = x0*x34
    x45 = x0*x35
    x46 = x39*x8
    x47 = x41*x8

    return np.array([[0., 0., 0., x0*x7, -x1*x12 + x1*x14 - x10 - x9, x1*x10 + x1*x9 + x12 - x14 - x15*x8],
                     [0., 0., 0., x7*x8, -x1*x18 + x1*x19 + x16 +
                         x17, x0*x15 - x1*x16 - x1*x17 + x18 - x19],
                     [0., 0., 0., -x1*x4 - x1*x6 + x15, x2*(-x11 + x13), 0.],
                     [0.893629833571167, -0.448564787567025,
                         0.0146748733049962, 0., 0., 0.],
                     [0., 0., 0., x0*x22, x1*x26 - x1*x28 + x23 +
                      x24, -x1*x23 - x1*x24 - x26 + x28 - x29*x8],
                     [0., 0., 0., x22*x8, x1*x32 - x1*x33 - x30 -
                      x31, x0*x29 + x1*x30 + x1*x31 - x32 + x33],
                     [0., 0., 0., x1*x20 + x1*x21 + x29, x2*(x25 - x27), 0.],
                     [-0.0915174007171296, -0.214136510870782, -
                         0.97250713111939, 0., 0., 0.],
                     [0., 0., 0., x0*x36, x1*x40 + x1*x42 - x37 +
                      x38, x1*x37 - x1*x38 - x40 - x42 + x43*x8],
                     [0., 0., 0., x36*x8, x1*x46 + x1*x47 + x44 -
                      x45, -x0*x43 - x1*x44 + x1*x45 - x46 - x47],
                     [0., 0., 0., -x1*x34 + x1*x35 - x43, x2*(x39 + x41), 0.],
                     [0.43937488084499, 0.867718379468268, -
                         0.232410257982334, 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0.]])


@nb.njit('float64[:,:](float64[:])', cache=True, parallel=True)
def arm_t(array):
    """
    base_x, base_y, base_z, base_p, base_q, base_r, hinge_x, hinge_y, hinge_z, hinge_p, hinge_q, hinge_r, rotors_x, rotors_y, rotors_z, rotors_p, rotors_q, rotors_r
    """

    base_x = array[0]
    base_y = array[1]
    base_z = array[2]
    base_p = array[3]
    base_q = array[4]
    base_r = array[5]
    hinge_x = array[6]
    hinge_y = array[7]
    hinge_z = array[8]
    hinge_p = array[9]
    hinge_q = array[10]
    hinge_r = array[11]
    rotors_x = array[12]
    rotors_y = array[13]
    rotors_z = array[14]
    rotors_p = array[15]
    rotors_q = array[16]
    rotors_r = array[17]

    x0 = np.sin(hinge_q)
    x1 = np.cos(hinge_p)
    x2 = x0*x1
    x3 = np.sin(hinge_r)
    x4 = np.sin(hinge_p)
    x5 = np.cos(hinge_q)
    x6 = x4*x5
    x7 = -x2 + x3*x6
    x8 = np.sin(base_p)
    x9 = np.sin(base_q)
    x10 = np.cos(base_p)
    x11 = x10*x9
    x12 = np.cos(base_q)
    x13 = x10*x12
    x14 = 0.893629833571167*x11 + 0.0146748733049962*x13 + 0.448564787567025*x8
    x15 = np.cos(base_r)
    x16 = 0.0146748733049962*x9
    x17 = np.sin(base_r)
    x18 = 0.448564787567025*x10
    x19 = 0.893629833571167*x12
    x20 = 0.893629833571167*x9
    x21 = x17*x8
    x22 = 0.0146748733049962*x12
    x23 = -x15*x16 + x15*x19 - x17*x18 + x20*x21 + x21*x22
    x24 = np.cos(hinge_r)
    x25 = x23*x24
    x26 = x0*x4
    x27 = x1*x5
    x28 = x26 + x27*x3
    x29 = x15*x8
    x30 = -x15*x18 + x16*x17 - x17*x19 + x20*x29 + x22*x29
    x31 = x24*x4
    x32 = x1*x24
    x33 = x26*x3 + x27
    x34 = x14*x33
    x35 = x0*x25
    x36 = -x2*x3 + x6
    x37 = x30*x36
    x38 = 0.0915174007171296*x11 + 0.97250713111939*x13 - 0.214136510870782*x8
    x39 = 0.97250713111939*x9
    x40 = 0.214136510870782*x10
    x41 = 0.0915174007171296*x12
    x42 = 0.0915174007171296*x9
    x43 = 0.97250713111939*x12
    x44 = -x15*x39 + x15*x41 + x17*x40 + x21*x42 + x21*x43
    x45 = x24*x44
    x46 = x15*x40 + x17*x39 - x17*x41 + x29*x42 + x29*x43
    x47 = x33*x38
    x48 = x0*x45
    x49 = x36*x46
    x50 = -0.43937488084499*x11 + 0.232410257982334*x13 + 0.867718379468268*x8
    x51 = 0.232410257982334*x9
    x52 = 0.867718379468268*x10
    x53 = 0.43937488084499*x12
    x54 = 0.43937488084499*x9
    x55 = 0.232410257982334*x12
    x56 = x15*x51 + x15*x53 + x17*x52 + x21*x54 - x21*x55
    x57 = x24*x56
    x58 = -x15*x52 + x17*x51 + x17*x53 - x29*x54 + x29*x55
    x59 = x33*x50
    x60 = x0*x57
    x61 = x36*x58

    return np.array([[x14*x7 + x25*x5 + x28*x30, x14*x31 - x23*x3 + x30*x32, x34 + x35 - x37, 0.893629833571167*base_x - 0.448564787567025*base_y + 0.0146748733049962*base_z + hinge_x*x23 + hinge_y*x30 + hinge_z*x14 - 0.05*x34 - 0.05*x35 + 0.05*x37 - 0.258257036698329],
                     [-x28*x46 - x38*x7 - x45*x5, x3*x44 - x31*x38 - x32*x46, -x47 - x48 + x49, -0.0915174007171296*base_x - 0.214136510870782 *
                         base_y - 0.97250713111939*base_z - hinge_x*x44 - hinge_y*x46 - hinge_z*x38 + 0.05*x47 + 0.05*x48 - 0.05*x49 + 0.116344254058977],
                     [-x28*x58 + x5*x57 - x50*x7, -x3*x56 - x31*x50 - x32*x58, -x59 + x60 + x61, 0.43937488084499*base_x + 0.867718379468268*base_y -
                         0.232410257982334*base_z + hinge_x*x56 - hinge_y*x58 - hinge_z*x50 + 0.05*x59 - 0.05*x60 - 0.05*x61 + 0.790231880961314],
                     [0., 0., 0., 1.]])


@nb.njit('float64[:,:](float64[:])', cache=True, parallel=True)
def arm_t_diff(array):
    """
    base_x, base_y, base_z, base_p, base_q, base_r, hinge_x, hinge_y, hinge_z, hinge_p, hinge_q, hinge_r, rotors_x, rotors_y, rotors_z, rotors_p, rotors_q, rotors_r
    """

    base_x = array[0]
    base_y = array[1]
    base_z = array[2]
    base_p = array[3]
    base_q = array[4]
    base_r = array[5]
    hinge_x = array[6]
    hinge_y = array[7]
    hinge_z = array[8]
    hinge_p = array[9]
    hinge_q = array[10]
    hinge_r = array[11]
    rotors_x = array[12]
    rotors_y = array[13]
    rotors_z = array[14]
    rotors_p = array[15]
    rotors_q = array[16]
    rotors_r = array[17]

    x0 = np.sin(base_r)
    x1 = np.sin(base_p)
    x2 = np.cos(base_p)
    x3 = np.sin(base_q)
    x4 = 0.893629833571167*x3
    x5 = np.cos(base_q)
    x6 = 0.0146748733049962*x5
    x7 = 0.448564787567025*x1 + x2*x4 + x2*x6
    x8 = np.cos(base_r)
    x9 = x4*x8
    x10 = x6*x8
    x11 = 0.0146748733049962*x3
    x12 = x0*x11
    x13 = 0.893629833571167*x5
    x14 = x0*x13
    x15 = 0.448564787567025*x2
    x16 = x0*x4
    x17 = x0*x6
    x18 = x11*x8
    x19 = x13*x8
    x20 = 0.0915174007171296*x3
    x21 = 0.97250713111939*x5
    x22 = 0.214136510870782*x1 - x2*x20 - x2*x21
    x23 = x20*x8
    x24 = x21*x8
    x25 = 0.97250713111939*x3
    x26 = x0*x25
    x27 = 0.0915174007171296*x5
    x28 = x0*x27
    x29 = 0.214136510870782*x2
    x30 = x0*x20
    x31 = x0*x21
    x32 = x25*x8
    x33 = x27*x8
    x34 = 0.43937488084499*x3
    x35 = 0.232410257982334*x5
    x36 = -0.867718379468268*x1 + x2*x34 - x2*x35
    x37 = x34*x8
    x38 = x35*x8
    x39 = 0.232410257982334*x3
    x40 = x0*x39
    x41 = 0.43937488084499*x5
    x42 = x0*x41
    x43 = 0.867718379468268*x2
    x44 = x0*x34
    x45 = x0*x35
    x46 = x39*x8
    x47 = x41*x8

    return np.array([[0., 0., 0., x0*x7, -x1*x12 + x1*x14 - x10 - x9, x1*x10 + x1*x9 + x12 - x14 - x15*x8, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., x7*x8, -x1*x18 + x1*x19 + x16 + x17, x0*x15 - x1*x16 -
                         x1*x17 + x18 - x19, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., -x1*x4 - x1*x6 + x15, x2 *
                         (-x11 + x13), 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                     [0.893629833571167, -0.448564787567025, 0.0146748733049962, 0.,
                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., x0*x22, x1*x26 - x1*x28 + x23 + x24, -x1*x23 - x1*x24 -
                      x26 + x28 - x29*x8, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., x22*x8, x1*x32 - x1*x33 - x30 - x31, x0*x29 + x1*x30 +
                      x1*x31 - x32 + x33, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., x1*x20 + x1*x21 + x29, x2 *
                      (x25 - x27), 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                     [-0.0915174007171296, -0.214136510870782, -0.97250713111939,
                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., x0*x36, x1*x40 + x1*x42 - x37 + x38, x1*x37 - x1*x38 -
                      x40 - x42 + x43*x8, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., x36*x8, x1*x46 + x1*x47 + x44 - x45, -x0*x43 - x1*x44 +
                      x1*x45 - x46 - x47, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., -x1*x34 + x1*x35 - x43, x2 *
                      (x39 + x41), 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                     [0.43937488084499, 0.867718379468268, -0.232410257982334, 0.,
                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0.,
                         0., 0., 0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0.,
                         0., 0., 0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0.,
                         0., 0., 0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])