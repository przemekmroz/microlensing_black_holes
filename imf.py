class IMF:

    def __init__(self, alpha0, alpha1, alpha2, m0, m1, m2, m3):

        self.alpha0 = alpha0
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.m0 = m0
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3

        tmp1 = (pow(m1, 1.0-alpha0) - pow(m0, 1.0-alpha0)) / (1.0-alpha0)
        tmp1 *= pow(m1, alpha0-alpha1)
        tmp2 = (pow(m2, 1.0-alpha1) - pow(m1, 1.0-alpha1)) / (1.0-alpha1)
        tmp3 = (pow(m3, 1.0-alpha2) - pow(m2, 1.0-alpha2)) / (1.0-alpha2)
        tmp3 *= pow(m2, alpha2-alpha1)
        tmp = tmp1 + tmp2 + tmp3

        self.a1 = 1.0/tmp
        self.a2 = pow(m2, alpha2-alpha1) * self.a1
        self.a0 = pow(m1, alpha0-alpha1) * self.a1

    def get_imf(self, mass):

        if mass >= self.m0 and mass <= self.m1:
            return self.a0 * pow(mass, -self.alpha0)
        elif mass > self.m1 and mass <= self.m2:
            return self.a1 * pow(mass, -self.alpha1)
        elif mass > self.m2 and mass <= self.m3:
            return self.a2 * pow(mass, -self.alpha2)
        else:
            return 0.0
