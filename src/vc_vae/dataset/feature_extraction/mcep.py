from pysptk.conversion import sp2mc, mc2sp


def spec2mcep(spec, order=25, alpha=0.35):
    return sp2mc(spec, order, alpha)


def mcep2spec(mcep, alpha=0.35, fftlen=513):
    return mc2sp(mcep, alpha, fftlen)