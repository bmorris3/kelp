"""
# cython: linetrace=True
"""

import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange

from libc.math cimport sin, cos, exp, pi

__all__ = ["_h_ml_sum_cy", "_integrate_planck",
           "_integrated_blackbody", "_phase_curve"]

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef float mu(float theta) nogil:
    r"""
    Angle :math:`\mu = \cos(\theta)`

    Parameters
    ----------
    theta : `~numpy.ndarray`
        Angle :math:`\theta`
    """
    return cos(theta)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef float tilda_mu(float theta, float alpha) nogil:
    r"""
    The normalized quantity
    :math:`\tilde{\mu} = \alpha \mu(\theta)`

    Parameters
    ----------
    theta : `~numpy.ndarray`
        Angle :math:`\theta`
    alpha : float
        Dimensionless fluid number :math:`\alpha`
    """
    return alpha * mu(theta)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef float H(int l, float theta, float alpha) nogil:
    r"""
    Hermite Polynomials in :math:`\tilde{\mu}(\theta)`.

    Parameters
    ----------
    l : int
        Implemented through :math:`\ell leq 51`.
    theta : float
        Angle :math:`\theta`
    alpha : float
        Dimensionless fluid number :math:`\alpha`

    Returns
    -------
    result : `~numpy.ndarray`
        Hermite Polynomial evaluated at angles :math:`\theta`.
    """
    if l == 0:
        return 1
    elif l == 1:
        return 2*tilda_mu(theta, alpha)
    elif l == 2:
        return 4*tilda_mu(theta, alpha)**2 - 2
    elif l == 3:
        return 8*tilda_mu(theta, alpha)**3 - 12*tilda_mu(theta, alpha)
    elif l == 4:
        return 16*tilda_mu(theta, alpha)**4 - 48*tilda_mu(theta, alpha)**2 + 12
    elif l == 5:
        return (32*tilda_mu(theta, alpha)**5 - 160*tilda_mu(theta, alpha)**3 +
                120*tilda_mu(theta, alpha))
    elif l == 6:
        return (64*tilda_mu(theta, alpha)**6 - 480*tilda_mu(theta, alpha)**4 +
                720*tilda_mu(theta, alpha)**2 - 120)
    elif l == 7:
        return (128*tilda_mu(theta, alpha)**7 - 1344*tilda_mu(theta, alpha)**5 +
                3360*tilda_mu(theta, alpha)**3 - 1680*tilda_mu(theta, alpha))
    elif l == 8:
        return (256*tilda_mu(theta, alpha)**8 -
                3584*tilda_mu(theta, alpha)**6 +
                13440*tilda_mu(theta, alpha)**4 -
                13440*tilda_mu(theta, alpha)**2 + 1680)
    elif l == 9:
        return (512*tilda_mu(theta, alpha)**9 -
                9216*tilda_mu(theta, alpha)**7 +
                48384*tilda_mu(theta, alpha)**5 -
                80640*tilda_mu(theta, alpha)**3 +
                30240*tilda_mu(theta, alpha))
    elif l == 10:
        return (1024*tilda_mu(theta, alpha)**10 -
                23040*tilda_mu(theta, alpha)**8 +
                161280*tilda_mu(theta, alpha)**6 -
                403200*tilda_mu(theta, alpha)**4 +
                302400*tilda_mu(theta, alpha)**2 -
                30240)
    elif l == 11:
        return (2048*tilda_mu(theta, alpha)**11 -
                56320*tilda_mu(theta, alpha)**9 +
                506880*tilda_mu(theta, alpha)**7 -
                1774080*tilda_mu(theta, alpha)**5 +
                2217600*tilda_mu(theta, alpha)**3 -
                665280*tilda_mu(theta, alpha))
    elif l == 12:
        return (4096*tilda_mu(theta, alpha)**12 -
                135168*tilda_mu(theta, alpha)**10 +
                1520640*tilda_mu(theta, alpha)**8 -
                7096320*tilda_mu(theta, alpha)**6 +
                13305600*tilda_mu(theta, alpha)**4 -
                7983360*tilda_mu(theta, alpha)**2 + 665280)
    elif l == 13:
        return (8192*tilda_mu(theta, alpha)**13 -
                319488*tilda_mu(theta, alpha)**11 +
                4392960*tilda_mu(theta, alpha)**9 -
                26357760*tilda_mu(theta, alpha)**7 +
                69189120*tilda_mu(theta, alpha)**5 -
                69189120*tilda_mu(theta, alpha)**3 +
                17297280*tilda_mu(theta, alpha))
    elif l == 14:
        return (16384*tilda_mu(theta, alpha)**14 -
                745472*tilda_mu(theta, alpha)**12 +
                12300288*tilda_mu(theta, alpha)**10 -
                92252160*tilda_mu(theta, alpha)**8 +
                322882560*tilda_mu(theta, alpha)**6 -
                484323840*tilda_mu(theta, alpha)**4 +
                242161920*tilda_mu(theta, alpha)**2 -
                17297280)
    elif l == 15:
        return (32768*tilda_mu(theta, alpha)**15 -
                1720320*tilda_mu(theta, alpha)**13 +
                33546240*tilda_mu(theta, alpha)**11 -
                307507200*tilda_mu(theta, alpha)**9 +
                1383782400*tilda_mu(theta, alpha)**7 -
                2905943040.0*tilda_mu(theta, alpha)**5 +
                2421619200.0*tilda_mu(theta, alpha)**3 -
                518918400.0*tilda_mu(theta, alpha))
    elif l == 16:
        return (65536*tilda_mu(theta, alpha)**16 -
                3932160*tilda_mu(theta, alpha)**14 +
                89456640*tilda_mu(theta, alpha)**12 -
                984023040*tilda_mu(theta, alpha)**10 +
                5535129600.0*tilda_mu(theta, alpha)**8 -
                15498362880.0*tilda_mu(theta, alpha)**6 +
                19372953600.0*tilda_mu(theta, alpha)**4 -
                8302694400.0*tilda_mu(theta, alpha)**2 +
                518918400.0)
    elif l == 17:
        return (131072*tilda_mu(theta, alpha)**17 -
                8912896*tilda_mu(theta, alpha)**15 +
                233963520*tilda_mu(theta, alpha)**13 -
                3041525760.0*tilda_mu(theta, alpha)**11 +
                20910489600.0*tilda_mu(theta, alpha)**9 -
                75277762560.0*tilda_mu(theta, alpha)**7 +
                131736084480.0*tilda_mu(theta, alpha)**5 -
                94097203200.0*tilda_mu(theta, alpha)**3 +
                17643225600.0*tilda_mu(theta, alpha))
    elif l == 18:
        return (262144*tilda_mu(theta, alpha)**18 -
                20054016*tilda_mu(theta, alpha)**16 +
                601620480*tilda_mu(theta, alpha)**14 -
                9124577280.0*tilda_mu(theta, alpha)**12 +
                75277762560.0*tilda_mu(theta, alpha)**10 -
                338749931520.0*tilda_mu(theta, alpha)**8 +
                790416506880.0*tilda_mu(theta, alpha)**6 -
                846874828800.0*tilda_mu(theta, alpha)**4 +
                317578060800.0*tilda_mu(theta, alpha)**2 - 17643225600.0)
    elif l == 19:
        return (524288*tilda_mu(theta, alpha)**19 -
                44826624*tilda_mu(theta, alpha)**17 +
                1524105216*tilda_mu(theta, alpha)**15 -
                26671841280.0*tilda_mu(theta, alpha)**13 +
                260050452480.0*tilda_mu(theta, alpha)**11 -
                1430277488640.0*tilda_mu(theta, alpha)**9 +
                4290832465920.0*tilda_mu(theta, alpha)**7 -
                6436248698880.0*tilda_mu(theta, alpha)**5 +
                4022655436800.0*tilda_mu(theta, alpha)**3 -
                670442572800.0*tilda_mu(theta, alpha))
    elif l == 20:
        return (1048576.0*tilda_mu(theta, alpha)**20 +
                -99614720.0*tilda_mu(theta, alpha)**18 +
                3810263040.0*tilda_mu(theta, alpha)**16 +
                -76205260800.0*tilda_mu(theta, alpha)**14 +
                866834841600.0*tilda_mu(theta, alpha)**12 +
                -5721109954560.0*tilda_mu(theta, alpha)**10 +
                21454162329600.0*tilda_mu(theta, alpha)**8 +
                -42908324659199.9*tilda_mu(theta, alpha)**6 +
                40226554368000.0*tilda_mu(theta, alpha)**4 +
                -13408851456000.0*tilda_mu(theta, alpha)**2 +
                670442572800.0*tilda_mu(theta, alpha)**0)
    elif l == 21:
        return (2097152.0*tilda_mu(theta, alpha)**21 +
                -220200960.0*tilda_mu(theta, alpha)**19 +
                9413591040.0*tilda_mu(theta, alpha)**17 +
                -213374730240.0*tilda_mu(theta, alpha)**15 +
                2800543334400.0*tilda_mu(theta, alpha)**13 +
                -21844238008319.8*tilda_mu(theta, alpha)**11 +
                100119424204800.9*tilda_mu(theta, alpha)**9 +
                -257449947955198.9*tilda_mu(theta, alpha)**7 +
                337903056691199.9*tilda_mu(theta, alpha)**5 +
                -187723920383999.9*tilda_mu(theta, alpha)**3 +
                28158588057600.0*tilda_mu(theta, alpha)**1)
    elif l == 22:
        return (4194304.0*tilda_mu(theta, alpha)**22 +
                -484442112.0*tilda_mu(theta, alpha)**20 +
                23011000320.0*tilda_mu(theta, alpha)**18 +
                -586780508160.0*tilda_mu(theta, alpha)**16 +
                8801707622400.1*tilda_mu(theta, alpha)**14 +
                -80095539363839.0*tilda_mu(theta, alpha)**12 +
                440525466501124.6*tilda_mu(theta, alpha)**10 +
                -1415974713753586.8*tilda_mu(theta, alpha)**8 +
                2477955749068807.5*tilda_mu(theta, alpha)**6 +
                -2064963124223999.5*tilda_mu(theta, alpha)**4 +
                619488937267199.5*tilda_mu(theta, alpha)**2 +
                -28158588057600.0*tilda_mu(theta, alpha)**0)
    elif l == 23:
        return (8388608.0*tilda_mu(theta, alpha)**23 +
                -1061158912.0*tilda_mu(theta, alpha)**21 +
                55710842880.0*tilda_mu(theta, alpha)**19 +
                -1587759022080.0*tilda_mu(theta, alpha)**17 +
                26991903375360.3*tilda_mu(theta, alpha)**15 +
                -283414985441279.1*tilda_mu(theta, alpha)**13 +
                1842197405368290.2*tilda_mu(theta, alpha)**11 +
                -7237204092518508.0*tilda_mu(theta, alpha)**9 +
                16283709208166298.0*tilda_mu(theta, alpha)**7 +
                -18997660742860812.0*tilda_mu(theta, alpha)**5 +
                9498830371430392.0*tilda_mu(theta, alpha)**3 +
                -1295295050649598.8*tilda_mu(theta, alpha)**1)
    elif l == 24:
        return (16777216.0*tilda_mu(theta, alpha)**24 +
                -2315255808.0*tilda_mu(theta, alpha)**22 +
                133706022912.0*tilda_mu(theta, alpha)**20 +
                -4234024058880.0*tilda_mu(theta, alpha)**18 +
                80975710126080.4*tilda_mu(theta, alpha)**16 +
                -971708521512948.4*tilda_mu(theta, alpha)**14 +
                7368789621473255.0*tilda_mu(theta, alpha)**12 +
                -34738579644088584.0*tilda_mu(theta, alpha)**10 +
                97702255248998768.0*tilda_mu(theta, alpha)**8 +
                -151981285942885440.0*tilda_mu(theta, alpha)**6 +
                113985964457164448.0*tilda_mu(theta, alpha)**4 +
                -31087081215590364.0*tilda_mu(theta, alpha)**2 +
                1295295050649597.5*tilda_mu(theta, alpha)**0)
    elif l == 25:
        return (33554432.0*tilda_mu(theta, alpha)**25 +
                -5033164800.0*tilda_mu(theta, alpha)**23 +
                318347673600.0*tilda_mu(theta, alpha)**21 +
                -11142168575999.9*tilda_mu(theta, alpha)**19 +
                238163853312003.8*tilda_mu(theta, alpha)**17 +
                -3239028405043092.0*tilda_mu(theta, alpha)**15 +
                28341498544128700.0*tilda_mu(theta, alpha)**13 +
                -157902634745853312.0*tilda_mu(theta, alpha)**11 +
                542790306938886080.0*tilda_mu(theta, alpha)**9 +
                -1085580613877755520.0*tilda_mu(theta, alpha)**7 +
                1139859644571645952.0*tilda_mu(theta, alpha)**5 +
                -518118020259839104.0*tilda_mu(theta, alpha)**3 +
                64764752532479880.0*tilda_mu(theta, alpha)**1)
    elif l == 26:
        return (67108864.0*tilda_mu(theta, alpha)**26 +
                -10905190400.0*tilda_mu(theta, alpha)**24 +
                752458137600.0*tilda_mu(theta, alpha)**22 +
                -28969638297599.1*tilda_mu(theta, alpha)**20 +
                688028909568037.6*tilda_mu(theta, alpha)**18 +
                -10526842316390140.0*tilda_mu(theta, alpha)**16 +
                105268423163901136.0*tilda_mu(theta, alpha)**14 +
                -684244750565396096.0*tilda_mu(theta, alpha)**12 +
                2822509596082127360.0*tilda_mu(theta, alpha)**10 +
                -7056273990205456384.0*tilda_mu(theta, alpha)**8 +
                9878783586287646720.0*tilda_mu(theta, alpha)**6 +
                -6735534263377889280.0*tilda_mu(theta, alpha)**4 +
                1683883565844477696.0*tilda_mu(theta, alpha)**2 +
                -64764752532479864.0*tilda_mu(theta, alpha)**0)
    elif l == 27:
        return (134217728.0*tilda_mu(theta, alpha)**27 +
                -23555211264.0*tilda_mu(theta, alpha)**25 +
                1766640844800.0*tilda_mu(theta, alpha)**23 +
                -74493355622401.1*tilda_mu(theta, alpha)**21 +
                1955450585087907.8*tilda_mu(theta, alpha)**19 +
                -33438205005007912.0*tilda_mu(theta, alpha)**17 +
                378966323390022656.0*tilda_mu(theta, alpha)**15 +
                -2842247425425488384.0*tilda_mu(theta, alpha)**13 +
                13855956198948780032.0*tilda_mu(theta, alpha)**11 +
                -42337643941232533504.0*tilda_mu(theta, alpha)**9 +
                76207759094218915840.0*tilda_mu(theta, alpha)**7 +
                -72743770044481314816.0*tilda_mu(theta, alpha)**5 +
                30309904185200590848.0*tilda_mu(theta, alpha)**3 +
                -3497296636753914368.0*tilda_mu(theta, alpha)**1)
    elif l == 28:
        return (268435456.0*tilda_mu(theta, alpha)**28 +
                -50734301184.0*tilda_mu(theta, alpha)**26 +
                4122161971199.9*tilda_mu(theta, alpha)**24 +
                -189619450675206.6*tilda_mu(theta, alpha)**22 +
                5475261638245897.0*tilda_mu(theta, alpha)**20 +
                -104029971126697872.0*tilda_mu(theta, alpha)**18 +
                1326382131864903168.0*tilda_mu(theta, alpha)**16 +
                -11368989701703942144.0*tilda_mu(theta, alpha)**14 +
                64661128928420298752.0*tilda_mu(theta, alpha)**12 +
                -237090806070914023424.0*tilda_mu(theta, alpha)**10 +
                533454313659522940928.0*tilda_mu(theta, alpha)**8 +
                -678941853748495253504.0*tilda_mu(theta, alpha)**6 +
                424338658592808108032.0*tilda_mu(theta, alpha)**4 +
                -97924305829109547008.0*tilda_mu(theta, alpha)**2 +
                3497296636753913856.0*tilda_mu(theta, alpha)**0)
    elif l == 29:
        return (536870912.0*tilda_mu(theta, alpha)**29 +
                -108984795136.0*tilda_mu(theta, alpha)**27 +
                9563415773184.0*tilda_mu(theta, alpha)**25 +
                -478170788659212.8*tilda_mu(theta, alpha)**23 +
                15122151191346048.0*tilda_mu(theta, alpha)**21 +
                -317565175018326976.0*tilda_mu(theta, alpha)**19 +
                4525303744010010624.0*tilda_mu(theta, alpha)**17 +
                -43960093513252364288.0*tilda_mu(theta, alpha)**15 +
                288488113680659087360.0*tilda_mu(theta, alpha)**13 +
                -1250115159282941165568.0*tilda_mu(theta, alpha)**11 +
                3437816688028146466816.0*tilda_mu(theta, alpha)**9 +
                -5625518216773171675136.0*tilda_mu(theta, alpha)**7 +
                4922328439676601368576.0*tilda_mu(theta, alpha)**5 +
                -1893203246029452869632.0*tilda_mu(theta, alpha)**3 +
                202843204931727163392.0*tilda_mu(theta, alpha)**1)
    elif l == 30:
        return (1073741824.0*tilda_mu(theta, alpha)**30 +
            -233538846720.0*tilda_mu(theta, alpha)**28 +
            22069421015040.1*tilda_mu(theta, alpha)**26 +
            -1195426971647973.0*tilda_mu(theta, alpha)**24 +
            41242230521857320.0*tilda_mu(theta, alpha)**22 +
            -952695525054843520.0*tilda_mu(theta, alpha)**20 +
            15084345813369604096.0*tilda_mu(theta, alpha)**18 +
            -164850350674660622336.0*tilda_mu(theta, alpha)**16 +
            1236377630060125356032.0*tilda_mu(theta, alpha)**14 +
            -6250575796414522064896.0*tilda_mu(theta, alpha)**12 +
            20626900128168713125888.0*tilda_mu(theta, alpha)**10 +
            -42191386625799398359040.0*tilda_mu(theta, alpha)**8 +
            49223284396765632004096.0*tilda_mu(theta, alpha)**6 +
            -28398048690441784655872.0*tilda_mu(theta, alpha)**4 +
            6085296147951808151552.0*tilda_mu(theta, alpha)**2 +
            -202843204931726934016.0*tilda_mu(theta, alpha)**0)
    elif l == 31:
        return (2147483648.0*tilda_mu(theta, alpha)**31 +
            -499289948160.0*tilda_mu(theta, alpha)**29 +
            50677929738240.5*tilda_mu(theta, alpha)**27 +
            -2964658889686922.0*tilda_mu(theta, alpha)**25 +
            111174708363271776.0*tilda_mu(theta, alpha)**23 +
            -2812720121590365184.0*tilda_mu(theta, alpha)**21 +
            49222602127839821824.0*tilda_mu(theta, alpha)**19 +
            -601218925989908185088.0*tilda_mu(theta, alpha)**17 +
            5110360870915287285760.0*tilda_mu(theta, alpha)**15 +
            -29810438413669621563392.0*tilda_mu(theta, alpha)**13 +
            116260709813313431666688.0*tilda_mu(theta, alpha)**11 +
            -290651774533281347796992.0*tilda_mu(theta, alpha)**9 +
            435977661799927608508416.0*tilda_mu(theta, alpha)**7 +
            -352135803761477898207232.0*tilda_mu(theta, alpha)**5 +
            125762787057670555697152.0*tilda_mu(theta, alpha)**3 +
            -12576278705767060602880.0*tilda_mu(theta, alpha)**1)
    elif l == 32:
        return (4294967296.0*tilda_mu(theta, alpha)**32 +
            -1065151889408.0*tilda_mu(theta, alpha)**30 +
            115835267973119.9*tilda_mu(theta, alpha)**28 +
            -7297621882306294.0*tilda_mu(theta, alpha)**26 +
            296465888968728064.0*tilda_mu(theta, alpha)**24 +
            -8182458535536088064.0*tilda_mu(theta, alpha)**22 +
            157512326809063882752.0*tilda_mu(theta, alpha)**20 +
            -2137667292408740577280.0*tilda_mu(theta, alpha)**18 +
            20441443483661920894976.0*tilda_mu(theta, alpha)**16 +
            -136276289891060710113280.0*tilda_mu(theta, alpha)**14 +
            620057119004297365815296.0*tilda_mu(theta, alpha)**12 +
            -1860171357013073157160960.0*tilda_mu(theta, alpha)**10 +
            3487821294399409593778176.0*tilda_mu(theta, alpha)**8 +
            -3756115240122433419608064.0*tilda_mu(theta, alpha)**6 +
            2012204592922727012106240.0*tilda_mu(theta, alpha)**4 +
            -402440918584546073509888.0*tilda_mu(theta, alpha)**2 +
            12576278705767056408576.0*tilda_mu(theta, alpha)**0)
    elif l == 33:
        return (8589934592.0*tilda_mu(theta, alpha)**33 +
            -2267742732288.0*tilda_mu(theta, alpha)**31 +
            263625092628477.3*tilda_mu(theta, alpha)**29 +
            -17838631267861220.0*tilda_mu(theta, alpha)**27 +
            782669946877279488.0*tilda_mu(theta, alpha)**25 +
            -23480098406327300096.0*tilda_mu(theta, alpha)**23 +
            495038741399822532608.0*tilda_mu(theta, alpha)**21 +
            -7425581120999220314112.0*tilda_mu(theta, alpha)**19 +
            79360898230695925645312.0*tilda_mu(theta, alpha)**17 +
            -599615675520506465878016.0*tilda_mu(theta, alpha)**15 +
            3147982296484102541410304.0*tilda_mu(theta, alpha)**13 +
            -11161028142077519283093504.0*tilda_mu(theta, alpha)**11 +
            25577356158929160454012928.0*tilda_mu(theta, alpha)**9 +
            -35414800835440145736925184.0*tilda_mu(theta, alpha)**7 +
            26561100626579964347547648.0*tilda_mu(theta, alpha)**5 +
            -8853700208860016033136640.0*tilda_mu(theta, alpha)**3 +
            830034394580626167562240.0*tilda_mu(theta, alpha)**1)
    elif l == 34:
        return (17179869184.0*tilda_mu(theta, alpha)**34 +
            -4818953306112.0*tilda_mu(theta, alpha)**32 +
            597550209957884.6*tilda_mu(theta, alpha)**30 +
            -43322390221949592.0*tilda_mu(theta, alpha)**28 +
            2046982937986694912.0*tilda_mu(theta, alpha)**26 +
            -66526945484581355520.0*tilda_mu(theta, alpha)**24 +
            1530119746145634222080.0*tilda_mu(theta, alpha)**22 +
            -25246975811382459498496.0*tilda_mu(theta, alpha)**20 +
            299807837760571261845504.0*tilda_mu(theta, alpha)**18 +
            -2548366620961299811532800.0*tilda_mu(theta, alpha)**16 +
            15290199725781102530396160.0*tilda_mu(theta, alpha)**14 +
            -63245826138442082698657792.0*tilda_mu(theta, alpha)**12 +
            173926021880708273505566720.0*tilda_mu(theta, alpha)**10 +
            -301025807101248950377644032.0*tilda_mu(theta, alpha)**8 +
            301025807101238642456133632.0*tilda_mu(theta, alpha)**6 +
            -150512903550620248941002752.0*tilda_mu(theta, alpha)**4 +
            28221169415741326741209088.0*tilda_mu(theta, alpha)**2 +
            -830034394580626301779968.0*tilda_mu(theta, alpha)**0)
    elif l == 35:
        return (34359738368.0*tilda_mu(theta, alpha)**35 +
            -10222022164480.0*tilda_mu(theta, alpha)**33 +
            1349306925711375.5*tilda_mu(theta, alpha)**31 +
            -104571286742627840.0*tilda_mu(theta, alpha)**29 +
            5306992802188616704.0*tilda_mu(theta, alpha)**27 +
            -186275447356796141568.0*tilda_mu(theta, alpha)**25 +
            4656886183921868341248.0*tilda_mu(theta, alpha)**23 +
            -84156586037944361418752.0*tilda_mu(theta, alpha)**21 +
            1104555191749320531509248.0*tilda_mu(theta, alpha)**19 +
            -10493274321606772409761792.0*tilda_mu(theta, alpha)**17 +
            71354265386980230821838848.0*tilda_mu(theta, alpha)**15 +
            -340554448437696120513101824.0*tilda_mu(theta, alpha)**13 +
            1106801957422934604132646912.0*tilda_mu(theta, alpha)**11 +
            -2341311833009462988066783232.0*tilda_mu(theta, alpha)**9 +
            3010258071012456106110746624.0*tilda_mu(theta, alpha)**7 +
            -2107180649708681286150782976.0*tilda_mu(theta, alpha)**5 +
            658493953033963365778587648.0*tilda_mu(theta, alpha)**3 +
            -58102407620643792537780224.0*tilda_mu(theta, alpha)**1)
    elif l == 36:
        return (68719476736.0*tilda_mu(theta, alpha)**36 +
            -21646635171840.0*tilda_mu(theta, alpha)**34 +
            3035940582850575.0*tilda_mu(theta, alpha)**32 +
            -250971088182307584.0*tilda_mu(theta, alpha)**30 +
            13646552919915014144.0*tilda_mu(theta, alpha)**28 +
            -515839700372526071808.0*tilda_mu(theta, alpha)**26 +
            13970658551768408915968.0*tilda_mu(theta, alpha)**24 +
            -275421554306074084900864.0*tilda_mu(theta, alpha)**22 +
            3976398690295018594238464.0*tilda_mu(theta, alpha)**20 +
            -41973097286443968860520448.0*tilda_mu(theta, alpha)**18 +
            321094194241417691602616320.0*tilda_mu(theta, alpha)**16 +
            -1751422877679432242575704064.0*tilda_mu(theta, alpha)**14 +
            6640811744537440773906366464.0*tilda_mu(theta, alpha)**12 +
            -16857445197668627854508687360.0*tilda_mu(theta, alpha)**10 +
            27092322639112131343275786240.0*tilda_mu(theta, alpha)**8 +
            -25286167796504211717693112320.0*tilda_mu(theta, alpha)**6 +
            11852891154611318318904115200.0*tilda_mu(theta, alpha)**4 +
            -2091686674343181857119535104.0*tilda_mu(theta, alpha)**2 +
            58102407620643861257256960.0*tilda_mu(theta, alpha)**0)
    elif l == 37:
        return (137438953472.0*tilda_mu(theta, alpha)**37 +
            -45767171506175.8*tilda_mu(theta, alpha)**35 +
            6807866761543872.0*tilda_mu(theta, alpha)**33 +
            -599092275015834624.0*tilda_mu(theta, alpha)**31 +
            34822238485287190528.0*tilda_mu(theta, alpha)**29 +
            -1413782882503917830144.0*tilda_mu(theta, alpha)**27 +
            41353149313192873164800.0*tilda_mu(theta, alpha)**25 +
            -886138913854343056719872.0*tilda_mu(theta, alpha)**23 +
            14012071575338828827721728.0*tilda_mu(theta, alpha)**21 +
            -163474168378599608070176768.0*tilda_mu(theta, alpha)**19 +
            1397704139639967789346717696.0*tilda_mu(theta, alpha)**17 +
            -8640352863216952691700793344.0*tilda_mu(theta, alpha)**15 +
            37801543776599181503406014464.0*tilda_mu(theta, alpha)**13 +
            -113404631329775198035895123968.0*tilda_mu(theta, alpha)**11 +
            222759097254911126171824422912.0*tilda_mu(theta, alpha)**9 +
            -267310916705905307055825092608.0*tilda_mu(theta, alpha)**7 +
            175422789088247775002571571200.0*tilda_mu(theta, alpha)**5 +
            -51594937967131706808843894784.0*tilda_mu(theta, alpha)**3 +
            4299578163927645715857145856.0*tilda_mu(theta, alpha)**1)
    elif l == 38:
        return (274877906944.0*tilda_mu(theta, alpha)**38 +
            -96619584290816.3*tilda_mu(theta, alpha)**36 +
            15217584525803296.0*tilda_mu(theta, alpha)**34 +
            -1422844153162714112.0*tilda_mu(theta, alpha)**32 +
            88216337496061968384.0*tilda_mu(theta, alpha)**30 +
            -3837410681081981042688.0*tilda_mu(theta, alpha)**28 +
            120878436453877999992832.0*tilda_mu(theta, alpha)**26 +
            -2806106560543170297331712.0*tilda_mu(theta, alpha)**24 +
            48405338169252676479483904.0*tilda_mu(theta, alpha)**22 +
            -621201839839686357384429568.0*tilda_mu(theta, alpha)**20 +
            5901417478475558250845569024.0*tilda_mu(theta, alpha)**18 +
            -41041676100284966487921262592.0*tilda_mu(theta, alpha)**16 +
            205208380501568032834007859200.0*tilda_mu(theta, alpha)**14 +
            -718229331755128213976930844672.0*tilda_mu(theta, alpha)**12 +
            1692969139137392830661466783744.0*tilda_mu(theta, alpha)**10 +
            -2539453708706143289187728621568.0*tilda_mu(theta, alpha)**8 +
            2222021995117794946564667473920.0*tilda_mu(theta, alpha)**6 +
            -980303821375502183077429379072.0*tilda_mu(theta, alpha)**4 +
            163383970229250586680594792448.0*tilda_mu(theta, alpha)**2 +
            -4299578163927644616345518080.0*tilda_mu(theta, alpha)**0)
    elif l == 39:
        return (549755813888.0*tilda_mu(theta, alpha)**39 +
            -203684529045503.9*tilda_mu(theta, alpha)**37 +
            33913474086076000.0*tilda_mu(theta, alpha)**35 +
            -3363086180202583040.0*tilda_mu(theta, alpha)**33 +
            221963687893379710976.0*tilda_mu(theta, alpha)**31 +
            -10321311487040698712064.0*tilda_mu(theta, alpha)**29 +
            349204371978382754185216.0*tilda_mu(theta, alpha)**27 +
            -8755052468877257454125056.0*tilda_mu(theta, alpha)**25 +
            164157233791611538256166912.0*tilda_mu(theta, alpha)**23 +
            -2307321119403861543636434944.0*tilda_mu(theta, alpha)**21 +
            24226871753738851860764164096.0*tilda_mu(theta, alpha)**19 +
            -188308866813057303772466774016.0*tilda_mu(theta, alpha)**17 +
            1067083578608148985662236786688.0*tilda_mu(theta, alpha)**15 +
            -4309375990531399787809416937472.0*tilda_mu(theta, alpha)**13 +
            12004690259337530067846332153856.0*tilda_mu(theta, alpha)**11 +
            -22008598808785662068135865352192.0*tilda_mu(theta, alpha)**9 +
            24759673659884320749565538992128.0*tilda_mu(theta, alpha)**7 +
            -15292739613457832873812996128768.0*tilda_mu(theta, alpha)**5 +
            4247983225960509131614721146880.0*tilda_mu(theta, alpha)**3 +
            -335367096786356358140275982336.0*tilda_mu(theta, alpha)**1)
    elif l == 40:
        return (1099511627776.0*tilda_mu(theta, alpha)**40 +
            -428809534832641.4*tilda_mu(theta, alpha)**38 +
            75363275746833936.0*tilda_mu(theta, alpha)**36 +
            -7913143953418817536.0*tilda_mu(theta, alpha)**34 +
            554909219733289566208.0*tilda_mu(theta, alpha)**32 +
            -27523497298787053862912.0*tilda_mu(theta, alpha)**30 +
            997726777080042198401024.0*tilda_mu(theta, alpha)**28 +
            -26938622981202979317612544.0*tilda_mu(theta, alpha)**26 +
            547190779304994524259942400.0*tilda_mu(theta, alpha)**24 +
            -8390258616009405636192239616.0*tilda_mu(theta, alpha)**22 +
            96907487015035311152070393856.0*tilda_mu(theta, alpha)**20 +
            -836928296946785538423589437440.0*tilda_mu(theta, alpha)**18 +
            5335417893037123330523336278016.0*tilda_mu(theta, alpha)**16 +
            -24625005660201126988658375655424.0*tilda_mu(theta, alpha)**14 +
            80031268395542460957040595435520.0*tilda_mu(theta, alpha)**12 +
            -176068790470310156415030007955456.0*tilda_mu(theta, alpha)**10 +
            247596736598843874028400240754688.0*tilda_mu(theta, alpha)**8 +
            -203903194846102114460098891874304.0*tilda_mu(theta, alpha)**6 +
            84959664519210610474259023134720.0*tilda_mu(theta, alpha)**4 +
            -13414683871454236874162483232768.0*tilda_mu(theta, alpha)**2 +
            335367096786356217402787627008.0*tilda_mu(theta, alpha)**0)
    elif l == 41:
        return (2199023255552.0*tilda_mu(theta, alpha)**41 +
            -901599534776320.5*tilda_mu(theta, alpha)**39 +
            167021313817314112.0*tilda_mu(theta, alpha)**37 +
            -18539365833721475072.0*tilda_mu(theta, alpha)**35 +
            1378865333883126677504.0*tilda_mu(theta, alpha)**33 +
            -72804089628991290343424.0*tilda_mu(theta, alpha)**31 +
            2821158473128778225680384.0*tilda_mu(theta, alpha)**29 +
            -81813595720515458640642048.0*tilda_mu(theta, alpha)**27 +
            1794785756119027853564051456.0*tilda_mu(theta, alpha)**25 +
            -29913095935440384130436562944.0*tilda_mu(theta, alpha)**23 +
            378400663581057633751306076160.0*tilda_mu(theta, alpha)**21 +
            -3612006334202213064644478631936.0*tilda_mu(theta, alpha)**19 +
            25735545131089778804597800828928.0*tilda_mu(theta, alpha)**17 +
            -134616697609068401953318371328000.0*tilda_mu(theta, alpha)**15 +
            504812616033607938757921603584000.0*tilda_mu(theta, alpha)**13 +
            -1312512801687661016868997746917376.0*tilda_mu(theta, alpha)**11 +
            2255881377900479536610331508867072.0*tilda_mu(theta, alpha)**9 +
            -2388580282482981233938847401771008.0*tilda_mu(theta, alpha)**7 +
            1393338498115042842850772100579328.0*tilda_mu(theta, alpha)**5 +
            -366668025819750217587418816577536.0*tilda_mu(theta, alpha)**3 +
            27500101936481224885939839434752.0*tilda_mu(theta, alpha)**1)
    elif l == 42:
        return (4398046511104.0*tilda_mu(theta, alpha)**42 +
            -1893359023030269.2*tilda_mu(theta, alpha)**40 +
            369205009490917312.0*tilda_mu(theta, alpha)**38 +
            -43258520278673113088.0*tilda_mu(theta, alpha)**36 +
            3406608471949001621504.0*tilda_mu(theta, alpha)**34 +
            -191110735275955033473024.0*tilda_mu(theta, alpha)**32 +
            7899243724759165128671232.0*tilda_mu(theta, alpha)**30 +
            -245440787161643064225693696.0*tilda_mu(theta, alpha)**28 +
            5798538596699902905723912192.0*tilda_mu(theta, alpha)**26 +
            -104695835773732280534053158912.0*tilda_mu(theta, alpha)**24 +
            1444802533677531901870246723584.0*tilda_mu(theta, alpha)**22 +
            -15170426603619025277691327676416.0*tilda_mu(theta, alpha)**20 +
            120099210611803591083079317323776.0*tilda_mu(theta, alpha)**18 +
            -706737662447933603615272748449792.0*tilda_mu(theta, alpha)**16 +
            3028875696200938729937384486469632.0*tilda_mu(theta, alpha)**14 +
            -9187589611812058568375966613110784.0*tilda_mu(theta, alpha)**12 +
            18949403574367920601110638311243776.0*tilda_mu(theta, alpha)**10 +
            -25080092966069264302907376661430272.0*tilda_mu(theta, alpha)**8 +
            19506738973610877653993419658231808.0*tilda_mu(theta, alpha)**6 +
            -7700028542214760478058506258219008.0*tilda_mu(theta, alpha)**4 +
            1155004281332210814705525424390144.0*tilda_mu(theta, alpha)**2 +
            -27500101936481202367941702582272.0*tilda_mu(theta, alpha)**0)
    elif l == 43:
        return (8796093022208.0*tilda_mu(theta, alpha)**43 +
            -3971435999526915.5*tilda_mu(theta, alpha)**41 +
            814144379903005312.0*tilda_mu(theta, alpha)**39 +
            -100546830918017040384.0*tilda_mu(theta, alpha)**37 +
            8370523673928971321344.0*tilda_mu(theta, alpha)**35 +
            -498046158598212276977664.0*tilda_mu(theta, alpha)**33 +
            21914030978331244381601792.0*tilda_mu(theta, alpha)**31 +
            -727858886067842591796232192.0*tilda_mu(theta, alpha)**29 +
            18469419233870720854535438336.0*tilda_mu(theta, alpha)**27 +
            -360153675062152911384125374464.0*tilda_mu(theta, alpha)**25 +
            5402305125930828171295636586496.0*tilda_mu(theta, alpha)**23 +
            -62126508947971670417716210040832.0*tilda_mu(theta, alpha)**21 +
            543606953297229546310033347706880.0*tilda_mu(theta, alpha)**19 +
            -3575261115905834834919776891961344.0*tilda_mu(theta, alpha)**17 +
            17365553991564888267074442027335680.0*tilda_mu(theta, alpha)**15 +
            -60779438970449962245013074276777984.0*tilda_mu(theta, alpha)**13 +
            148149882490494858696134074591870976.0*tilda_mu(theta, alpha)**11 +
            -239654221675779351554189765285773312.0*tilda_mu(theta, alpha)**9 +
            239654221675788648713202914899787776.0*tilda_mu(theta, alpha)**7 +
            -132440490926093841484443752851308544.0*tilda_mu(theta, alpha)**5 +
            33110122731523335855588440673353728.0*tilda_mu(theta, alpha)**3 +
            -2365008766537382079584695975149568.0*tilda_mu(theta, alpha)**1)
    elif l == 44:
        return (17592186044416.0*tilda_mu(theta, alpha)**44 +
            -8321103999008770.0*tilda_mu(theta, alpha)**42 +
            1791117635786605056.0*tilda_mu(theta, alpha)**40 +
            -232845292652293685248.0*tilda_mu(theta, alpha)**38 +
            20461280091803219918848.0*tilda_mu(theta, alpha)**36 +
            -1289060645787030087794688.0*tilda_mu(theta, alpha)**34 +
            60263585190271333464801280.0*tilda_mu(theta, alpha)**32 +
            -2135052732461521459312852992.0*tilda_mu(theta, alpha)**30 +
            58046746164004558362736852992.0*tilda_mu(theta, alpha)**28 +
            -1218981669430645866067820281856.0*tilda_mu(theta, alpha)**26 +
            19808452128575016469981728604160.0*tilda_mu(theta, alpha)**24 +
            -248506035789670298164649233743872.0*tilda_mu(theta, alpha)**22 +
            2391870594529003439207394018590720.0*tilda_mu(theta, alpha)**20 +
            -17479054344326736982013847634182144.0*tilda_mu(theta, alpha)**18 +
            95510546953808284714863182821195776.0*tilda_mu(theta, alpha)**16 +
            -382042187814172082140332957875830784.0*tilda_mu(theta, alpha)**14 +
            1086432471596752988048893522628050944.0*tilda_mu(theta, alpha)**12 +
            -2108957150747133784731564342442459136.0*tilda_mu(theta, alpha)**10 +
            2636196438433630199576668507429928960.0*tilda_mu(theta, alpha)**8 +
            -1942460533582697967571602927490433024.0*tilda_mu(theta, alpha)**6 +
            728422700093515943696999903586680832.0*tilda_mu(theta, alpha)**4 +
            -104060385727644808042962109086040064.0*tilda_mu(theta, alpha)**2 +
            2365008766537383520736576733708288.0*tilda_mu(theta, alpha)**0)
    elif l == 45:
        return (35184372088832.0*tilda_mu(theta, alpha)**45 +
            -17416264183971828.0*tilda_mu(theta, alpha)**43 +
            3931721639531627520.0*tilda_mu(theta, alpha)**41 +
            -537335290736001220608.0*tilda_mu(theta, alpha)**39 +
            49770681304413215129600.0*tilda_mu(theta, alpha)**37 +
            -3314727374875534280359936.0*tilda_mu(theta, alpha)**35 +
            164355232337566912198213632.0*tilda_mu(theta, alpha)**33 +
            -6198540191002220311477223424.0*tilda_mu(theta, alpha)**31 +
            180145074302035113397163393024.0*tilda_mu(theta, alpha)**29 +
            -4063272231454263791222404415488.0*tilda_mu(theta, alpha)**27 +
            71310427662250949451159600889856.0*tilda_mu(theta, alpha)**25 +
            -972414922664026873118642531205120.0*tilda_mu(theta, alpha)**23 +
            10250873976500845805312802706948096.0*tilda_mu(theta, alpha)**21 +
            -82795520578470836750174952877981696.0*tilda_mu(theta, alpha)**19 +
            505644072108411567507708323146235904.0*tilda_mu(theta, alpha)**17 +
            -2292253126886322879483441877809627136.0*tilda_mu(theta, alpha)**15 +
            7521455572588167928258291392040665088.0*tilda_mu(theta, alpha)**13 +
            -17255103960662868066859597769348218880.0*tilda_mu(theta, alpha)**11 +
            26361964384334622013890404198014517248.0*tilda_mu(theta, alpha)**9 +
            -24974492574635014841059084728547672064.0*tilda_mu(theta, alpha)**7 +
            13111608601683272524298644476271788032.0*tilda_mu(theta, alpha)**5 +
            -3121811571829339076200522633906749440.0*tilda_mu(theta, alpha)**3 +
            212850788988364282823226470843809792.0*tilda_mu(theta, alpha)**1)
    elif l == 46:
        return (70368744177664.0*tilda_mu(theta, alpha)**46 +
            -36415825111941088.0*tilda_mu(theta, alpha)**44 +
            8612342638974439424.0*tilda_mu(theta, alpha)**42 +
            -1235871168692441317376.0*tilda_mu(theta, alpha)**40 +
            120497438947619039608832.0*tilda_mu(theta, alpha)**38 +
            -8470969958006605758857216.0*tilda_mu(theta, alpha)**36 +
            444725922796015170118221824.0*tilda_mu(theta, alpha)**34 +
            -17820803049166447096184700928.0*tilda_mu(theta, alpha)**32 +
            552444894521125084743761657856.0*tilda_mu(theta, alpha)**30 +
            -13350751617926566533249831534592.0*tilda_mu(theta, alpha)**28 +
            252329205565600040015713179533312.0*tilda_mu(theta, alpha)**26 +
            -3727590537009843092361466959888384.0*tilda_mu(theta, alpha)**24 +
            42867291173408984889417278861344768.0*tilda_mu(theta, alpha)**22 +
            -380859394665510147268456502779707392.0*tilda_mu(theta, alpha)**20 +
            2584403035208120047138347333177049088.0*tilda_mu(theta, alpha)**18 +
            -13180455479623155322737289380449222656.0*tilda_mu(theta, alpha)**16 +
            49426708048410026468850385301178155008.0*tilda_mu(theta, alpha)**14 +
            -132289130365065581437306599460982226944.0*tilda_mu(theta, alpha)**12 +
            242530072335914244396814737765988040704.0*tilda_mu(theta, alpha)**10 +
            -287206664608288168284710581697846968320.0*tilda_mu(theta, alpha)**8 +
            201044665225813297041913403558556860416.0*tilda_mu(theta, alpha)**6 +
            -71801666152074686596408052425781411840.0*tilda_mu(theta, alpha)**4 +
            9791136293464773243003202523220672512.0*tilda_mu(theta, alpha)**2 +
            -212850788988364688651596092453945344.0*tilda_mu(theta, alpha)**0)
    elif l == 47:
        return (140737488355328.0*tilda_mu(theta, alpha)**47 +
            -76068612456054960.0*tilda_mu(theta, alpha)**45 +
            18826981582873137152.0*tilda_mu(theta, alpha)**43 +
            -2833460728222645747712.0*tilda_mu(theta, alpha)**41 +
            290429724642674256904192.0*tilda_mu(theta, alpha)**39 +
            -21520842596070907376041984.0*tilda_mu(theta, alpha)**37 +
            1194406764074901036163137536.0*tilda_mu(theta, alpha)**35 +
            -50762287473711443416543395840.0*tilda_mu(theta, alpha)**33 +
            1675155486610408595993026101248.0*tilda_mu(theta, alpha)**31 +
            -43274850071172784867551324667904.0*tilda_mu(theta, alpha)**29 +
            878479456447997300922249293856768.0*tilda_mu(theta, alpha)**27 +
            -14015740418519592193651559804960768.0*tilda_mu(theta, alpha)**25 +
            175196755235925425271366993173282816.0*tilda_mu(theta, alpha)**23 +
            -1704799195146009458155075160405704704.0*tilda_mu(theta, alpha)**21 +
            12785993963734363923977123013735743488.0*tilda_mu(theta, alpha)**19 +
            -72880165592903706942073761113120440320.0*tilda_mu(theta, alpha)**17 +
            309740703770944336133111499294863523840.0*tilda_mu(theta, alpha)**15 +
            -956552173407778446150408266692006248448.0*tilda_mu(theta, alpha)**13 +
            2072529709052891478917499434087212908544.0*tilda_mu(theta, alpha)**11 +
            -2999714052575492560293573153571257450496.0*tilda_mu(theta, alpha)**9 +
            2699742647317976670616837201979353595904.0*tilda_mu(theta, alpha)**7 +
            -1349871323659010095973171664314821509120.0*tilda_mu(theta, alpha)**5 +
            306788937195229017754893735240107294720.0*tilda_mu(theta, alpha)**3 +
            -20007974164906257637926452406312239104.0*tilda_mu(theta, alpha)**1)
    elif l == 48:
        return (281474976710656.0*tilda_mu(theta, alpha)**48 +
            -158751886864809920.0*tilda_mu(theta, alpha)**46 +
            41077050726271336448.0*tilda_mu(theta, alpha)**44 +
            -6476481664506639941632.0*tilda_mu(theta, alpha)**42 +
            697031339143178613686272.0*tilda_mu(theta, alpha)**40 +
            -54368444453115449246220288.0*tilda_mu(theta, alpha)**38 +
            3185084704205065062790987776.0*tilda_mu(theta, alpha)**36 +
            -143328811690623356515311943680.0*tilda_mu(theta, alpha)**34 +
            5025466459831719495088228794368.0*tilda_mu(theta, alpha)**32 +
            -138479520226632214227611153334272.0*tilda_mu(theta, alpha)**30 +
            3011929564995085936128107367890944.0*tilda_mu(theta, alpha)**28 +
            -51750426160816968587615114155261952.0*tilda_mu(theta, alpha)**26 +
            700787020933353225234069503912968192.0*tilda_mu(theta, alpha)**24 +
            -7439123760744427459999944566928048128.0*tilda_mu(theta, alpha)**22 +
            61372771025327593920259484416412221440.0*tilda_mu(theta, alpha)**20 +
            -388694216498480341816669021754468859904.0*tilda_mu(theta, alpha)**18 +
            1858444222614558708599504686569274474496.0*tilda_mu(theta, alpha)**16 +
            -6559214903389407092385604172611477045248.0*tilda_mu(theta, alpha)**14 +
            16580237672404381391877772574198010478592.0*tilda_mu(theta, alpha)**12 +
            -28797254904729828311495764625994652057600.0*tilda_mu(theta, alpha)**10 +
            32396911767815403308837307390908470132736.0*tilda_mu(theta, alpha)**8 +
            -21597941178544294517410904238246361825280.0*tilda_mu(theta, alpha)**6 +
            7362934492685510933227285021312671547392.0*tilda_mu(theta, alpha)**4 +
            -960382759915499271031445689745297899520.0*tilda_mu(theta, alpha)**2 +
            20007974164906248193193486667021811712.0*tilda_mu(theta, alpha)**0)
    elif l == 49:
        return (562949953421312.0*tilda_mu(theta, alpha)**49 +
            -331014572611733312.0*tilda_mu(theta, alpha)**47 +
            89456688248315953152.0*tilda_mu(theta, alpha)**45 +
            -14760353560974092926976.0*tilda_mu(theta, alpha)**43 +
            1666074908195641125502976.0*tilda_mu(theta, alpha)**41 +
            -136618142471498074591068160.0*tilda_mu(theta, alpha)**39 +
            8436170297725881700356980736.0*tilda_mu(theta, alpha)**37 +
            -401320672724951345914246594560.0*tilda_mu(theta, alpha)**35 +
            14924112517366896590094401011712.0*tilda_mu(theta, alpha)**33 +
            -437773967167431127962340149952512.0*tilda_mu(theta, alpha)**31 +
            10178244736764252488407610543833088.0*tilda_mu(theta, alpha)**29 +
            -187834880139911814187745768401534976.0*tilda_mu(theta, alpha)**27 +
            2747085122084222565671271450409959424.0*tilda_mu(theta, alpha)**25 +
            -31697136023728861590163163813760729088.0*tilda_mu(theta, alpha)**23 +
            286406264786082893365430525000073347072.0*tilda_mu(theta, alpha)**21 +
            -2004843853514776426131332135515213266944.0*tilda_mu(theta, alpha)**19 +
            10713384342156464817206636705885454336000.0*tilda_mu(theta, alpha)**17 +
            -42853537368700807833939375373856901431296.0*tilda_mu(theta, alpha)**15 +
            124989483992172036407004111128601991053312.0*tilda_mu(theta, alpha)**13 +
            -256557361878338732979490752813874989236224.0*tilda_mu(theta, alpha)**11 +
            352766372582949776871257506576201032925184.0*tilda_mu(theta, alpha)**9 +
            -302371176499605659655246789912002880864256.0*tilda_mu(theta, alpha)**7 +
            144313516056636191761565105845291209195520.0*tilda_mu(theta, alpha)**5 +
            -31372503490573013260752360713641089040384.0*tilda_mu(theta, alpha)**3 +
            1960781468160812724334112737287980711936.0*tilda_mu(theta, alpha)**1)
    elif l == 50:
        return (1125899906842624.0*tilda_mu(theta, alpha)**50 +
            -689613692941110400.0*tilda_mu(theta, alpha)**48 +
            194471061409375387648.0*tilda_mu(theta, alpha)**46 +
            -33546258093136123265024.0*tilda_mu(theta, alpha)**44 +
            3966845019505472779059200.0*tilda_mu(theta, alpha)**42 +
            -341545356180992004129292288.0*tilda_mu(theta, alpha)**40 +
            22200448151583685509864488960.0*tilda_mu(theta, alpha)**38 +
            -1114779646485118004199023443968.0*tilda_mu(theta, alpha)**36 +
            43894448578937017597455931277312.0*tilda_mu(theta, alpha)**34 +
            -1368043647467760969619855567224832.0*tilda_mu(theta, alpha)**32 +
            33927482453949784961228913510449152.0*tilda_mu(theta, alpha)**30 +
            -670838857666192057857793366966665216.0*tilda_mu(theta, alpha)**28 +
            10565712008039427365813347789005914112.0*tilda_mu(theta, alpha)**26 +
            -132071400095559388169196170442648322048.0*tilda_mu(theta, alpha)**24 +
            1301846658149671150179682555588438392832.0*tilda_mu(theta, alpha)**22 +
            -10024219267444887025620685611719274266624.0*tilda_mu(theta, alpha)**20 +
            59518801901162469912276111290258035834880.0*tilda_mu(theta, alpha)**18 +
            -267834608553966428291123966475400837595136.0*tilda_mu(theta, alpha)**16 +
            892782028516053911818217293656572502736896.0*tilda_mu(theta, alpha)**14 +
            -2137978015651753168844729553112689673240576.0*tilda_mu(theta, alpha)**12 +
            3527663725830615319083039942805175513645056.0*tilda_mu(theta, alpha)**10 +
            -3779639706244714799243664099402868921991168.0*tilda_mu(theta, alpha)**8 +
            2405225267610634363745615855381149644750848.0*tilda_mu(theta, alpha)**6 +
            -784312587264326366359310607963600774496256.0*tilda_mu(theta, alpha)**4 +
            98039073408040621709595801488848939122688.0*tilda_mu(theta, alpha)**2 +
            -1960781468160813026565567640945274388480.0*tilda_mu(theta, alpha)**0)
    elif l == 51:
        return (2251799813685248.0*tilda_mu(theta, alpha)**51 +
            -1435522381224342272.0*tilda_mu(theta, alpha)**49 +
            422043580079964553216.0*tilda_mu(theta, alpha)**47 +
            -76038185011077797904384.0*tilda_mu(theta, alpha)**45 +
            9409725395115301029281792.0*tilda_mu(theta, alpha)**43 +
            -849698203181088561455693824.0*tilda_mu(theta, alpha)**41 +
            58062710550177821702207045632.0*tilda_mu(theta, alpha)**39 +
            -3073176322761444750897759387648.0*tilda_mu(theta, alpha)**37 +
            127920964430256629616723137396736.0*tilda_mu(theta, alpha)**35 +
            -4228498546623658056479289294979072.0*tilda_mu(theta, alpha)**33 +
            111632361625699172139271115400806400.0*tilda_mu(theta, alpha)**31 +
            -2359502189047605952561305303287595008.0*tilda_mu(theta, alpha)**29 +
            39914912028078095924853648087987519488.0*tilda_mu(theta, alpha)**27 +
            -538851312437880932321793491355972927488.0*tilda_mu(theta, alpha)**25 +
            5773406918263494304320366075429477416960.0*tilda_mu(theta, alpha)**23 +
            -48689065016090894043450338788633326125056.0*tilda_mu(theta, alpha)**21 +
            319521989144081197678459714349676462342144.0*tilda_mu(theta, alpha)**19 +
            -1607007651342727524031194271954551628103680.0*tilda_mu(theta, alpha)**17 +
            6070917793887656216138256793477059581050880.0*tilda_mu(theta, alpha)**15 +
            -16774904430519180503471333131755940583833600.0*tilda_mu(theta, alpha)**13 +
            32711063639501865255124722092714280242642944.0*tilda_mu(theta, alpha)**11 +
            -42835916670780538104020292400376643923738624.0*tilda_mu(theta, alpha)**9 +
            35047568185182874253163900600048433915494400.0*tilda_mu(theta, alpha)**7 +
            -15999976780192179140743437852271972215422976.0*tilda_mu(theta, alpha)**5 +
            3333328495873377153506755800803104098615296.0*tilda_mu(theta, alpha)**3 +
            -199999709752402912389189334578924129091584.0*tilda_mu(theta, alpha)**1)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef float h_ml_cython(float omega_drag, float alpha, int m, int l, float theta,
                       float phi, float C) nogil:
    r"""
    The :math:`h_{m\ell}` basis function.

    Parameters
    ----------
    omega_drag : float
        Dimensionless drag
    alpha : float
        Dimensionless fluid number
    m : int
        Spherical harmonic ``m`` index
    l : int
        Spherical harmonic ``l`` index
    theta : `~numpy.ndarray`
        Latitudinal coordinate
    phi : `~numpy.ndarray`
        Longitudinal coordinate
    C_ml : list
        Spherical harmonic coefficients

    Returns
    -------
    hml : `~numpy.ndarray`
        :math:`h_{m\ell}` basis function.
    """
    cdef float prefactor, result = 0

    if m == 0:
        return result

    prefactor = (C /
                 (omega_drag ** 2 * alpha ** 4 + m ** 2) *
                 exp(-tilda_mu(theta, alpha) ** 2 / 2))

    result = prefactor * (mu(theta) * m * H(l, theta, alpha) * cos(m * phi) +
                          alpha * omega_drag * (tilda_mu(theta, alpha) *
                                                H(l, theta, alpha) -
                                                H(l + 1, theta, alpha)) *
                          sin(m * phi))
    return result

@cython.boundscheck(False)
def _h_ml_sum_cy(float hotspot_offset, float omega_drag, float alpha,
                double [:, :] theta2d, double [:, :] phi2d, list C, int lmax):
    """
    Cythonized implementation of the quadruple loop over: theta's, phi's,
    l's and m's to compute the h_ml_sum term at C speeds
    """
    cdef Py_ssize_t theta_max = theta2d.shape[1]
    cdef Py_ssize_t phi_max = phi2d.shape[0]
    cdef Py_ssize_t l, m, i, j
    cdef float Cml, phase_offset = pi / 2
    cdef DTYPE_t tmp
    hml_sum = np.zeros((theta_max, phi_max), dtype=DTYPE)
    cdef double [:, ::1] h_ml_sum_view = hml_sum

    for l in range(1, lmax + 1):
        for m in range(-l, l + 1):
            Cml = C[l][m]
            if Cml != 0:
                for i in prange(phi_max, nogil=True):
                    for j in range(theta_max):
                        tmp = h_ml_cython(omega_drag, alpha,
                                          m, l, theta2d[i, j],
                                          phi2d[i, j] +
                                          phase_offset +
                                          hotspot_offset,
                                          Cml)
                        h_ml_sum_view[j, i] = h_ml_sum_view[j, i] + tmp
    return hml_sum

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef float blackbody_lambda(float lam, float temperature) nogil:
    """
    Compute the blackbody flux as a function of wavelength `lam` in mks units
    """
    cdef float h = 6.62607015e-34  # J s
    cdef float c = 299792458.0  # m/s
    cdef float k_B = 1.380649e-23  # J/K

    return (2 * h * c**2 / lam**5 /
            (exp(h * c / (lam * k_B * temperature)) - 1))

def bl_test(float lam, float temperature):
    return blackbody_lambda(lam, temperature)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef blackbody(double [:] wavelengths, float temperature):
    """
    Planck function evaluated for a vector of wavelengths in units of meters
    and temperature in units of Kelvin

    Parameters
    ----------
    wavelengths : `~numpy.ndarray`
        Wavelength array in units of meters
    temperature : float
        Temperature in units of Kelvin

    Returns
    -------
    pl : `~numpy.ndarray`
        Planck function evaluated at each wavelength
    """
    cdef Py_ssize_t i, n=len(wavelengths)
    planck = np.zeros(n, dtype=DTYPE)
    cdef double [::1] planck_view = planck

    for i in prange(n, nogil=True):
        planck_view[i] = blackbody_lambda(wavelengths[i], temperature)

    return planck

@cython.boundscheck(False)
@cython.wraparound(False)
cdef blackbody2d(double [:] wavelengths, double [:, :] temperature):
    """
    Planck function evaluated for a vector of wavelengths in units of meters
    and temperature in units of Kelvin

    Parameters
    ----------
    wavelengths : `~numpy.ndarray`
        Wavelength array in units of meters
    temperature : `~numpy.ndarray`
        Temperature in units of Kelvin

    Returns
    -------
    pl : `~numpy.ndarray`
        Planck function evaluated at each wavelength
    """
    cdef int i, j, k, l=temperature.shape[0], m=temperature.shape[1], n=len(wavelengths)
    cdef np.ndarray[DTYPE_t, ndim=3] planck = np.zeros((n, l, m), dtype=DTYPE)
    cdef double [:, :, :] planck_view = planck

    for i in prange(n, nogil=True):
        for j in range(l):
            for k in range(m):
                planck_view[i, j, k] = blackbody_lambda(wavelengths[i],
                                                        temperature[j, k])

    return planck

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef float trapz(double [:] y, double [:] x):
    """
    Pure cython version of trapezoid rule
    """
    cdef Py_ssize_t i, n = len(x)
    cdef float s = 0

    for i in range(1, n):
        s += (x[i] - x[i-1]) * (y[i] + y[i-1]) / 2
    return s

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef trapz3d(double [:, :, :] y_3d, double [:] x):
    """
    Pure cython version of trapezoid rule in ~more dimensions~
    """
    cdef int i, j, k, l = len(x), m = y_3d.shape[1], n = y_3d.shape[2]

    s = np.zeros((m, n), dtype=DTYPE)
    cdef double [:, ::1] s_view = s

    for i in prange(1, l, nogil=True):
        for k in range(m):
            for j in range(n):
                s_view[k, j] += ((x[i] - x[i-1]) *
                                 (y_3d[i, k, j] + y_3d[i-1, k, j]) / 2)
    return s

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int argmin_lowest(double [:] arr, float value):
    """
    Return the index of `arr` which is closest to *and less than* `value`
    """
    cdef int i, min_ind = 0, n = len(arr)
    cdef float dist, min_dist = 1e10

    for i in range(n):
        dist = abs(arr[i] - value)
        if dist < min_dist and value > arr[i]:
            min_dist = dist
            min_ind = i

    return min_ind

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int argmin(double [:] arr, float value):
    """
    Return the index of `arr` which is closest to `value`
    """
    cdef int i, min_ind = 0, n = len(arr)
    cdef float dist, min_dist = 1e10

    for i in range(n):
        dist = abs(arr[i] - value)
        if dist < min_dist:
            min_dist = dist
            min_ind = i

    return min_ind

def argmin_test(arr, value):
    return argmin(arr, value)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef float bilinear_interpolate(double [:, :] im,
                                double [:] x_grid, double [:] y_grid,
                                float x, float y):
    """
    Bilinear interpolation over an image `im` which is computed on grid
    `x_grid` vs `y_grid`, evaluated at position (x, y).

    Source: https://stackoverflow.com/a/12729229/1340208
    """
    cdef int minind0, minind1
    cdef float x0, x1, y0, y1
    cdef float Ia, Ib, Ic, Id, wa, wb, wc, wd
    minind0 = min([argmin_lowest(x_grid, x), len(x_grid) - 1])
    minind1 = min([argmin_lowest(y_grid, y), len(y_grid) - 1])

    x0 = x_grid[minind0]
    x1 = x_grid[minind0 + 1]
    y0 = y_grid[minind1]
    y1 = y_grid[minind1 + 1]

    Ia = im[minind0, minind1]
    Ib = im[minind0, minind1 + 1]
    Ic = im[minind0 + 1, minind1]
    Id = im[minind0 + 1, minind1 + 1]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return (wa*Ia + wb*Ib + wc*Ic + wd*Id) / ((x1 - x0) * (y1 - y0))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def _integrate_planck(double [:] filt_wavelength, double [:] filt_trans,
                     double [:, :] temperature, double [:, :] T_s,
                     double [:] theta_grid, double [:] phi_grid, float rp_rs,
                     int n_phi, bint return_interp=True):
    """
    Integrate the Planck function over wavelength for the temperature map of the
    planet `temperature` and the temperature of the host star `T_s`. If
    `return_interp`, returns the interpolation function for the integral over
    the ratio of the blackbodies over wavelength; else returns only the map
    (which can be used for trapezoidal approximation integration)
    """

    cdef int i, j, k
    cdef Py_ssize_t l = len(filt_wavelength), m = len(theta_grid), n = len(phi_grid)
    cdef np.ndarray[DTYPE_t, ndim=3] bb = np.zeros((l, m, n), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3] bb_num = blackbody2d(filt_wavelength, temperature)
    cdef np.ndarray[DTYPE_t, ndim=3] broadcast_trans = filt_trans[:, None, None] * np.ones((l, m, n), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] int_bb_num = trapz3d(bb_num * broadcast_trans, filt_wavelength)
    cdef np.ndarray[DTYPE_t, ndim=2] int_bb = int_bb_num

    if return_interp:
        def interp(theta, phi, theta_grid=theta_grid, phi_grid=phi_grid,
                   int_bb=int_bb):
            return bilinear_interpolate(int_bb, theta_grid, phi_grid,
                                        theta, phi)
        return int_bb, interp
    else:
        return int_bb

@cython.boundscheck(False)
@cython.wraparound(False)
def _integrated_blackbody(float hotspot_offset, float omega_drag,
                         float alpha, list C_ml,
                         int lmax, float T_s, float a_rs, float rp_a, float A_B,
                         int n_theta, int n_phi, double [:] filt_wavelength,
                         double [:] filt_transmittance, float f=2**-0.5):
    """
    Compute the temperature field using `_h_ml_sum_cy`, then integrate the
    temperature map over wavelength and take the ratio of blackbodies with
    `_integrate_planck`
    """
    cdef float T_eq, rp_rs
    cdef np.ndarray[DTYPE_t, ndim=1] phi = np.linspace(-2 * pi, 2 * pi, n_phi, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] theta = np.linspace(0, pi, n_theta, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] theta2d = np.zeros((n_theta, n_phi), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] phi2d = np.zeros((n_theta, n_phi), dtype=DTYPE)

    theta2d, phi2d = np.meshgrid(theta, phi)

    # Cython alternative to the pure python implementation:
    cdef np.ndarray[DTYPE_t, ndim=2] h_ml_sum = _h_ml_sum_cy(hotspot_offset,
                                                            omega_drag,
                                                            alpha, theta2d,
                                                            phi2d, C_ml, lmax)
    T_eq = f * T_s * a_rs**-0.5

    cdef np.ndarray[DTYPE_t, ndim=2] T = T_eq * (1 - A_B)**0.25 * (1 + h_ml_sum)

    rp_rs = rp_a * a_rs

    int_bb, func = _integrate_planck(filt_wavelength,
                                    filt_transmittance, T,
                                    T_s * np.ones_like(T),
                                    theta, phi, rp_rs, n_phi)

    return int_bb, func

@cython.boundscheck(False)
@cython.wraparound(False)
def _phase_curve(double [:] xi, float hotspot_offset, float omega_drag,
                 float alpha, list C_ml,
                 int lmax, float T_s, float a_rs, float rp_a, float A_B,
                 int n_theta, int n_phi, double [:] filt_wavelength,
                 double [:] filt_transmittance, float f,
                 double [:] stellar_spectrum_wavelength,
                 double [:] stellar_spectrum_spectral_flux_density):
    """
    Compute the phase curve evaluated at phases `xi`.
    """
    cdef float T_eq, rp_rs
    cdef DTYPE_t integral
    cdef int k, phi_min, phi_max, n_xi = len(xi)
    cdef np.ndarray[DTYPE_t, ndim=1] phi = np.linspace(-2 * pi, 2 * pi, n_phi, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] theta = np.linspace(0, pi, n_theta, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] fluxes = np.zeros(n_xi, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] theta2d = np.zeros((n_theta, n_phi), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] phi2d = np.zeros((n_theta, n_phi), dtype=DTYPE)

    theta2d, phi2d = np.meshgrid(theta, phi)

    # If stellar spectrum is supplied, interpolate it into the same wavelength
    # grid as the filter bandpass, otherwise assume it is a Planck function.
    cdef np.ndarray[DTYPE_t, ndim=1] stellar_spectrum = blackbody(
        filt_wavelength, T_s
    )

    if np.count_nonzero(stellar_spectrum_spectral_flux_density) > 0:
         stellar_spectrum = np.interp(
            filt_wavelength, stellar_spectrum_wavelength,
            stellar_spectrum_spectral_flux_density
        )

    cdef double [::1] fluxes_view = fluxes

    # Cython alternative to the pure python implementation:
    h_ml_sum = _h_ml_sum_cy(hotspot_offset, omega_drag,
                           alpha, theta2d, phi2d, C_ml,
                           lmax)
    T_eq = f * T_s * a_rs**-0.5

    T = T_eq * (1 - A_B)**0.25 * (1 + h_ml_sum)

    rp_rs = rp_a * a_rs
    cdef np.ndarray[DTYPE_t, ndim=2] ones = np.ones((n_theta, n_phi), dtype=DTYPE)
    int_bb = _integrate_planck(filt_wavelength,
                              filt_transmittance, T,
                              T_s * ones,
                              theta, phi, rp_rs, n_phi,
                              return_interp=False).T
    cdef double [:, :] int_bb_view = int_bb
    cdef double [:] xi_view = xi

    cdef DTYPE_t planck_star = trapz(filt_transmittance *
                                     stellar_spectrum,
                                     filt_wavelength)

    for k in range(n_xi):
        phi_min = argmin(phi, -xi_view[k] - pi/2)
        phi_max = argmin(phi, -xi_view[k] + pi/2)
        integral = trapz2d((int_bb_view[phi_min:phi_max] *
                           sinsq_2d(theta2d[phi_min:phi_max]) *
                           cos_2d(phi2d[phi_min:phi_max] + xi_view[k])),
                           phi[phi_min:phi_max], theta)

        fluxes_view[k] = integral * rp_rs**2 / pi / planck_star
    return fluxes

@cython.boundscheck(False)
@cython.wraparound(False)
cdef float sum2d(double [:, :] z):
    """
    Sum a 2d array over its axes
    """
    cdef int m = z.shape[0], n = z.shape[1]
    cdef float s = 0

    for i in range(m):
        for j in range(n):
            s += z[i, j]
    return s

@cython.boundscheck(False)
@cython.wraparound(False)
cdef float sum1d(double [:] z):
    """
    Sum a 1d array over its first axis
    """
    cdef int m = z.shape[0]
    cdef float s = 0

    for i in range(m):
        s += z[i]
    return s

@cython.boundscheck(False)
@cython.wraparound(False)
cdef sinsq_2d(double [:, :] z):
    """
    The square of the sine of a 2d array
    """
    cdef int m = z.shape[0], n = z.shape[1]
    cdef np.ndarray s = np.zeros((m, n), dtype=DTYPE)
    cdef double [:, :] s_view = s

    for i in range(m):
        for j in range(n):
            s_view[i, j] = sin(z[i, j])**2
    return s


@cython.boundscheck(False)
@cython.wraparound(False)
cdef cos_2d(double [:, :] z):
    """
    The cosine of a 2d array
    """
    cdef int m = z.shape[0], n = z.shape[1]
    cdef np.ndarray s = np.zeros((m, n), dtype=DTYPE)
    cdef double [:, :] s_view = s

    for i in range(m):
        for j in range(n):
            s_view[i, j] = cos(z[i, j])
    return s


@cython.boundscheck(False)
@cython.wraparound(False)
cdef float trapz2d(double [:, :] z, double [:] x, double [:] y):
    """
    Integrates a regularly spaced 2D grid using the composite trapezium rule.
    
    Source: https://github.com/tiagopereira/python_tips/blob/master/code/trapz2d.py
    
    Parameters
    ----------
    z : `~numpy.ndarray`
        2D array
    x : `~numpy.ndarray`
        grid values for x (1D array)
    y : `~numpy.ndarray`
        grid values for y (1D array)
    
    Returns
    -------
    t : `~numpy.ndarray`
        Trapezoidal approximation to the integral under z
        """
    cdef float dx, dy, s1, s2, s3
    cdef int m = z.shape[0] - 1, n = z.shape[1] - 1
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    s1 = z[0, 0] + z[m, 0] + z[0, n] + z[m, n]
    s2 = sum1d(z[1:m, 0]) + sum1d(z[1:m, n]) + sum1d(z[0, 1:n]) + sum1d(z[m, 1:n])
    s3 = sum2d(z[1:m, 1:n])

    return 0.25 * dx * dy * (s1 + 2 * s2 + 4 * s3)

def trapz2d_test(double [:,:] z , double [:] x, double [:] y):
    return trapz2d(z, x, y)