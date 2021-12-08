import spams

from staintools.utils.optical_density_conversion import convert_RGB_to_OD


def get_concentrations(I, stain_matrix, regularizer=0.01, is_parallel=True):
    """
    Estimate concentration matrix given an image and stain matrix.

    :param I:
    :param stain_matrix:
    :param regularizer:
    :return:
    """
    
    numThreads = -1 if is_parallel else 1
    OD = convert_RGB_to_OD(I).reshape((-1, 3))
    return spams.lasso(X=OD.T, D=stain_matrix.T, mode=2, lambda1=regularizer, pos=True, numThreads=numThreads).toarray().T
