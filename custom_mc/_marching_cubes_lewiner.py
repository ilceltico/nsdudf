import base64
import numpy as np
import _marching_cubes_lewiner_luts as mcluts
import _marching_cubes_lewiner_cy_pseudosdf


def pseudosdf_mc_lewiner(pseudosdf, spacing=(1., 1., 1.),
                           gradient_direction='descent', step_size=1,
                           allow_degenerate=True, use_classic=False):
    """Lewiner et al. algorithm for marching cubes. See
    marching_cubes_lewiner for documentation. This slightly modified version takes inputs with one extra dimension, which is required by the output of our network.
    """

    # Check volume and ensure its in the format that the alg needs
    if not isinstance(pseudosdf, np.ndarray) or (pseudosdf.ndim != 4):
        raise ValueError('Input volume should be a 4D numpy array.')
    if pseudosdf.shape[0] < 1 or pseudosdf.shape[1] < 1 or pseudosdf.shape[2] < 1:
        raise ValueError("Input array must be at least 1x1x1x8.")
    if pseudosdf.shape[3] != 8:
        raise ValueError("Input array must have last dimension = 8")
    pseudosdf = np.ascontiguousarray(pseudosdf,
                                  np.float32)  # no copy if not necessary

    # spacing
    if len(spacing) != 3:
        raise ValueError("`spacing` must consist of three floats.")
    # step_size
    step_size = int(step_size)
    if step_size < 1:
        raise ValueError('step_size must be at least one.')
    # use_classic
    use_classic = bool(use_classic)

    # Get LutProvider class (reuse if possible)
    L = _get_mc_luts_pseudosdf()

    # Check if a mask array is passed
    # if mask is not None:
    #     if not mask.shape == pseudosdf.shape:
    #         raise ValueError('volume and mask must have the same shape.')

    # Apply algorithm
    func = _marching_cubes_lewiner_cy_pseudosdf.marching_cubes_pseudosdf
    # vertices, faces, normals, values = func(pseudosdf, grads, L,
    #                                         step_size, use_classic, mask)

    # import pdb
    # pdb.set_trace()

    vertices, faces, normals, values = func(pseudosdf, L,
                                            step_size, use_classic)


    if not len(vertices):
        raise RuntimeError('No surface found at the given iso value.')

    # Output in z-y-x order, as is common in skimage
    vertices = np.fliplr(vertices)
    normals = np.fliplr(normals)

    # Finishing touches to output
    faces.shape = -1, 3
    if gradient_direction == 'descent':
        # MC implementation is right-handed, but gradient_direction is
        # left-handed
        faces = np.fliplr(faces)
    elif not gradient_direction == 'ascent':
        raise ValueError("Incorrect input %s in `gradient_direction`, see "
                         "docstring." % (gradient_direction))
    if not np.array_equal(spacing, (1, 1, 1)):
        vertices = vertices * np.r_[spacing]

    if allow_degenerate:
        return vertices, faces, normals, values#, cur_cubes
    else:
        fun = _marching_cubes_lewiner_cy.remove_degenerate_faces
        return fun(vertices.astype(np.float32), faces, normals, values)



def _to_array(args):
    shape, text = args
    byts = base64.decodebytes(text.encode('utf-8'))
    ar = np.frombuffer(byts, dtype='int8')
    ar.shape = shape
    return ar

 
# Map an edge-index to two relative pixel positions. The ege index
# represents a point that lies somewhere in between these pixels.
# Linear interpolation should be used to determine where it is exactly.
#   0
# 3   1   ->  0x
#   2         xx
EDGETORELATIVEPOSX = np.array([ [0,1],[1,1],[1,0],[0,0], [0,1],[1,1],[1,0],[0,0], [0,0],[1,1],[1,1],[0,0] ], 'int8')
EDGETORELATIVEPOSY = np.array([ [0,0],[0,1],[1,1],[1,0], [0,0],[0,1],[1,1],[1,0], [0,0],[0,0],[1,1],[1,1] ], 'int8')
EDGETORELATIVEPOSZ = np.array([ [0,0],[0,0],[0,0],[0,0], [1,1],[1,1],[1,1],[1,1], [0,1],[0,1],[0,1],[0,1] ], 'int8')


def _get_mc_luts_pseudosdf():
    """ Kind of lazy obtaining of the luts.
    """
    if not hasattr(mcluts, 'THE_LUTS_PSEUDOSDF'):

        mcluts.THE_LUTS_PSEUDOSDF = _marching_cubes_lewiner_cy_pseudosdf.LutProvider(
                EDGETORELATIVEPOSX, EDGETORELATIVEPOSY, EDGETORELATIVEPOSZ,

                _to_array(mcluts.CASESCLASSIC), _to_array(mcluts.CASES),

                _to_array(mcluts.TILING1), _to_array(mcluts.TILING2), _to_array(mcluts.TILING3_1), _to_array(mcluts.TILING3_2),
                _to_array(mcluts.TILING4_1), _to_array(mcluts.TILING4_2), _to_array(mcluts.TILING5), _to_array(mcluts.TILING6_1_1),
                _to_array(mcluts.TILING6_1_2), _to_array(mcluts.TILING6_2), _to_array(mcluts.TILING7_1),
                _to_array(mcluts.TILING7_2), _to_array(mcluts.TILING7_3), _to_array(mcluts.TILING7_4_1),
                _to_array(mcluts.TILING7_4_2), _to_array(mcluts.TILING8), _to_array(mcluts.TILING9),
                _to_array(mcluts.TILING10_1_1), _to_array(mcluts.TILING10_1_1_), _to_array(mcluts.TILING10_1_2),
                _to_array(mcluts.TILING10_2), _to_array(mcluts.TILING10_2_), _to_array(mcluts.TILING11),
                _to_array(mcluts.TILING12_1_1), _to_array(mcluts.TILING12_1_1_), _to_array(mcluts.TILING12_1_2),
                _to_array(mcluts.TILING12_2), _to_array(mcluts.TILING12_2_), _to_array(mcluts.TILING13_1),
                _to_array(mcluts.TILING13_1_), _to_array(mcluts.TILING13_2), _to_array(mcluts.TILING13_2_),
                _to_array(mcluts.TILING13_3), _to_array(mcluts.TILING13_3_), _to_array(mcluts.TILING13_4),
                _to_array(mcluts.TILING13_5_1), _to_array(mcluts.TILING13_5_2), _to_array(mcluts.TILING14),

                _to_array(mcluts.TEST3), _to_array(mcluts.TEST4), _to_array(mcluts.TEST6),
                _to_array(mcluts.TEST7), _to_array(mcluts.TEST10), _to_array(mcluts.TEST12),
                _to_array(mcluts.TEST13), _to_array(mcluts.SUBCONFIG13),
                )

    return mcluts.THE_LUTS_PSEUDOSDF
