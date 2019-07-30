import numpy as np
import sys
from scipy.ndimage.filters import gaussian_filter
from scipy.misc import imresize # preserves single-pixel info _unlike_ img = img[::2,::2]


occlude = lambda I, mask: I*(1-mask) + gaussian_filter(I, sigma=3)*mask # choose an area to blur
searchlight = lambda I, mask: I*mask + gaussian_filter(I, sigma=3)*(1-mask) # choose an area NOT to blur

def get_mask(center, size, r):
    y,x = np.ogrid[-center[0]:size[0]-center[0], -center[1]:size[1]-center[1]]
    keep = x*x + y*y <= 1
    mask = np.zeros(size) ; mask[keep] = 1 # select a circle of pixels
    mask = gaussian_filter(mask, sigma=r) # blur the circle of pixels. this is a 2D Gaussian for r=r^2=1
    return mask/mask.max()


def score_frame(agent, logits, state, r=8, d=8):
    [y, x, w] = agent.states['state']['shape']

    num = len(logits)

    names = logits.keys()
    scores = np.zeros((int(y/d)+1,int(x/d)+1, num))
    scoresResized = np.zeros((y,x, num))

    for i in range(0, y, d):
        for j in range(0, x, d):
           
            mask = get_mask(center=[i,j], size=[y,x], r=r)
            mask = np.dstack([mask]*3)
            im = occlude(state, mask)
            
            _, newLogits = agent.act(states=im, deterministic=False)
            
            for k in range(0,num):
                original = logits[names[k]][0,:]
                altered = newLogits[names[k]][0,:]
                scores[int(i/d), int(j/d), k] = (0.5*np.sum(np.power(original-altered, 2)))

    for i in range(0,num):
        pmaxM = np.amax(scores[:,:,i])
        scoresResized[:,:,i] = imresize(scores[:,:,i], size=[y, x], interp='bilinear').astype(np.float32)
        scoresResized[:,:,i] = pmaxM * scoresResized[:,:,i] / np.amax(scoresResized[:,:,i])
    
    return scoresResized


def salienize(saliency, frame, fudge=3000, channel=2, sigma=0, normVal=1):
    pmax = np.amax(saliency)
    S =  saliency.astype(np.float32)
    S = S if sigma == 0 else gaussian_filter(S, sigma=sigma)
    S-= S.min() ; S = fudge*pmax * S / np.amax(S)


    if normVal != 1:
	S = normVal * S / np.amax(S)
    I = (frame).astype('uint16')
    print('Max I: ' + str(np.amax(I[:,:,channel])))
    I[:,:, channel] += S.astype('uint16')
    print('Max I2: ' + str(np.amax(I[:,:,channel])))
    I = I.clip(1,255).astype('uint8')
    return I
