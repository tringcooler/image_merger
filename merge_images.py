#! python3
# coding: utf-8

import os, os.path
from PIL import Image as IMG, ImageChops as IMGCHPS, ImageDraw as IMGDRW

import numpy as np
import scipy.signal

# from https://stackoverflow.com/questions/24768222/how-to-detect-a-shift-between-images
def cross_image(im1, im2, exact=True):
    # get rid of the color channels by performing a grayscale transform
    # the type cast into 'float' is to avoid overflows
    im1_gray = np.sum(im1.astype('float'), axis=2)
    im2_gray = np.sum(im2.astype('float'), axis=2)

    if not exact:
        # get rid of the averages, otherwise the results are not good
        im1_gray -= np.mean(im1_gray)
        im2_gray -= np.mean(im2_gray)

    # calculate the correlation image; note the flipping of onw of the images
    return scipy.signal.fftconvolve(im1_gray, im2_gray[::-1,::-1], mode='same')
    #return scipy.signal.fftconvolve(im1_gray[::-1,::-1], im2_gray, mode='same')[::-1,::-1]

# from https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array
# failed, not used
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
def detect_peaks(image):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """

    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2,2)

    #apply the local maximum filter; all pixel of maximal value 
    #in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood)==image
    #local_max is a mask that contains the peaks we are 
    #looking for, but also the background.
    #In order to isolate the peaks we must remove the background from the mask.

    #we create the mask of the background
    background = (image==0)

    #a little technicality: we must erode the background in order to 
    #successfully subtract it form local_max, otherwise a line will 
    #appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    #we obtain the final mask, containing only peaks, 
    #by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background

    return detected_peaks

class c_img_merger:

    def __init__(self, imgs_fn):
        self.imgs = []
        self.load_imgs(imgs_fn)

    def load_imgs(self, imgs_fn):
        for fn in imgs_fn:
            self.imgs.append(
                IMG.open(fn))

    def _cmp_window(self, src, win):
        sh, sw, *_ = src.shape
        wh, ww, *_ = win.shape
        assert sh >= wh and sw >= ww
        dif = cross_image(src, win, True)
        th, tw = np.unravel_index(np.argmax(dif), dif.shape)
        #print(win.T.shape)
        #print(dif.T.shape, tw, th)
        #ndif = dif / dif.max() * 255
        #pks = ndif[scipy.signal.find_peaks(ndif[:,tw])[0], tw]
        #pks = [(i - wh / 2, ndif[i, tw])for i in scipy.signal.find_peaks(ndif[:,tw])[0] if ndif[i, tw] > 250]
        #print(pks)
        #pks = detect_peaks(ndif) # not used
        #print(np.where(pks))
        #print(th - wh / 2, tw - ww / 2)
        return th - wh / 2, tw - ww / 2

    def _cmp_imgs(self, im1, im2, wrct):
        src = np.array(im1)
        dst = np.array(im2)
        wx, wy, ww, wh = wrct
        wx = 0 if wx is None else wx
        wy = 0 if wy is None else wy
        wr = None if ww is None else wx + ww
        wb = None if wh is None else wy + wh
        sy, sx = self._cmp_window(src, dst[wy:wb, wx:wr, ...])
        return sx - wx, sy - wy

    def _img_paste(self, im1, im2, wrct, alpha2=None):
        sx, sy = self._cmp_imgs(im1, im2, wrct)
        print(sx, sy)
        if sx < 0:
            sx1 = -int(sx)
            sx2 = 0
        else:
            sx1 = 0
            sx2 = int(sx)
        if sy < 0:
            sy1 = -int(sy)
            sy2 = 0
        else:
            sy1 = 0
            sy2 = int(sy)
        rsz = (
            max(sx1 + im1.size[0], sx2 + im2.size[0]),
            max(sy1 + im1.size[1], sy2 + im2.size[1]))
        rim = IMG.new('RGBA', rsz)
        rim.paste(im1, (sx1, sy1))
        if alpha2 is None:
            rim.paste(im2, (sx2, sy2))
        else:
            im2a = IMG.new('L', im2.size, alpha2)
            im2c = im2.copy()
            im2c.putalpha(im2a)
            rim.alpha_composite(im2c, (sx2, sy2))
        return rim

    def _img_trim_head(self, im1, im2):
        dif = IMGCHPS.difference(im1, im2)
        dbb = dif.getbbox()
        if not dbb:
            return None
        return dbb

    def _img_merge(self, im1, im2, wsz):
        hbb = self._img_trim_head(im1, im2)
        if not hbb:
            return
        hl, ht, hr, hb = hbb
        wrct = [hl, ht, hr - hl, hb - ht]
        if wsz[0] is None and wsz[1] is None:
            cim2 = im2
        else:
            crp2 = [0, 0, *im2.size]
            wx, wy = wsz
            if wx is not None:
                if wx >= 1:
                    crp2[0] = hl
                    wrct[0] = 0
                    wrct[2] = min(wrct[2], wx)
                else:
                    nwh = wrct[2] * wx
                    wrct[0] += int((wrct[2] - nwh) / 2)
                    wrct[2] = int(nwh)
            if wy is not None:
                if wy >= 1:
                    crp2[1] = ht
                    wrct[1] = 0
                    wrct[3] = min(wrct[3], wy)
                else:
                    nww = wrct[3] * wy
                    wrct[1] += int((wrct[3] - nww) / 2)
                    wrct[3] = int(nww)
            cim2 = im2.crop(crp2)
            #him2c = cim2.convert('RGBA')
            #him2 = IMG.new('RGBA', im2.size)
            #dr = IMGDRW.Draw(him2)
            #dr.rectangle((wrct[0], wrct[1], wrct[2] + wrct[0], wrct[3] + wrct[1]), fill=(255, 0, 0, 100))
            #him2c.alpha_composite(him2)
            #him2c.show()
            #print(crp2)
        #wrct[1] = 24
        #print(wrct)
        return self._img_paste(im1, cim2, wrct, 100)

def iter_imgs(path, ext = None):
    for fn in os.listdir(path):
        fn = os.path.join(path, fn)
        if not os.path.isfile(fn):
            continue
        if ext is not None and not os.path.splitext(fn)[1].endswith(ext):
            continue
        yield fn

if __name__ == '__main__':

    wpath = r'E:\temp\tabletmp\merge_imgs'

    #dif = cross_image(foo, bar[200:600,200:,...])
    #top = np.unravel_index(np.argmax(dif), dif.shape)
    #ndif = dif / dif.max() * 255
    #pks = ndif[scipy.signal.find_peaks(ndif[:,1300])[0], 1300]

    def main(path):
        im = c_img_merger(list(iter_imgs(path, 'jpg')))
        return im
    
    im = main(wpath)
    r = im._img_merge(im.imgs[0], im.imgs[1], (0.8, 307))
