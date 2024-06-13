#! python3
# coding: utf-8

import os, os.path
from PIL import (
    Image as IMG, ImageChops as IMGCHPS,
    ImageDraw as IMGDRW, ImageFilter as IMGFLT)

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

class c_img_merger:

    def __init__(self, imgs_fn):
        self.imgs = []
        self.load_imgs(imgs_fn)

    def load_imgs(self, imgs_fn):
        for fn in imgs_fn:
            self.imgs.append(
                IMG.open(fn))

    def _hint_rect(self, im, bx, clr=(255, 0, 0, 100)):
        cim = im.convert('RGBA')
        him = IMG.new('RGBA', cim.size)
        dr = IMGDRW.Draw(him)
        dr.rectangle((bx[0], bx[1], bx[2], bx[3]), fill=clr)
        cim.alpha_composite(him)
        cim.show()
        return cim

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
        #print(th - wh / 2, tw - ww / 2)
        return th - wh / 2, tw - ww / 2

    def _cmp_imgs(self, im1, im2, wrct):
        src = np.array(im1)
        dst = np.array(im2)
        wx, wy, wr, wb = wrct
        sy, sx = self._cmp_window(src, dst[wy:wb, wx:wr, ...])
        return sx - wx, sy - wy + 1

    def _cmp_cover_axis(self, dif, rng, axs):
        dsum = dif.sum(1 - axs)
        dd = np.diff(dsum)
        dd = np.insert(dd, 0, 0)
        dd = np.append(dd, 0)
        from matplotlib import pyplot as plt
        b, a = scipy.signal.butter(3, 0.1)
        dsumf = scipy.signal.filtfilt(b, a, dsum)
        plt.plot(dsum)
        plt.plot(dsumf)
        #plt.plot(dd)
        #plt.plot(np.diff(dsumf))
        #plt.plot(np.cumsum(dsum))
        plt.show()
        breakpoint()
        pks_f = scipy.signal.find_peaks(np.clip(-dd, 0, None))[0]
        pks_b = scipy.signal.find_peaks(np.clip(dd, 0, None))[0]
        dirty = False
        crng = [0, len(dsum)]
        if len(pks_b):
            pk = pks_b[-1] - 1
            crng[1] = pk
            rng[axs][1] = rng[axs][0] + pk
            dirty = True
        if len(pks_f):
            pk = pks_f[0]
            crng[0] = pk
            rng[axs][0] += pk
            dirty = True
        if dirty:
            crng_idx = [..., ...]
            crng_idx[axs] = slice(*crng)
            print(f'trim {axs} to rng: {rng}')
            return self._cmp_cover_axis(
                dif[tuple(crng_idx)], rng, 1 - axs)
        else:
            return rng

    def _cmp_cover_center(self, dif, cent, axis, step, thr):
        alen = dif.shape[axis]
        raxis = 1 - axis
        rlen = dif.shape[raxis]
        cur_idx = [slice(None),slice(None)]
        cur_cent = cent.copy()
        cur = cent[axis]
        rcent = cent[raxis]
        rseq = []
        while 0 <= cur < alen:
            cur_idx[axis] = cur
            cur_cent[axis] = cur
            if dif[tuple(cur_cent)] > thr:
                break
            row = dif[tuple(cur_idx)]
            rpair = []
            for rstep in (1, -1):
                rcur = rcent
                rcur_pos = cur_cent.copy()
                rv = 0
                while 0 <= rcur < rlen:
                    rcur_pos[raxis] = rcur
                    if dif[tuple(rcur_pos)] > thr:
                        break
                    rv += 1
                    rcur += rstep
                rpair.append(rv)
            rseq.append(rpair)
            cur += step
        return rseq

    def _cmp_cover(self, src, dst, thr = 0):
        cdif = IMGCHPS.difference(src, dst)
        #cdif.show()
        dif = np.array(cdif)
        fdif = np.sum(dif.astype('float'), axis=2) / 3
        b, a = scipy.signal.butter(3, 0.03)
        fdif = scipy.signal.filtfilt(b, a, fdif, 1)
        fdif = scipy.signal.filtfilt(b, a, fdif, 0)
        print('max dif:', fdif[50:,...].max() * 3)
        tdif = IMG.fromarray(fdif * 10)
        #tdif = tdif.filter(IMGFLT.SMOOTH)
        tdif.show()
        dif = np.sum(dif.astype('float'), axis=2) #/ (255 * 3)
        #print('max dif:', dif[50:,...].max())
        dif[dif<=thr] = 0
        dif[dif>thr] = 1
        #IMG.fromarray(dif * 255).show()
        #rng = [[0, dif.shape[0]], [0, dif.shape[1]]]
        #crng = self._cmp_cover_axis(dif, rng, 0)
        rseq = self._cmp_cover_center(dif, np.array(dif.shape) // 2, 0, -1, 0)
        #breakpoint()
        #self._hint_rect(dst, (
        #    crng[1][0], crng[0][0],
        #    crng[1][1], crng[0][1]))

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
        rim = IMG.new('RGB', rsz)
        rim.paste(im1, (sx1, sy1))
        cvsim = rim.crop((sx2+wrct[0], sy2+wrct[1], sx2+wrct[2], sy2+wrct[3]))
        cvim2 = im2.crop(wrct)
        self._cmp_cover(cvsim, cvim2, 20)
        if alpha2 is None:
            rim.paste(im2, (sx2, sy2))
        else:
            rim = rim.convert('RGBA')
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
        wrct = [hl, ht, hr, hb]
        if wsz[0] is None and wsz[1] is None:
            cim2 = im2
        else:
            crp2 = [0, 0, *im2.size]
            wx, wy = wsz
            if wx is not None:
                if isinstance(wx, (tuple, list)):
                    wxb = wx[0]
                    wx = wx[1]
                    wrct[0] += wxb
                if wx >= 1:
                    crp2[0] = hl
                    wrct[0] = wxb
                    wrct[2] = min(wrct[2], wxb + wx)
                else:
                    dww = int((wrct[2] - wrct[0]) * (1 - wx) / 2)
                    wrct[0] += dww
                    wrct[2] -= dww
            if wy is not None:
                if isinstance(wy, (tuple, list)):
                    wyb = wy[0]
                    wy = wy[1]
                    wrct[1] += wyb
                if wy >= 1:
                    crp2[1] = ht
                    wrct[1] = wyb
                    wrct[3] = min(wrct[3], wyb + wy)
                else:
                    dwh = int((wrct[3] - wrct[1]) * (1 - wy) / 2)
                    wrct[1] += dwh
                    wrct[3] -= dwh
            cim2 = im2.crop(crp2)
            #self._hint_rect(cim2, wrct)
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
    r = im._img_merge(im.imgs[1], im.imgs[2], (0.8, (0, 307)))
