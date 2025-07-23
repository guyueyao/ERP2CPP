import numpy as np
class ERP2CPP:
    LANCZOS_TAB_SIZE=3
    LANCZOS_FAST_SCALE=100
    LANCZOS_FAST_MAX_SIZE=LANCZOS_TAB_SIZE*2

    def __init__(self,interpolation='nearest'):
        """
        :param interpolation: Used pixel interpolation algorithm, 'nearest' or 'lanczos'. Default is 'nearest'.
        """
        self.m_cppMap = None
        self.m_lanczosCoef=None
        assert interpolation in ('nearest','lanczos')
        if interpolation == 'lanczos':
            self.InitLanczosCoef()
            self.interpolation=self.IFilterLanczos
        else:
            self.interpolation=self.IFilterNearest

    def InitLanczosCoef(self):
        if self.m_lanczosCoef is not None and len(self.m_lanczosCoef) > 0:
            return
        size = self.LANCZOS_FAST_MAX_SIZE*self.LANCZOS_FAST_SCALE
        x = np.arange(0,size) / self.LANCZOS_FAST_SCALE
        self.m_lanczosCoef = np.sinc(x) * np.sinc(x / self.LANCZOS_TAB_SIZE)

    def lanczos_coef(self, x):
        idx = (np.abs(x) * self.LANCZOS_FAST_SCALE+0.5).astype(int)
        return self.m_lanczosCoef[idx]


    def IFilterLanczos(self, img, x_in, y_in):
        h, w, _ = img.shape
        x_in=x_in.flatten().astype(int)
        y_in=y_in.flatten().astype(int)

        j=np.arange(-self.LANCZOS_TAB_SIZE, self.LANCZOS_TAB_SIZE)
        i=np.arange(-self.LANCZOS_TAB_SIZE, self.LANCZOS_TAB_SIZE)
        x_in=x_in[:,np.newaxis].repeat(j.shape[0],axis=-1)
        y_in=y_in[:,np.newaxis].repeat(j.shape[0],axis=-1)
        j=j[np.newaxis,:].repeat(x_in.shape[0],axis=0)
        i=i[np.newaxis,:].repeat(y_in.shape[0],axis=0)
        idx_x=x_in+i+1
        idx_y=y_in+j+1
        diff_x=x_in - idx_x
        diff_y=y_in - idx_y
        index=(idx_x>=0)*(idx_x<w)*(idx_y>=0)*(idx_y<h)
        diff_x[~index]=0
        diff_y[~index]=0
        coef_x = self.lanczos_coef(diff_x)
        coef_y = self.lanczos_coef(diff_y)
        coef = coef_x * coef_y*index
        idx_x[~index]=0
        idx_y[~index] = 0
        res=img[idx_y, idx_x,:]*coef[:,:,np.newaxis]
        res=res.sum(axis=1)
        sum_coef=coef.sum(axis=1,keepdims=True)

        return np.uint8(res / sum_coef)

    def IFilterNearest(self, img, x,y):
        h, w,_ = img.shape
        x = np.clip(x.flatten() - 0.5, 0.0, w - 1.0)
        y = np.clip(y.flatten() - 0.5, 0.0, h - 1.0)

        lx = np.round(x).astype(int)
        ly = np.round(y).astype(int)

        return np.uint8(img[ly, lx])

    def GenerateCPPMap(self, w, h):
        if self.m_cppMap is not None and self.m_cppMap.any():
            return

        self.m_cppMap = np.zeros((h, w), dtype=bool)

        j=np.arange(0,h).astype(float)
        i=np.arange(0,w).astype(float)
        j=j[:,np.newaxis].repeat(w, axis=1)
        i=i[np.newaxis,:].repeat(h, axis=0)
        phi = 3 * np.arcsin(0.5 - j / h)
        denom = (2 * np.cos(2 * phi / 3) - 1)
        # denom[denom==0]=1e-6
        lambda_ = (2 * np.pi * i / w - np.pi) / denom

        x = w * (lambda_ / (2 * np.pi) + 0.5)
        y = h * (0.5 - phi / np.pi)
        x[x<0]=x[x<0]-0.5
        y[y<0]=y[y<0]-0.5
        x[x>0]=x[x>0]+0.5
        y[y>0]=y[y>0]+0.5
        idx_x=x.astype(int)
        idx_y=y.astype(int)
        index=(idx_x>=0)*(idx_x<w)*(idx_y>=0)*(idx_y<h)
        self.m_cppMap[index]=True

    def cvt(self, erp:np.ndarray):
        rows, cols = self.m_cppMap.shape
        cpp = np.zeros((rows, cols,erp.shape[-1]), dtype=np.uint8)

        j = np.arange(0, rows).astype(float)
        i = np.arange(0, cols).astype(float)
        j = j[:, np.newaxis].repeat(cols, axis=1)
        i = i[np.newaxis, :].repeat(rows, axis=0)
        phi = 3 * np.arcsin(0.5 - j / rows)
        denom = (2 * np.cos(2 * phi / 3) - 1)
        lambda_ = (2 * np.pi * i / cols - np.pi) / denom

        x = erp.shape[1] * (lambda_ / (2 * np.pi) + 0.5) - 0.5
        y = erp.shape[0] * (0.5 - phi / np.pi) - 0.5
        cpp[self.m_cppMap] = self.interpolation(erp,x[self.m_cppMap], y[self.m_cppMap])
        return cpp

    def __call__(self, erp:np.ndarray,cpp_w=None,cpp_h=None):
        """
        :param erp: The input ERP image/Video, image dimensions should be (H,W,C)，video dimensions should be (T,H,W,C)
        :param cpp_w: Width of CPP view, the default is the same of ERP input
        :param cpp_h: Hight of CPP view, the default is the same of ERP input
        """
        #check input
        if erp.ndim==4:
            D,H,W,C = erp.shape
        elif erp.ndim==3:
            H,W,C = erp.shape
            D=None
        else:
            raise ValueError('ERP input must have 3 dimensions for image (H,W,C), or 4 dimebsions for video (T,H,W,C)')
        if not(C==1 or C==3 or C==4):
            raise ValueError('ERP input must have 3 dimensions for image (H,W,C), or 4 dimebsions for video (T,H,W,C)')

           # init CPP Mask Map
        if cpp_w is None or cpp_h is None:
            cpp_w=W
            cpp_h=H
        self.GenerateCPPMap(cpp_w, cpp_h)

        # convert
        if D is None:# for image
            return self.cvt(erp)
        else: # for video
            outputs=np.zeros((D,cpp_h,cpp_w,C), dtype=np.uint8)
            for i in range(D):
                outputs[i]=self.cvt(erp[i])
            return outputs



# Example
if __name__ == '__main__':

    import skimage.io
    img=skimage.io.imread('erp.png')
    erp2cpp=ERP2CPP('lanczos')
    cpp=erp2cpp(img)
    skimage.io.imsave('cpp.png', cpp)
    '''
    import skvideo.io
    video=skvideo.io.vread('erp.mp4')
    erp2cpp=ERP2CPP('lanczos')
    cpp=erp2cpp(video)
    skvideo.io.vwrite('cpp.mp4', cpp)
    '''

