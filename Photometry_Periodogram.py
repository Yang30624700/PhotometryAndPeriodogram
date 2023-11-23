import numpy as np
import matplotlib.pyplot as plt
from numpy import sum, pi, cos, sin, arctan2, exp, log, sqrt, dot, arange

from scipy.optimize import curve_fit
import scipy.optimize as optimize
import scipy.signal

from astropy.timeseries import LombScargle
from astropy.time import Time
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

from PyAstronomy.pyTiming import pyPDM
from multiprocessing import Pool
import sys
import os
from matplotlib.colors import hsv_to_rgb
import gc

from photutils.detection import IRAFStarFinder
from photutils.aperture import CircularAperture
from photutils.aperture import aperture_photometry

sys.modules['__main__'].__file__='ipython'

# MIT License

# Copyright (c) 2023 林浦 张靖毅 杨浩楠

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


class Photometry:
    '''
    输入fits文件
    输出light curve
    '''
    def __init__(self,light_path='',crop_path='Crop',result_path='Result',calibrated=0):
        '''
        功能:导入各种路径和已算出的数据,因此程序在重新打开后可以直接跳过已计算的部分

        
        light_path:未裁剪的亮场的路径,已有裁剪数据的情况下可以跳过

        crop_path:裁剪数据的路径

        result_path:各种计算结果的存储路径

        calibrated:亮场是否已经完成平、暗、偏置的校准
        '''
        if os.path.exists(light_path):
            self.light = light_path

        self.calibrated = calibrated
        self.load_calibration = 0

        if not os.path.exists(crop_path):
            os.mkdir(crop_path)
        self.crop = crop_path

        if not os.path.exists(result_path):
            os.mkdir(result_path)
        self.result = result_path

        #载入寻星数据
        if os.path.exists(self.result+'/Pos_r.txt'):
            self.Pos_r = np.loadtxt(self.result+'/Pos_r.txt',delimiter='\t')
        if os.path.exists(self.result+'/Pos_c.txt'):
            self.Pos_c = np.loadtxt(self.result+'/Pos_c.txt',delimiter='\t')
        if os.path.exists(self.result+'/Fwhm.txt'):
            self.Fwhm = np.loadtxt(self.result+'/Fwhm.txt',delimiter='\t')

        #载入测光数据
        if os.path.exists(self.result+'/T.npy'):
            self.T = np.load(self.result+'/T.npy')
        if os.path.exists(self.result+'/Mag.npy'):
            self.Mag = np.load(self.result+'/Mag.npy')
        if os.path.exists(self.result+'/Err.npy'):
            self.Err = np.load(self.result+'/Err.npy')

    def Crop(self,size,posn,posc,posr,search_radius=50,flip=False,use_low_std=False,flag_check=True,subcolumns=4):
        '''
        功能:将文件校准并以目标星为中心进行裁剪,裁剪后文件存入save_path中


        posn:此次处理的图片编号的范围,0为起始

        posc:第一张图中目标星的c坐标,相当于x坐标

        posr:第一张图中目标星的r坐标,相当于y坐标

        save_path:裁剪后文件保存路径

        size:裁剪尺寸的一半,裁剪后边缘到中心的距离,裁剪后图片尺寸为2size*2size

        search_radius:以大致位置为中心,目标星的寻找范围,
            如果过小可能在大致位置误差过大或星点FWHM过大时时无法找到目标星,
            如果过大可能在目标星附近存在更亮的其他星时寻找到错误的目标星.

        flip:该部分是否进行了中天翻转,进行了翻转会翻转输出的图像

        flag_check:在寻找到多个源或高阈值模式下无法找到源时是否绘制星点附近的图像以人工检验寻星是否准确

        sub_columns:绘图时分几列绘制
        '''

        n = 0
        plot_n = 0

        #检验pos输入是否正确
        if np.array(posn).shape[0] != 2 or (type(posr) != type(0) and type(posr) != type(0.)) or (type(posc) != type(0) and type(posc) != type(0.)):
            raise Exception('Wrong setting of pos')


        #主裁剪循环
        for name in os.listdir(self.light):
            #检测是否在处理范围内
            if n < posn[0] or n > posn[-1]:
                n += 1
                continue
            print('\rProcess: ',n-posn[0]+1,'/',posn[-1]-posn[0]+1,end='',flush=True)

            #获取亮场数组
            hdul = fits.open(self.light+'/'+name)
            if self.calibrated == 0:    #是否进行校准
                if  self.load_calibration == 1:
                    data = (hdul[0].data.astype('float32') - self.bias - self.dark)/self.flat
                else:
                    raise Exception('No data of calibrations, please use method Photometry.Load_calibration')
            else:
                data = hdul[0].data.astype('float32')
            hdul.close()
            gc.collect()

            #目标星位置寻找
            if n == posn[0]:
                c_obj0, r_obj0 = posc, posr
            else:
                c_obj0, r_obj0 = c_obj, r_obj   #使用上一张的精确位置作为这一张的初始位置
            radius = search_radius
            if use_low_std:
                find = IRAF_Find2(r_obj0,c_obj0,data,radius)
            else:
                find = IRAF_Find(r_obj0,c_obj0,data,radius)
            c_obj, r_obj = find[1:3]+[c_obj0-radius,r_obj0-radius]
            flag = find[0]

            #flag=1说明检测到了多个源,打印图像检测是否定位准确
            if flag != 0 and flag_check:
                plot_n  = np.mod(plot_n,subcolumns) + 1
                plot_detect_result(data,r_obj0,c_obj0,r_obj,c_obj,radius,sub=plot_n,subcolumns=subcolumns)
                plt.title(n)

            #由于要进行裁剪,因此对目标星的坐标四舍五入
            r_obj = np.rint(r_obj).astype('int')    
            c_obj = np.rint(c_obj).astype('int')
            data = data[r_obj-size:r_obj+size,c_obj-size:c_obj+size].astype('float32')
            #中天翻转
            if flip:
                data = data[::-1,::-1]

            #保存裁剪后的文件
            hdul = fits.PrimaryHDU(data.astype('float32'), header = hdul[0].header)
            hdul.writeto(self.crop+'/'+name,overwrite=True)

            n += 1
            
        print('Crop complete!')

    def Detect(self,Posc0,Posr0,search_radius=20,use_low_std=False,flag_check=True,subcolumns=4):
        '''
        功能:输出每张图中每颗星的位置和Fwhm,输入第一张图中的坐标,第n张图的搜索结果会作为第n+1张图的初始坐标


        Posc0:目标星和标准星在裁剪后的第一张图中的c坐标,可以由其他程序读出后导入,c坐标和x坐标等价

        Posr0:目标星和标准星在裁剪后的第一张图中的r坐标,可以由其他程序读出后导入,r坐标和y坐标等价

        search_radius:寻星范围

        search_radius:以大致位置为中心,目标星的寻找范围,
            如果过小可能在大致位置误差过大或星点FWHM过大时时无法找到目标星
            如果过大可能在目标星附近存在更亮的其他星时寻找到错误的目标星

        rotate_rate: 场旋速率,弧度/天,逆时针旋转为正,通过第一张图片的坐标和场旋大小可以计算出后续图片的坐标,需要人工比对头尾图片来计算场旋大小

        use_low_std:是否直接使用较低的寻星筛选阈值（默认是在高阈值下无法找到源才使用低阈值搜索）,在星场不密集时推荐开启

        flag_check:在寻找到多个源或高阈值模式下无法找到源时是否绘制星点附近的图像以人工检验寻星是否准确

        sub_columns:绘图时分几列绘制
        '''

        #裁剪中心位置确定,为目标星DAO后四舍五入
        n = 0
        plot_n = 0
        N = 0

        for name in os.listdir(self.crop):
            N += 1

        num_star = Posr0.shape[0]
        Pos0 = np.zeros([num_star,2]).astype('int')
        Pos0[:,0] = Posc0
        Pos0[:,1] = Posr0
        Pos_r = np.zeros([N,num_star])
        Pos_c = np.zeros([N,num_star])
        Fwhm = np.zeros([N,num_star])
        t = np.zeros(N)

        for name in os.listdir(self.crop):
            print('\rProcess: ',n+1,'/',N,end='',flush=True)

            hdul = fits.open(self.crop+'/'+name)
            data = hdul[0].data
            t[n] = Time(hdul[0].header['DATE-OBS']).mjd

            radius = search_radius
            for i in range(num_star):
                #输入初始位置
                c0, r0 = Pos0[i]
                
                if use_low_std:
                    find = IRAF_Find2(r0,c0,data,radius,5)
                else:
                    find = IRAF_Find(r0,c0,data,radius,5)
                fwhm = find[3]
                flag = find[0]
                c, r = find[1:3]+[c0-radius,r0-radius]
                Pos_r[n,i] = r
                Pos_c[n,i] = c

                
                if flag != 0 and flag_check:
                    plot_n  = np.mod(plot_n,subcolumns)+1
                    plot_detect_result(data,r0,c0,r,c,radius,sub=plot_n,subcolumns=subcolumns)
                    plt.title(n)
                Fwhm[n,i] = fwhm
            
            #这一张图的精确位置作为下一张图的初始位置
            Pos0[:,0] = Pos_c[n]
            Pos0[:,1] = Pos_r[n]
            n += 1

            hdul.close()
        gc.collect()
                
        self.Pos_r = Pos_r
        self.Pos_c = Pos_c
        self.Fwhm  = Fwhm
        np.savetxt(self.result+'/Pos_r.txt',Pos_r,delimiter='\t')
        np.savetxt(self.result+'/Pos_c.txt',Pos_c,delimiter='\t')
        np.savetxt(self.result+'/Fwhm.txt',Fwhm,delimiter='\t')

        print('Detect complete!')

    def Aperture_photometry(self,ref_mag,r1=2.,r2=3.,r3=5.,min_stars=10):
        '''
        功能:孔径测光,通过对比前后两张图片的测光结果来获取测光误差,图像中星点过少会导致误差不准确

        ref_mag:参考星的标准星等,在只关注光变曲线的变化而不关注平均星等时,该参数不重要,可以直接填写正确数量的0

        r1、r2、r3:孔径测光时的半径对于FWHM的倍数

        min_star:如果全图能被检测到的星点数量小于min_star,则跳过这张
        '''

        #读取位置和孔径信息信息
        Pos_c = self.Pos_c
        Pos_r = self.Pos_r
        fwhm = self.Fwhm
        Pos = np.concatenate((Pos_c[:,:,np.newaxis],Pos_r[:,:,np.newaxis]),axis = 2)
        flag_sourse = 0

        for i, name in enumerate(os.listdir(self.crop)):
            print('\rProcess: ',i+1,'/',Pos_r.shape[0],end='',flush=True)

            hdul = fits.open(self.crop+'/'+name)
            data = np.array(hdul[0].data)
            t = Time(hdul[0].header['DATE-OBS']).mjd
            hdul.close()
            gc.collect()

            Sourse = IRAF_Find_Full(data)
            if Sourse.shape[0] < 10:
                continue
            Sourse[:,2] = find_mag(Sourse[:,:2],data,np.mean(Sourse[:,2]))

            if flag_sourse == 0:
                #第一张跳过测光，第二张开始才能进行前后比对计算误差
                Sourse0 = Sourse.copy()
                flag_sourse = 1
                continue
            else:
                #交叉相关匹配
                Delta = []#储存平均星等和星等差
                for a in range(Sourse0.shape[0]):
                    for b in range(Sourse.shape[0]):
                        if np.abs(Sourse[b,0] - Sourse0[a,0]) < 2:
                            if np.abs(Sourse[b,1] - Sourse0[a,1]) < 2:
                                Delta.append([(Sourse[b,2]+Sourse0[a,2])/2,(Sourse[b,2]-Sourse0[a,2])])
                                if Sourse0[a,0] > data.shape[0]/2-2 and Sourse0[a,0] < data.shape[0]/2+2 and Sourse[b,1] > data.shape[0]/2-2 and Sourse[b,1] < data.shape[0]/2+2:
                                    mag_obj = (Sourse[b,2]+Sourse0[a,2])/2
                                    flux_obj = 10**(-(mag_obj-25)/2.5)
                Delta = np.array(Delta)
                Delta = Delta[np.argsort(Delta[:,0],axis=0),:]


                #初步计算星等-误差关系
                num = 10
                skip = int(Delta.shape[0]/5)

                Mag_dMag = []
                for star in range(0,Delta.shape[0]-num-skip):
                    Mag_dMag.append([np.mean(Delta[star:star+num,0]),np.std(Delta[star:star+num,1])])
                Mag_dMag = np.array(Mag_dMag)
                if Mag_dMag.shape[0] < min_stars:
                    continue

                Flux = 10**(-(Mag_dMag[:,0]-25)/2.5)
                dFlux = np.abs(np.log(10)*Mag_dMag[:,1]*Flux/-2.5)
                c = np.polyfit(Flux, dFlux, 1)
                dFlux_fit = Flux*c[0]+c[1]
                

                #剔除离散点
                num = 10
                Delta=Delta[np.argsort(Delta[:,0],axis=0),:]
                n = 0
                for star in range(0,Delta.shape[0]-num):
                    mean = np.mean(Delta[star:star+num,1]) 
                    mag = Delta[star,0]
                    flux = 10**(-(mag-25)/2.5)
                    dflux = flux*c[0]+c[1]
                    dmag = np.abs(-2.5*dflux/flux/np.log(10))
                    if abs(Delta[star,1]-mean)>3*dmag or (star < 5 and Delta[star,0] < mag_obj-2):
                        n += 1
                        Delta[star,0] = 25
                Delta = Delta[np.argsort(Delta[:,0],axis=0),:]
                if Delta.shape[0] < min_stars:
                    continue


                #重新计算星等-误差关系
                num = 10
                skip = int(Delta.shape[0]/5)+n

                Mag_dMag = []
                for star in range(0,Delta.shape[0]-num-skip):
                    Mag_dMag.append([np.mean(Delta[star:star+num,0]),np.std(Delta[star:star+num,1])])
                Mag_dMag = np.array(Mag_dMag)
                if Mag_dMag.shape[0] < min_stars:
                    continue

                Flux = 10**(-(Mag_dMag[:,0]-25)/2.5)
                dFlux = np.abs(np.log(10)*Mag_dMag[:,1]*Flux/-2.5)
                c = np.polyfit(Flux, dFlux, 1)
                dFlux_fit = Flux*c[0]+c[1]

                dflux_obj_liner = flux_obj*c[0]+c[1]
                err = np.abs(-2.5*dflux_obj_liner/10**(-(mag_obj-25)/2.5))/np.log(10)

                Sourse0 = Sourse.copy()
                
            loc = np.mean(fwhm[i,:])
            Mag = np.zeros(self.Pos_r.shape[1])

            for j in range(self.Pos_r.shape[1]):
                Mag[j] = np.array(find_mag(Pos[i,j,:],data,fwhm=loc,r1=r1,r2=r2,r3=r3))
        
            mag_obj = Mag[0] + np.mean(ref_mag) - np.mean(Mag[1:],axis=0)

            if i == 1:
                Mag_obj = np.array([mag_obj])
                Err = np.array([err])
                T = np.array([t])
            else:
                Mag_obj = np.append(Mag_obj,mag_obj)
                Err = np.append(Err,err)
                T = np.append(T,t)

                self.Mag = Mag_obj
                self.Err = Err
                self.T = T
                np.save(self.result+'/Mag.npy',self.Mag)
                np.save(self.result+'/Err.npy',self.Err)
                np.save(self.result+'/T.npy',self.T)

        print('Photometry complete!')

    def Load_calibration(self,flat,dark,bias,exptime=0,dark_exptime=1):
        '''
        载入校准场路径和亮场、暗场曝光时间

        flat,dark,bias:平场,暗场,偏置场的路径
        
        exptime:亮场曝光时间,用于校准,如果无需暗场校准则亮场曝光时间填写0

        dark_exptime:暗场曝光时间,用于校准
        '''

        hdul = fits.open(flat)
        Flat = hdul[0].data
        Flat_mean = np.mean(Flat)

        hdul = fits.open(dark)
        Dark = hdul[0].data*exptime/dark_exptime

        hdul = fits.open(bias)
        Bias = hdul[0].data

        hdul.close()

        self.flat = Flat/Flat_mean
        self.dark = Dark
        self.bias = Bias
        self.load_calibration = 1

def IRAF_Find(r0,c0,data,radius,fwhm=5.):
    '''
    IRAF寻星函数,输入图片、大致位置、搜索范围,输出精确位置
    '''
    r0 = int(r0)
    c0 = int(c0)
    mean, median, std = sigma_clipped_stats(data[r0-radius:r0+radius,c0-radius:c0+radius], sigma=3.0)  
    iraffind = IRAFStarFinder(fwhm=fwhm, threshold=5.*std)
    sources = iraffind(data[r0-radius:r0+radius,c0-radius:c0+radius] - median)

    flag = 0

    if sources is None:
        return IRAF_Find2(r0,c0,data,radius,fwhm=fwhm,Flag=1)

    max_flux = 0
    max_star = 0
    for i in range(len(sources['xcentroid'])):
        if sources['flux'][i] > max_flux:
            max_star = i
            max_flux = sources['flux'][i]
    if i > 0:
        flag = 1
    return np.transpose((flag,sources['xcentroid'][max_star],sources['ycentroid'][max_star],sources['fwhm'][max_star]))

def IRAF_Find2(r0,c0,data,radius,fwhm=5.,Flag=0):
    '''
    IRAF寻星函数,但是使用更宽泛的限制,用于IRAF_Find无法寻找到源时使用（通常由于拉线或虚焦导致）,或设定使用低寻星标准时使用（use_low_std=True）
    '''
    r0 = int(r0)
    c0 = int(c0)
    mean, median, std = sigma_clipped_stats(data[r0-radius:r0+radius,c0-radius:c0+radius], sigma=3.0)  
    iraffind = IRAFStarFinder(fwhm=fwhm, threshold=5.*std,sharplo=0,sharphi=2,roundlo=-2,roundhi=2)
    sources = iraffind(data[r0-radius:r0+radius,c0-radius:c0+radius] - median)

    if Flag == 1:
        flag = 1
    else:
        flag = 0

    if sources is None:
        return np.transpose((2,radius,radius,3))

    max_flux = 0
    max_star = 0
    for i in range(len(sources['xcentroid'])):
        if sources['flux'][i] > max_flux:
            max_star = i
            max_flux = sources['flux'][i]
    if i > 0:
        flag = 1
    return np.transpose((flag,sources['xcentroid'][max_star],sources['ycentroid'][max_star],sources['fwhm'][max_star]))

def IRAF_Find_Full(data,fwhm=5.):
    '''
    标记图片中的所有星点
    '''
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)  
    iraffind = IRAFStarFinder(fwhm=fwhm, threshold=5.*std,sharplo=0,sharphi=2,roundlo=-2,roundhi=2)
    sources = iraffind(data - median)

    return np.transpose((sources['xcentroid'],sources['ycentroid'],sources['fwhm']))

def plot_detect_result(data,r0,c0,r,c,radius,sub=0,subcolumns=4):
    '''
    绘图函数,用于检测寻星是否准确
    '''
    if sub == 0 or sub == 1:
        plt.figure(figsize=[subcolumns*5,5],dpi=100)
    if sub != 0:
        plt.subplot(1,subcolumns,sub)
    plt.imshow(data[r0-radius:r0+radius,c0-radius:c0+radius],'gray')
    apertures = CircularAperture(np.array([c,r])-[c0-radius,r0-radius], r=radius/5)
    apertures.plot(color='blue', lw=1, alpha=1);

def find_mag(positions,data,fwhm,r1=2.,r2=3.,r3=5.):
    aperture_m = CircularAperture(positions, r=r1*fwhm)
    aperture_bga = CircularAperture(positions, r=r2*fwhm)
    aperture_bgb = CircularAperture(positions, r=r3*fwhm)
    sum_m = aperture_photometry(data,aperture_m)['aperture_sum']
    sum_bg = (aperture_photometry(data,aperture_bgb)['aperture_sum']-aperture_photometry(data,aperture_bga)['aperture_sum'])
    flux = sum_m - sum_bg/((r3**2-r2**2)/r1**2)
    flux = np.abs(flux)
    return 25 - 2.5*np.log10(flux)


class Periodogram:
  '''
  输入：T,Mag,Err (若不输入数据，会自动识别上一步中产生的文件；若输入数据，则直接可以单独使用)
  输出：不同方法求得的周期、相位图和残差图；也可以一步对比五种方法的结果
  '''
  def __init__(self,T=0.,Mag=0.,Err=0.,image_path='Image'):
    '''
    输入观测得到的时间、星等、误差数组，如果不输入误差数组(与频谱分析算法中的权重有关)则默认误差为平均星等的10%
    '''

    if not os.path.exists(image_path):
        os.mkdir(image_path)
    self.iamge = image_path

    self.result='Result'

    if type(T) == type(0.) and os.path.exists(self.result+'/T.npy'):
        self.T = np.load(self.result+'/T.npy')
    else:
        self.T = T

    if type(Mag) == type(0.) and os.path.exists(self.result+'/Mag.npy'):
        self.Mag = np.load(self.result+'/Mag.npy')
    else:
        self.Mag = Mag
    
    if type(Err) == type(0.):
        if os.path.exists(self.result+'/Err.npy'):
            self.Err = np.load(self.result+'/Err.npy')
        else:
            self.Err = np.ones_like(self.Mag)*(np.mean(self.Mag)*0.1)
    else:
        self.Err = Err

  def Find_Peaks(self,Y):
    '''
    find peaks of power spectrum or entropy spectrum
    调用scipy.signal.find_peaks寻找谱的峰值
    '''
    peaks = scipy.signal.find_peaks(Y,prominence=0.1)
    return peaks
  
  def get_phase(self,time, period, shift=0.0):
    '''
    divide the time of observations by the period
    将时间序列变换为相位序列
    '''
    return (time / period - shift) % 1

  def rephase(self,data, period, shift=0.0, col=0, copy=True):
    '''
    transform the time of observations to the phase space
    将观测数据从时域转换为相位域
    '''
    rephased = np.ma.array(data, copy=copy)
    rephased[:, col] = self.get_phase(rephased[:, col], period, shift)
    return rephased

  def cond_entropy(self,period, data, p_bins=10, m_bins=5):
    '''
    Compute the conditional entropy for the normalized observations
    计算归一化后观测数据的条件熵
    '''
    if period <= 0:
      return np.PINF
    r = self.rephase(data, period)
    bins, *_ = np.histogram2d(r[:,0], r[:,1], [p_bins, m_bins],
                  [[0,1], [0,1]])
    size = r.shape[0]
    if size > 0:
      divided_bins = bins / size
      arg_positive = divided_bins > 0
      column_sums = np.sum(divided_bins, axis=1) #change 0 by 1
      column_sums = np.repeat(np.reshape(column_sums,(p_bins,1)), 
                  m_bins, axis=1)
      #column_sums = np.repeat(np.reshape(column_sums, (1,-1)),
      #						p_bins, axis=0)

      select_divided_bins = divided_bins[arg_positive]
      select_column_sums  = column_sums[arg_positive]

      A = np.empty((p_bins, m_bins), dtype=float)
      # bins[i,j]/size * log(bins[i,:] / size / (bins[i,j]/size))
      A[ arg_positive] = select_divided_bins \
                          * np.log(select_column_sums / select_divided_bins)
      A[~arg_positive] = 0

      return np.sum(A)
    else:
      return np.PINF

  def normalization(self,data):
    '''
    Normalize the magnitude of the star
    对观测星等归一化
    '''
    norm = np.ma.copy(data)
    norm[:,1] = (norm[:,1] - np.min(norm[:,1])) \
      / (np.max(norm[:,1]) - np.min(norm[:,1]))

    return norm
 
  def Sine(self,x,period,Amp,Phi0,Offset):
    return Amp*np.sin(np.pi*2/period*(x-Phi0)) + Offset 

  def Gaussion(self,x,Amp,Phi,Sigma,Shift):
    return Amp*np.e**(-(x-Phi)**2/(2*Sigma**2))+Shift

  def GaussionFitting(self,x,y,bounds):
    x_fit = np.linspace(np.min(x),np.max(x),100)
    p,err_fit = curve_fit(self.Gaussion,x,y,bounds=bounds)
    y_fit = self.Gaussion(x_fit,p[0],p[1],p[2],p[3])
    return x_fit,y_fit,p[0],p[1],p[2]

  def FittingRange(self,x,y):
    Peaks_high = self.Find_Peaks(y)#所有峰位置
    Peaks_low = self.Find_Peaks(-y)#所有谷位置
    x_Peaks_high = x[Peaks_high[0]]#峰坐标
    y_Peaks_high = y[Peaks_high[0]]#峰坐标
    x_Peaks_low = x[Peaks_low[0]]#谷坐标
    y_Peaks_low = y[Peaks_low[0]]#谷坐标
    Position_high = np.argsort(-y_Peaks_high)[:20]
    Point_high = [x_Peaks_high[Position_high],y_Peaks_high[Position_high]]
    x_Peaks_all = np.hstack((x_Peaks_high,x_Peaks_low))
    y_Peaks_all = np.hstack((y_Peaks_high,y_Peaks_low))[np.argsort(x_Peaks_all)]
    x_Peaks_all = x_Peaks_all[np.argsort(x_Peaks_all)]
    New_Position_high = []
    for i in range(len(Position_high)-1):
        Position = np.argwhere(x_Peaks_all>Point_high[0][i])[0] 
        New_Position_high.append(Position[0])
    Right = New_Position_high
    Left_ = [x-2 for x in Right]
    Left = [0 if v<0 else v for v in Left_]
    Left_Fre = x_Peaks_all[Left]
    Right_fre = x_Peaks_all[Right]
    return Left_Fre,Right_fre,Point_high[1]

  def Double(self,A):
    B = A
    return np.hstack((A,B))

  def DoublePhase(self,A):
    B = A + 1
    return np.hstack((A,B))

  def Phase_Residual(self,fold_period,Save=False):
    #fold and sort
    #对观测数据折叠转为相位序列
    Phase = np.mod(self.T,fold_period)/fold_period
    Mag_sort = self.Mag[np.argsort(Phase)]
    Err_sort = self.Err[np.argsort(Phase)]
    Phase_sort = np.mod(Phase[np.argsort(Phase)]+ 0.25 ,1)
    Mag_fold = self.Double(Mag_sort)
    Phase_fold = self.DoublePhase(Phase_sort) - 1
    Err_fold = self.Double(Err_sort)
    #Sine Fitting
    #进行Sin函数拟合相位曲线
    p,err_fit = curve_fit(self.Sine,Phase_fold,Mag_fold)
    Mag_fit = self.Sine(Phase_fold,p[0],p[1],p[2],p[3])
    print('Amp = %.5f , Period = %.5f , Phi0 = %.5f , Offset = %.5f '%(p[0],p[1],p[2],p[3]))
    #Plot
    fig, ax = plt.subplots(2,1,figsize = (14,12),sharex=True)
    ax1 = ax[0]
    ax2 = ax[1]
    ax1.errorbar(Phase_fold,Mag_fold,Err_fold,fmt='k.',capsize=2,elinewidth=0.3,ecolor='k',markersize=2)
    ax1.scatter(Phase_fold,Mag_fit,zorder=3,s=1,color='r')

    ax2.errorbar(Phase_fold,Mag_fold-Mag_fit,Err_fold,fmt='k.',capsize=2,elinewidth=0.3,ecolor='k',markersize=2)
    ax2.scatter(Phase_fold,np.zeros_like(Phase_fold),zorder=3,s=1,color='r')

    ax1.set_ylabel('Magnitude', fontsize=22)
    ax2.xaxis.set_tick_params(labelsize=22,direction='in', which='major')
    ax1.xaxis.set_tick_params(labelsize=22,direction='in', which='major')
    ax2.yaxis.set_tick_params(labelsize=22,direction='in', which='major')
    ax1.yaxis.set_tick_params(labelsize=22,direction='in', which='major')
    ax2.set_xlabel('Phase',fontsize=22)
    ax2.set_ylabel('Residual', fontsize=22)
    fig.subplots_adjust(hspace=0.05)
    if Save:
        plt.savefig('./Image/Residual.png',dpi=200)

  def LombScargleMethod(self,frequency_min,frequency_max,Num,Output_num=0,residual=False,PowerPlot=False,Title='',Savefig=False,Save_residual=False,Error_Fit=False):
    '''
    使用LombScargleMethod来分析光变曲线，得到频谱
    *输入搜寻频率的范围、频率点的个数
    *返回多个峰值的频率以及功率
    frequency_min：频率起始点
    frequency_max：频率截止点
    Num：频率数组的长度
    '''
    if Output_num:
        z = Output_num
    else:
        z = 5
    Frequency = np.linspace(frequency_min,frequency_max,Num)
    Power = LombScargle(self.T,self.Mag,self.Err).power(Frequency)
    Period = 1/Frequency
    Frequency_best = Frequency[np.argmax(Power)]
    Period_best = 1/Frequency_best
    
    Peaks = self.Find_Peaks(Power)

    Peaks_Frequency = Frequency[Peaks[0]]
    Peaks_Power = Power[Peaks[0]]
    Peaks_sort_Frequency = Peaks_Frequency[np.argsort(-Peaks_Power)]
    Peaks_sort_Power = Peaks_Power[np.argsort(-Peaks_Power)]
    if Error_Fit:
        A = self.FittingRange(Frequency,Power)
        Sigma = []
        Fre = []
        for i in range(z):
            Fit = self.GaussionFitting(Frequency[(Frequency>A[0][i])&(Frequency<A[1][i])],Power[(Frequency>A[0][i])&(Frequency<A[1][i])],[[0,A[0][i],0,0],[1,A[1][i],0.1,1]])
            Sigma.append(Fit[4])
            Fre.append(Fit[3])
        FWHM = [2*np.sqrt(2*np.log(2))*x for x in Sigma]

    if PowerPlot:
        fig, ax = plt.subplots(1,1,figsize = (14,6))
        ax.plot(Frequency, Power,lw=2,c='k') 
        ax.scatter(Peaks_sort_Frequency[:z],Peaks_sort_Power[:z],marker='+',c='r',s=50,zorder=3)
        ax.scatter(Peaks_sort_Frequency[0],Peaks_sort_Power[0],marker='+',c='b',s=50,zorder=3)
        ax.set_xlabel('Frequency [1/days]', fontsize = 22)
        ax.set_ylabel('Lomb-Scargle Power', fontsize = 22)
        ax.spines['top'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        plt.tick_params(labelsize = 22, width = 2, length = 6)

        if Title:
            plt.title(Title,fontsize = 22)
        else:
            plt.title('Lomb-Scargle Periodogram (LS)',fontsize = 22)
        if Savefig:
            plt.savefig('./Image/LS_Power.png',dpi=200)
    if residual:
      self.Phase_Residual(Period_best)
      if Save_residual:
          plt.savefig('./Image/LS_Residual.png',dpi=200)
    if Error_Fit:
        return [np.array([Frequency, Power]),np.array(Peaks_sort_Frequency[:z]),np.array(Peaks_sort_Power[:z]),np.array(Sigma)]
    else:
        return [np.array([Frequency, Power]),np.array(Peaks_sort_Frequency[:z]),np.array(Peaks_sort_Power[:z])]
        
  def ConditionEntropyMethod(self,frequency_min,frequency_max,Num,Output_num=0,residual=False,PowerPlot=False,Title='',Savefig=False,Save_residual=False,Error_Fit=False):
    '''
    条件熵方法，对光变曲线进行周期折叠，求不同周期下的熵
    *输入搜寻周期的范围、周期点的个数
    *返回多个峰值的周期以及功率
    period_min：周期起始点
    period_max：周期截止点
    Num：周期数组的长度
    '''
    if Output_num:
        z = Output_num
    else:
        z = 5
    T_Mag_Array = np.transpose(np.vstack((self.T,self.Mag)))
    Frequency = np.linspace(frequency_min,frequency_max,Num)
    Period = 1/Frequency
    Entropy = np.zeros(0)

    for i in range(len(Frequency)):
      Entropy = np.append(Entropy,self.cond_entropy(Period[i],self.normalization(T_Mag_Array)))

    Frequency_best = Frequency[np.argmin(Entropy)]
    Period_best = 1/Frequency_best
    Peaks = self.Find_Peaks(-Entropy)
    Peaks_Period = Period[Peaks[0]]
    Peaks_Entropy = Entropy[Peaks[0]]
    Peaks_Fre = Frequency[Peaks[0]]
    Peaks_sort_Entropy = Peaks_Entropy[np.argsort(Peaks_Entropy)]
    Peaks_sort_Fre = Peaks_Fre[np.argsort(Peaks_Entropy)]
    if Error_Fit:
        A = self.FittingRange(Frequency,-Entropy)
        Sigma = []
        Fre = []
        for i in range(z):
            # print(A[0][i],A[1][i])
            Fit = self.GaussionFitting(Frequency[(Frequency>A[0][i])&(Frequency<A[1][i])],Entropy[(Frequency>A[0][i])&(Frequency<A[1][i])],[[-2,A[0][i],0,0],[0,A[1][i],0.1,2]])
            Sigma.append(Fit[4])
            Fre.append(Fit[3])
        FWHM = [2*np.sqrt(2*np.log(2))*x for x in Sigma]

    if PowerPlot:
        fig, ax = plt.subplots(1,1,figsize = (14,6))
        ax.plot(Frequency, Entropy,lw=2,c='k') 
        ax.scatter(Peaks_sort_Fre[:z],Peaks_sort_Entropy[:z],marker='+',c='r',s=50,zorder=3)
        ax.scatter(Peaks_sort_Fre[0],Peaks_sort_Entropy[0],marker='+',c='b',s=50,zorder=3)
        ax.set_xlabel('Frequency [1/days]',fontsize=22)
        ax.set_ylabel('Conditional entropy', fontsize=22)
        ax.spines['top'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        plt.tick_params(labelsize = 22, width = 2, length = 6)
        if Title:
            plt.title(Title,fontsize = 22)
        else:
            plt.title('Conditional entropy (CE)',fontsize = 22)
        plt.tick_params(labelsize = 22, width = 2, length = 6)
        #在功率谱或者周期折叠谱上画出高斯拟合函数
        # fig, ax1 = plt.subplots(1,1,figsize = (14,6))
        # ax1.plot(Frequency, Entropy,lw=2,c='k')
        # ax1.vlines(A[0],0.8,1.2)
        # ax1.vlines(A[1],0.8,1.2)
        # ax1.plot(B[0],B[1])             
        if Savefig:
            plt.savefig('./Image/CE_Entropy.png',dpi=200)
    if residual:
        self.Phase_Residual(Period_best)
        if Save_residual:
            plt.savefig('./Image/CE_Residual.png',dpi=200)
    if Error_Fit:
        return [np.array([Frequency, Entropy]),np.array(Peaks_sort_Fre[:z]),np.array(Peaks_sort_Entropy[:z]),np.array(Sigma)]
    else:
        return [np.array([Frequency, Entropy]),np.array(Peaks_sort_Fre[:z]),np.array(Peaks_sort_Entropy[:z])]
    
  def GeneralisedLombScargleMethod(self,xlim=0,Output_num=0,residual=False,PowerPlot=False,Title='',Savefig=False,Save_residual=False,Error_Fit=False):
    '''
    GeneralisedLombScargleMethod
    自动生成频率范围，可以限制画图横轴范围
    xlim:频谱示意图横轴的范围(长度为2的数组)
    '''
    gls = Gls((self.T,self.Mag,self.Err))
    if Output_num:
        z = Output_num
    else:
        z = 5    
    Frequency = gls.f
    Power = gls.p
    Frequency_best = Frequency[np.argmax(Power)]
    Period_best = 1/Frequency_best

    Peaks = self.Find_Peaks(Power)
    Peaks_Frequency = Frequency[Peaks[0]]
    Peaks_Power = Power[Peaks[0]]
    Peaks_sort_Frequency = Peaks_Frequency[np.argsort(-Peaks_Power)]
    Peaks_sort_Power = Peaks_Power[np.argsort(-Peaks_Power)]
    if Error_Fit:
        A = self.FittingRange(Frequency,Power)
        Sigma = []
        Fre = []
        for i in range(z):
            Fit = self.GaussionFitting(Frequency[(Frequency>A[0][i])&(Frequency<A[1][i])],Power[(Frequency>A[0][i])&(Frequency<A[1][i])],[[0,A[0][i],0,0],[1,A[1][i],0.1,1]])
            Sigma.append(Fit[4])
            Fre.append(Fit[3])
        FWHM = [2*np.sqrt(2*np.log(2))*x for x in Sigma]

    if PowerPlot:
        fig, ax = plt.subplots(1,1,figsize = (14,6))
        ax.plot(Frequency,Power,lw=2,c='k',) 
        ax.scatter(Peaks_sort_Frequency[:z],Peaks_sort_Power[:z],marker='+',c='r',s=50,zorder=3)
        ax.scatter(Peaks_sort_Frequency[0],Peaks_sort_Power[0],marker='+',c='b',s=50,zorder=3)
        # ax.vlines(A[0],0,0.6)
        # ax.vlines(A[1],0,0.6)
        # ax.scatter(Frequency[(Frequency>A[0])&(Frequency<A[1])],Power[(Frequency>A[0])&(Frequency<A[1])],marker='+',c='b',s=50,zorder=3)
        # ax.plot(B[0],B[1],lw=2,c='r') 
        ax.set_xlabel('Frequency [1/days]', fontsize = 22)
        ax.set_ylabel('GLS Power', fontsize = 22)
        ax.spines['top'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)

        if xlim:
            ax.set_xlim(xlim)
        if Title:
            plt.title(Title,fontsize = 22)
        else:
            plt.title('Generalised Lomb-Scargle Periodogram (GLS)',fontsize = 22)
        plt.tick_params(labelsize = 22, width = 2, length = 6)
        if Savefig:
            plt.savefig('./Image/GLS_Power.png',dpi=200)
    if residual:
        self.Phase_Residual(Period_best)
        if Save_residual:
            plt.savefig('./Image/GLS_Residual.png',dpi=200)
    if Error_Fit:
        return [np.array([Frequency,Power]),np.array(Peaks_sort_Frequency[:z]),np.array(Peaks_sort_Power[:z]),np.array(Sigma)]
    else:
        return [np.array([Frequency,Power]),np.array(Peaks_sort_Frequency[:z]),np.array(Peaks_sort_Power[:z]),np.array(Sigma)]
      
  def PDMMethod(self,frequency_min,frequency_max,Num=0,Output_num=0,residual=False,PowerPlot=False,Title='',Savefig=False,Save_residual=False,Error_Fit=False):
    '''
    Phase Dispersion Minimization
    *输入搜寻周期的范围，可选择输入点的个数(默认点的间隔为1e-6)
    frequency_min:频率起始点(请谨慎选择频率起始点,尽量不包括预测周期的倍数或稍大于预测频率的一半，同时不影响高斯拟合时的边缘)
    frequency_max:频率截止点
    '''
    if Num:
      S = pyPDM.Scanner(minVal=frequency_min, maxVal=frequency_max, dVal=(frequency_max-frequency_min)/Num, mode="frequency")
    else:
      S = pyPDM.Scanner(minVal=frequency_min, maxVal=frequency_max, dVal=1e-6, mode="frequency")
    P = pyPDM.PyPDM(self.T,self.Mag)
    Frequency ,Power = P.pdmEquiBin(10, S)

    if Output_num:
        z = Output_num
    else:
        z = 5    
    Frequency_best = Frequency[np.argmin(Power)]
    Period_best = 1/Frequency_best
    Peaks = self.Find_Peaks(-Power)
    Peaks_Fre = Frequency[Peaks[0]]
    Peaks_Power = Power[Peaks[0]]
    Peaks_sort_Power = Peaks_Power[np.argsort(Peaks_Power)]
    Peaks_sort_Fre = Peaks_Fre[np.argsort(Peaks_Power)]
    if Error_Fit:
        A = self.FittingRange(Frequency,-Power)
        Sigma = []
        Fre = []
        for i in range(z):
            Fit = self.GaussionFitting(Frequency[(Frequency>A[0][i])&(Frequency<A[1][i])],Power[(Frequency>A[0][i])&(Frequency<A[1][i])],[[-2,A[0][i],0,0],[0,A[1][i],0.1,2]])
            Sigma.append(Fit[4])
            Fre.append(Fit[3])
        FWHM = [2*np.sqrt(2*np.log(2))*x for x in Sigma]


    if PowerPlot:
        fig, ax = plt.subplots(1,1,figsize = (14,6))
        ax.plot(Frequency ,Power,lw=2,c='k') 
        ax.scatter(Peaks_sort_Fre[:z],Peaks_sort_Power[:z],marker='+',c='r',s=50,zorder=3)
        ax.scatter(Peaks_sort_Fre[0],Peaks_sort_Power[0],marker='+',c='b',s=50,zorder=3)
        ax.set_xlabel('Frequency [1/days]', fontsize = 22)
        ax.set_ylabel('Phase Dispersion', fontsize = 22)
        ax.spines['top'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        plt.tick_params(labelsize = 22, width = 2, length = 6)
        if Title:
            plt.title(Title,fontsize = 22)
        else:
            plt.title('Phase Dispersion Minimization (PDM)',fontsize = 22)
        if Savefig:
            plt.savefig('./Image/PDM_power.png',dpi=200)
    if residual:
        self.Phase_Residual(Period_best)
        if Save_residual:
            plt.savefig('./Image/PDM_Residual.png',dpi=200)
    if Error_Fit:        
        return [np.array([Frequency ,Power]),np.array(Peaks_sort_Fre[:z]),np.array(Peaks_sort_Power[:z]),np.array(Sigma)]
    else:
        return [np.array([Frequency ,Power]),np.array(Peaks_sort_Fre[:z]),np.array(Peaks_sort_Power[:z])]

  def aovwloop(self,ifr, npar, ncov, nfr, ctw, frs, fr0, no, v, w, transit, v2):
      nbc = npar * ncov
      ncnt = np.zeros(nbc+ncov)
      ave = np.zeros(nbc+ncov)
      ind = np.zeros(no)
      ph = np.zeros(no)
      
      fr = ifr * frs + fr0
      for ip in range(0,2):
          for ibin in range(0,nbc):
              ave[ibin] = 0.
              ncnt[ibin] = 0.
  #             print(ncnt)
          if ip == 0:
              for i in range(0,no):
                  dph = self.T[i] * fr
                  dph-= np.floor(dph)
                  ph[i] = dph
                  ibin = int(dph * nbc)
                  ave[ibin] += v[i]
                  ncnt[ibin] += w[i]
          else:
  #             iflex += 1
  #                 sortx_m (no, ph, ind)
              ind = np.argsort(ph,kind = 'stable')
              for i in range(0,no):
                  ibin = int(i* nbc / no)
  #                     print(ind)
                  ave[ibin] += v[ind[i]]
                  ncnt[ibin] += w[ind[i]]

          for ibin in range(0,ncov):
              ncnt[ibin + nbc] = ncnt[ibin]
  #             print(ncnt)
          sav = 0.
          for ibin in np.arange(0,ncov + nbc)[::-1]:
              sav += ncnt[ibin]
              ncnt[ibin] = sav
  #                 print(sav)
  #             print('end')
          for ibin in range(0,nbc):
      #             print(ncnt[ibin],ncnt[ibin+ncov],ctw)
              ncnt[ibin] -= ncnt[ibin+ncov]
          for ibin in range(0,nbc):
              if ncnt[ibin] < ctw:
                  break
          if ibin >= nbc:
              break

      for ibin in range(0,ncov):
          ave[ibin + nbc] = ave[ibin]
      sav = 0.
      for ibin in np.arange(0,ncov + nbc)[::-1]:
          sav += ave[ibin]
          ave[ibin] = sav
      for ibin in range(0,nbc):
          ave[ibin] -= ave[ibin+ncov]
      if transit:
          sav = ave[0] / ncnt[0]
          for ibin in range(0,nbc):
              if (ave[ibin] / ncnt[ibin]) >= sav:
                  ave[ibin] /= ncnt[ibin]
                  sav = ave[ibin]
                  i = ibin
          sav *= sav * ncnt[i] * sw / (sw - ncnt[i])
      else:
          sav = 0.
          for ibin in range(0,nbc):
              sav += (ave[ibin] * ave[ibin] / ncnt[ibin])
          sav /= ncov
      spec = sav / v2
      return np.array([fr,spec])

  def aovwmp(self, npar, ncov, frequency_min, frequency_max, Num, Output_num=0,residual=False,PowerPlot=False,Title='',Savefig=False,Save_residual=False,Error_Fit=False):
    '''
    分箱权重方差分析法（多核版）。
    输入变量：
        t：观测时间数组
        vin：观测值数组
        win：观测权重数组
        npar：谐波阶数
        nfr：功率谱上考察的频点个数 
        frs：功率谱上考察的频点间隔
        fr0：功率谱上考察的起始频点
        npar: 主分箱数 
        ncov: 亚分箱数
    输出：
        spec：AOVW功率谱
    '''
    no = len(self.T)
    vin = 1/self.Err
    nfr = Num
    fr0 = frequency_min
    frs = (frequency_max - frequency_min)/Num
    spec = np.zeros((nfr,2))
    transit = npar < 0
    if transit:
        npar = -npar
      
    v = np.zeros(no)
    w = np.zeros(no)
    noeff = 0
    sw = 0
    ctw = 0
    for i in range(0, no):
        if vin[i]>0. : 
            w[i]=vin[i]
            sw+=w[i]
            ctw+=self.Mag[i]*w[i]
            noeff+=1
        else: w[i]=0.
    ctw/=sw
    v2=0.
    for i in range(0, no):
        v[i]=self.Mag[i]-ctw
        v2+=w[i]*v[i]*v[i]
        v[i]*=w[i]
    if ( (noeff <= npar + npar) or (noeff < 5 * npar) ):
        print("AOVw: error: wrong size of arrays/n")
        return (-1)
      
    iflex = 0
    ctw = 5 * sw / noeff
      
    l = np.arange(0,nfr)
    inputs = []
    for i in l:
        inputs.append((i, npar, ncov, nfr, ctw, frs, fr0, no, v, w, transit, v2))
    pool = Pool()
    spec = pool.starmap(self.aovwloop,inputs)
    pool.close()
    pool.join()
      
  #     if iflex > 0:
  #         print("AOVw:warning: poor phase coverage at ",iflex," frequencies\n")
      
    Spec =  np.array(spec)
    if Output_num:
        z = Output_num
    else:
        z = 5
    Power = Spec[:,1]
    Frequency = Spec[:,0]
    Frequency_best = Frequency[np.argmax(Power)]
    Period_best = 1/Frequency_best

    Peaks = self.Find_Peaks(Power)
    Peaks_Frequency = Frequency[Peaks[0]]
    Peaks_Power = Power[Peaks[0]]
    Peaks_sort_Frequency = Peaks_Frequency[np.argsort(-Peaks_Power)]
    Peaks_sort_Power = Peaks_Power[np.argsort(-Peaks_Power)]
    if Error_Fit:
        A = self.FittingRange(Frequency,Power)

        Sigma = []
        Fre = []

        for i in range(z):
            Fit = self.GaussionFitting(Frequency[(Frequency>A[0][i])&(Frequency<A[1][i])],Power[(Frequency>A[0][i])&(Frequency<A[1][i])],[[0,A[0][i],0,0],[1,A[1][i],0.1,1]])
            Sigma.append(Fit[4])
            Fre.append(Fit[3])
        FWHM = [2*np.sqrt(2*np.log(2))*x for x in Sigma]


    if PowerPlot: 
      fig, ax = plt.subplots(1,1,figsize = (14,6))
      ax.plot(Frequency,Power ,lw=2,c='k')
      ax.scatter(Peaks_sort_Frequency[:z],Peaks_sort_Power[:z],marker='+',c='r',s=50,zorder=3)
      ax.scatter(Peaks_sort_Frequency[0],Peaks_sort_Power[0],marker='+',c='b',s=50,zorder=3)
      ax.set_xlabel('Frequency [1/days]', fontsize = 22)
      ax.set_ylabel('Normalized AOV Statics', fontsize = 22)
      ax.spines['top'].set_linewidth(2)
      ax.spines['left'].set_linewidth(2)
      ax.spines['bottom'].set_linewidth(2)
      plt.tick_params(labelsize = 22, width = 2, length = 6)
      if Title:
        plt.title(Title,fontsize = 22)
      else:
        plt.title('Weighted Binned Analysis of Variance (AOVW)',fontsize = 22)
      if Savefig:
        plt.savefig('./Image/AOV_Power.png',dpi=200)
    if residual:
        self.Phase_Residual(Period_best)
        if Save_residual:
            plt.savefig('./Image/AOV_Residual.png',dpi=200)
    if Error_Fit:
        return [np.array([Frequency,Power]),np.array(Peaks_sort_Frequency[:z]),np.array(Peaks_sort_Power[:z]),np.array(Sigma)]
    else:
        return [np.array([Frequency,Power]),np.array(Peaks_sort_Frequency[:z]),np.array(Peaks_sort_Power[:z])]
   
  def Err_F2P(self,Err_F,Frequency):
      return abs(-1/(Frequency**2)*Err_F)

  def Normalization_Power(self,x,a,b):
    k = (b-a)/(np.max(x)-np.min(x))
    c = b - k*np.max(x)
    output = k*x + c
    return output
  
  def Normalization_Entropy(self,x,a,b):
    k = (a-b)/(np.max(x)-np.min(x))
    c = a - k*np.max(x)
    output = k*x + c
    return output

  def Normalization_Power_plot(self,Data,a,b,PowerPlot=False,PeriodPlot=False,Color_Plot=False,Save=False):
    Max_Period = np.max(Data[1][:,0])
    Min_Period = np.min(Data[1][:,0])
    Methods = [0,1,2,3,4]
    Ticklabels = ['LS','GLS','AOVW','CE','PDM']
    Result1 = np.copy(Data)
    for i in range(Result1[1][:,0].shape[0]-2):
      Nor =  self.Normalization_Power(np.array(Data[1][:,1][i]),a,b)
      Result1[1][:,1][i] = Nor
    for i in range(3,5):
      Nor =  self.Normalization_Entropy(np.array(Data[1][:,1][i]),a,b)
      Result1[1][:,1][i] = Nor
    if PowerPlot:
      fig, ax = plt.subplots(1,1,figsize = (10,6))
      for i in range(Result1[1][:,0].shape[0]):
        Power = np.array(Result1[1][:,1,i])
        plt.scatter(Methods,Power)
        ax.set_xlabel('Methods', fontsize = 22)
        ax.set_ylabel('Normalization Power', fontsize = 22)
        ax.set_xticks([0,1,2,3,4])
        ax.set_xticklabels(Ticklabels)
        ax.spines['top'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        plt.tick_params(labelsize = 22, width = 2, length = 6)
        if Save:
          plt.savefig('./Image/Power.png',dpi=200)
    if PeriodPlot:
      fig, ax = plt.subplots(1,1,figsize = (10,6))
      for i in range(Result1[1][:,0].shape[0]):
        Power = np.array(Result1[1][:,1,i])
        Period = np.array(Data[1][:,0,i])
        Err = np.array(Data[1][:,2,i])
        ax.errorbar(Methods,Period,Err,fmt='o',capsize=3,elinewidth=1.5,markersize=5,capthick=1.5)
        ax.scatter(Methods,Period,s=(Power)*700,alpha=0.3)
        ax.set_xlabel('Methods', fontsize = 22)
        ax.set_ylabel('Periods', fontsize = 22)
        ax.set_xticks([0,1,2,3,4])
        ax.set_xticklabels(Ticklabels)
        ax.spines['top'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        plt.tick_params(labelsize = 22, width = 2, length = 6)
        if Save:
          plt.savefig('./Image/Period.png',dpi=200)
    if Color_Plot:
      fig, ax = plt.subplots(1,1,figsize = (10,6))
      for i in range(Result1[1][:,0].shape[0]):
        Power = np.array(Result1[1][:,1,i])
        Period = np.array(Data[1][:,0,i])
        Err = np.array(Data[1][:,2,i])
        # ax.errorbar(Methods,Period,Err,ecolor=Period,cmap='rainbow',capsize=3,elinewidth=1.5,markersize=5,capthick=1.5)
        # ax.scatter(Methods,Period,s=(Power)*700,alpha=0.3,c=Period,cmap='jet')
        for j in range(5):
          color_hsv = hsv_to_rgb([(Period[j]-Min_Period+0.01)/(Max_Period-Min_Period+0.01),0.7,0.7])
          ax.errorbar(Methods[j],Period[j],Err[j],fmt='o',color=color_hsv,ecolor=color_hsv,capsize=3,elinewidth=1,markersize=2,capthick=1,) 
          ax.scatter(Methods[j],Period[j],s=(Power[j])*500,alpha=0.3,color=color_hsv)
        ax.set_xlabel('Methods', fontsize = 22)
        ax.set_ylabel('Periods', fontsize = 22)
        ax.set_xticks([0,1,2,3,4])
        ax.set_xticklabels(Ticklabels)
        ax.spines['top'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        plt.tick_params(labelsize = 22, width = 2, length = 6)
        if Save:
          plt.savefig('./Image/Period_color.png',dpi=200)
    return Result1

  def DifferentMethodsCompare(self,frequency_min,frequency_max,Num,Output_num=0,PowerPlot=False,Error_fit=False):
    '''
    frequency_min:频率起始点(请谨慎选择频率起始点,尽量不包括预测周期的倍数或稍大于预测频率的一半，同时不影响高斯拟合时的边缘)
    *主要原因是为了排除CE、PDM方法对周期倍数的敏感因素
    frequency_max:频率截止点
    Num:周期数组的长度;
    Output_num:输出的周期个数
    来对比五种方法得到的结果，返回五种方法得到的功率前五的周期值。
    ''' 
    if Output_num:
        Z = Output_num
    else:
        Z = 0
    if PowerPlot:
        LS = self.LombScargleMethod(frequency_min,frequency_max,Num,Z,PowerPlot=True,Error_fit=True)
        GLS = self.GeneralisedLombScargleMethod((frequency_min,frequency_max),Z,PowerPlot=True,Error_fit=True)
        AOV = self.aovwmp(5,3,frequency_min,frequency_max,Num,Z,PowerPlot=True,Error_fit=True)
        CE = self.ConditionEntropyMethod(frequency_min,frequency_max,Num,Z,PowerPlot=True,Error_fit=True)
        PDM = self.PDMMethod(frequency_min,frequency_max,Num,Z,PowerPlot=True,Error_fit=True)
    else:
        LS = self.LombScargleMethod(frequency_min,frequency_max,Num,Z,Error_fit=True)
        GLS = self.GeneralisedLombScargleMethod((frequency_min,frequency_max),Z,Error_fit=True)
        AOV = self.aovwmp(5,3,frequency_min,frequency_max,Num,Z,Error_fit=True)
        CE = self.ConditionEntropyMethod(frequency_min,frequency_max,Num,Z,Error_fit=True)
        PDM = self.PDMMethod(frequency_min,frequency_max,Num,Z,Error_fit=True)
    Frequency_All_Data = np.array((LS[1:4],GLS[1:4],AOV[1:4],CE[1:4],PDM[1:4]))
    Period_All_Data = np.zeros_like(Frequency_All_Data)
    Period_All_Data[:,0] = 1/Frequency_All_Data[:,0]
    Period_All_Data[:,1] = Frequency_All_Data[:,1]

    for i in range(Frequency_All_Data.shape[0]):
        Period_All_Data[i,2] = self.Err_F2P(Frequency_All_Data[i,2],Frequency_All_Data[i,0])
    # Methods = ['LS','GLS','AOV','CE','PDM']

    return Frequency_All_Data,Period_All_Data

	
class Gls:
    """
    Compute the Generalized Lomb-Scargle (GLS) periodogram.

    The *Gls* class computes the error-weighted Lomb-Scargle periodogram as
    developed by [ZK09]_ using various possible normalizations.

    The constructor of *Gls* takes a *TimeSeries* instance (i.e., a light curve)
    as first argument. The constructor allows to pass keywords to adjust the
    `freq` array, which will be used to calculate the periodogram.

    The main result of the calculation, i.e., the power, are stored in the
    class property `power`.

    Parameters
    ----------
    lc : tuple or list or str or TimeSeries object
        The light curve data either in the form of a TimeSeries object (or any
        object providing the attributes time, flux, and error) or a tuple or list
        or a filename as str providing time as first element, flux as second
        element, and optionally, the error as third element. 
    fbeg, fend : float, optional
        The beginning and end frequencies for the periodogram
        (inverse units of time axis).
    Pbeg, Pend : float, optional
        The beginning and end periods for the periodogram
        (same units as for time axis).
    ofac : int
        Oversampling factor of frequency grid (default=10).
    hifac : float
        Maximum frequency `freq` = `hifac` * (average Nyquist frequency)
        (default=1).
    freq : array, optional
        Contains the frequencies at which to calculate the periodogram.
        If given, fast and verbose option are not available.
        If not given, a frequency array will be automatically generated.
    norm : string, optional
        The normalization; either of "ZK", "Scargle", "HorneBaliunas", "Cumming", "wrms", "chisq".
        The default is unity ("ZK").
    ls : boolean, optional
        If True, the conventional Lomb-Scargle periodogram will be computed
        (default is False).
    fast : boolean, optional
        If True, recursive relations for trigonometric functions will be used
        leading to faster evaluation (default is False).
    verbose : boolean, optional
        Set True to obtain some statistical output (default is False).

    Attributes
    ----------
    power : array
        The normalized power of the GLS.
    freq : array
        The frequency array.
    ofac : int
        The oversampling factor of frequency grid.
    hifac : float
        The maximum frequency.
    t : array
        The abscissa data values.
    y : array
        The ordinate data values.
    e_y : array
        The errors of the data values.
    norm : string, {'ZK', 'Scargle', 'HorneBaliunas', 'Cumming', 'wrms', 'chisq'}
        The used normalization.

    Examples
    --------
    Create 1000 unevenly sampled data points with frequency=0.1,
    measurement error and Gaussian noise

    >>> time = np.random.uniform(54000., 56000., 1000)
    >>> flux = 0.15 * np.sin(2. * np.pi * time / 10.)

    Add some noise

    >>> error = 0.5 * np.ones_like(time)
    >>> flux += np.random.normal(0, error)

    Compute the full error-weighted Lomb-Periodogram
    in 'ZK' normalization and calculate the significance
    of the maximum peak.

    >>> gls = Gls((time, flux, error), verbose=True)

    >>> maxPower = gls.pmax
    >>> print("GLS maximum power: ", maxPower)
    >>> print("GLS statistics of maximum power peak: ", gls.stats(maxPower))
    >>> gls.plot(block=True)

    """
    # Available normalizations
    norms = ['ZK', 'Scargle', 'HorneBaliunas', 'Cumming', 'wrms', 'chisq', 'lnL', 'dlnL']
    def __init__(self, lc, fbeg=None, fend=None, Pbeg=None, Pend=None, ofac=10, hifac=1, freq=None, norm="ZK", ls=False, fast=False, verbose=False, **kwargs):

        self._freq = freq
        self.fbeg = fbeg
        self.fend = fend
        self.Pbeg = Pbeg
        self.Pend = Pend
        self.ofac = ofac
        self.hifac = hifac
        self.ls = ls
        self.norm = norm
        self.fast = fast
        self.label = {'title': 'Generalized Lomb Periodogram',
                      'xlabel': 'Frequency'}

        self._normcheck(norm)

        self._assignTimeSeries(lc)
        self._buildFreq()
        self._calcPeriodogram()
        self.pnorm(norm)
        self._peakPeriodogram()

        # Output statistics
        if verbose:
            self.info(fap=kwargs.get('fap', []))

    def _assignTimeSeries(self, lc):
        """
        A container class that holds the observed light curve.

        Parameters
        ----------
        time : array
            The time array.
        flux : array
            The observed flux/data.
        error : array, optional
            The error of the data values.

        """
        self.df = ''
        if isinstance(lc, str):
            # A data file has been given.
            try:
               self.df = lc
               lc = np.genfromtxt(lc, unpack=True)[0:3]
            except Exception as e:
               print("An error occurred while trying to read data file:")
               print("  ", e)

        if isinstance(lc, (tuple, list, np.ndarray)):
            # t, y[, e_y] were given as list or tuple.
            if len(lc) in (2, 3):
                self.t = np.ravel(lc[0])
                self.y = np.ravel(lc[1])
                self.e_y = None
                if len(lc) == 3 and lc[2] is not None:
                    # Error has been specified.
                    self.e_y = np.ravel(lc[2])
            else:
                raise(ValueError("lc is a list or tuple with " + str(len(lc)) + " elements. Needs to have 2 or 3 elements." + \
                                   " solution=Use 2 or 3 elements (t, y[, e_y]) or an instance of TimeSeries"))
        else:
            # Assume lc is an instance of TimeSeries.
            self.t, self.y, self.e_y = lc.time, lc.flux, lc.error

        self.th = self.t - self.t.min()
        self.tbase = self.th.max()
        self.N = len(self.y)

        # Re-check array length compatibility.
        if (len(self.th) != self.N) or ((self.e_y is not None) and (len(self.e_y) != self.N)):
            raise(ValueError("Incompatible dimensions of input data arrays (time and flux [and error]). Current shapes are: " + \
                             ', '.join(str(np.shape(x)) for x in (self.t, self.y, self.e_y))))

    def _buildFreq(self):
        """
        Build frequency array (`freq` attribute).

        Attributes
        ----------
        fnyq : float
            Half of the average sampling frequency of the time series.

        """
        self.fstep = 1 / self.tbase / self.ofac   # frequency sampling depends on the time span, default for start frequency
        self.fnyq = 0.5 / self.tbase * self.N     # Nyquist frequency
        self.f = self._freq

        if self.f is None:
            # Build frequency array if not present.
            if self.fbeg is None:
                self.fbeg = self.fstep if self.Pend is None else 1 / self.Pend
            if self.fend is None:
                self.fend = self.fnyq * self.hifac if self.Pbeg is None else 1 / self.Pbeg

            if self.fend <= self.fbeg:
                raise(ValueError("fend is smaller than (or equal to) fbeg but it must be larger. " + \
                               "Choose fbeg and fend so that fend > fbeg."))

            self.f = arange(self.fbeg, self.fend, self.fstep)
        elif self.fast:
            raise(ValueError("freq and fast cannot be used together."))

        self.freq = self.f   # alias name
        self.nf = len(self.f)

        # An ad-hoc estimate of the number of independent frequencies (Eq. (24) in ZK_09).
        self.M = (self.fend-self.fbeg) * self.tbase

    def _calcPeriodogram(self):

        if self.e_y is None:
            w = np.ones(self.N)
        else:
            w = 1 / (self.e_y * self.e_y)
        self.wsum = w.sum()
        w /= self.wsum

        self._Y = dot(w, self.y)       # Eq. (7)
        wy = self.y - self._Y          # Subtract weighted mean
        self._YY = dot(w, wy**2)       # Eq. (10), weighted variance with offset only
        wy *= w                        # attach errors

        C, S, YC, YS, CC, CS = np.zeros((6, self.nf))

        if self.fast:
            # Prepare trigonometric recurrences.
            eid = exp(2j * pi * self.fstep * self.th)  # cos(dx)+i sin(dx)

        for k, omega in enumerate(2.*pi*self.f):
            # Circular frequencies.
            if self.fast:
                if k % 1000 == 0:
                    # init/refresh recurrences to stop error propagation
                    eix = exp(1j * omega * self.th)  # exp(ix) = cos(x) + i*sin(x)
                cosx = eix.real
                sinx = eix.imag
            else:
                x = omega * self.th
                cosx = cos(x)
                sinx = sin(x)

            C[k] = dot(w, cosx)         # Eq. (8)
            S[k] = dot(w, sinx)         # Eq. (9)

            YC[k] = dot(wy, cosx)       # Eq. (11)
            YS[k] = dot(wy, sinx)       # Eq. (12)
            wcosx = w * cosx
            CC[k] = dot(wcosx, cosx)    # Eq. (13)
            CS[k] = dot(wcosx, sinx)    # Eq. (15)

            if self.fast:
               eix *= eid              # increase freq for next loop

        SS = 1. - CC
        if not self.ls:
            CC -= C * C            # Eq. (13)
            SS -= S * S            # Eq. (14)
            CS -= C * S            # Eq. (15)
        D = CC*SS - CS*CS          # Eq. (6)

        self._a = (YC*SS-YS*CS) / D
        self._b = (YS*CC-YC*CS) / D
        self._off = -self._a*C - self._b*S

        # power
        self.p = (SS*YC*YC + CC*YS*YS - 2.*CS*YC*YS) / (self._YY*D)   # Eq. (5) in ZK09

    def _normcheck(self, norm):
        """
        Check normalization

        Parameters
        ----------
        norm : string
            Normalization string

        """
        if norm not in self.norms:
            raise(ValueError("Unknown norm: %s. " % norm + \
                "Use either of %s." % ', '.join(self.norms)))

    def pnorm(self, norm="ZK"):
        """
        Assign or modify normalization (can be done afterwards).

        Parameters
        ----------
        norm : string, optional
            The normalization to be used (default is 'ZK').

        Examples
        --------
        >>> gls.pnorm('wrms')

        """
        self._normcheck(norm)
        self.norm = norm
        p = self.p
        power = p   # default ZK
        self.label["ylabel"] = "Power ("+norm+")"

        if norm == "Scargle":
            popvar = input('pyTiming::gls - Input a priori known population variance:')
            power = p / float(popvar)
        elif norm == "HorneBaliunas":
            power = (self.N-1)/2. * p
        elif norm == "Cumming":
            power = (self.N-3)/2. * p / (1.-self.p.max())
        elif norm == "chisq":
            power = self._YY *self.wsum * (1.-p)
            self.label["ylabel"] = "$\chi^2$"
        elif norm == "wrms":
            power = sqrt(self._YY*(1.-p))
            self.label["ylabel"] = "wrms"
        elif norm == "lnL":
            chi2 = self._YY *self.wsum * (1.-p)
            power = -0.5*chi2 - 0.5*np.sum(np.log(2*np.pi * self.e_y**2))
            self.label["ylabel"] = "$\ln L$"
        elif norm == "dlnL":
            # dlnL = lnL - lnL0 = -0.5 chi^2 + 0.5 chi0^2 = 0.5 (chi0^2 - chi^2) = 0.5 chi0^2 p
            power = 0.5 * self._YY * self.wsum * p
            self.label["ylabel"] = "$\Delta\ln L$"

        self.power = power

    def _peakPeriodogram(self):
        """
        Analyze the highest periodogram peak.
        """
        # Index of maximum power
        k = self.p.argmax()
        # Maximum power
        self.pmax = pmax = self.p[k]
        self.rms = rms = sqrt(self._YY*(1.-pmax))

        # Statistics of highest peak
        self.hpstat = self.best = p = {}   # alias name for best and hpstat

        # Best parameters
        p["f"] = fbest = self.f[k]
        p["P"] = 1. / fbest
        p["amp"] = amp = sqrt(self._a[k]**2 + self._b[k]**2)
        p["ph"] = ph = arctan2(self._a[k], self._b[k]) / (2.*pi)
        p["T0"]  = self.t.min() - ph/fbest
        p["offset"] = self._off[k] + self._Y            # Re-add the mean.

        # Error estimates
        p["e_amp"] = sqrt(2./self.N) * rms
        p["e_ph"] = e_ph = sqrt(2./self.N) * rms/amp/(2.*pi)
        p["e_T0"] = e_ph / fbest
        p["e_offset"] = sqrt(1./self.N) * rms

        # Get the curvature in the power peak by fitting a parabola y=aa*x^2
        if 1 < k < self.nf-2:
            # Shift the parabola origin to power peak
            xh = (self.f[k-1:k+2] - self.f[k])**2
            yh = self.p[k-1:k+2] - pmax
            # Calculate the curvature (final equation from least square)
            aa = dot(yh, xh) / dot(xh, xh)
            p["e_f"] = e_f = sqrt(-2./self.N / aa * (1.-self.pmax))
            p["e_P"] = e_f / fbest**2
        else:
            p["e_f"] = np.nan
            p["e_P"] = np.nan
            print("WARNING: Highest peak is at the edge of the frequency range.\n"\
                  "No output of frequency error.\n"\
                  "Increase frequency range to sample the peak maximum.")

    def sinmod(self, t=None):
        """
        Calculate best-fit sine curve.

        The parameters of the best-fit sine curve can be accessed via
        the dictionary attribute `best`. Specifically, "amp" holds the
        amplitude, "fbest" the best-fit frequency, "T0" the reference time
        (i.e., time of zero phase), and "offset" holds the additive offset
        of the sine wave. 

        Parameters
        ----------
        t : array
            Time array at which to calculate the sine.
            If None, the time of the data are used.

        Returns
        -------
        Sine curve : array
            The best-fit sine curve (i.e., that for which the
            power is maximal).
        """
        if t is None:
            t = self.t

        try:
            p = self.best
            return p["amp"] * sin(2*np.pi*p["f"]*(t-p["T0"])) + p["offset"]
        except Exception as e:
            print("Failed to calcuate best-fit sine curve.")
            raise(e)

    def info(self, stdout=True, fap=[]):
        """
        Prints some basic statistical output screen.
        """
        lines = ("Generalized LS - statistical output",
           "-----------------------------------",
           "Number of input points:     %6d" % self.N,
           "Weighted mean of dataset:   %f"  % self._Y,
           "Weighted rms of dataset:    %f"  % sqrt(self._YY),
           "Time base:                  %f"  % self.tbase,
           "Number of frequency points: %6d" % self.nf,
           "",
           "Maximum power p [%s]: %f" % (self.norm, self.power.max()),
           "FAP(pmax):            %s" % self.FAP(),
           "RMS of residuals:     %f" % self.rms)
        if self.e_y is not None:
           lines += "  Mean weighted internal error:  %f" % (sqrt(self.N/sum(1./self.e_y**2))),
        lines += (
           "Best sine frequency:  {f:f} +/- {e_f:f}",
           "Best sine period:     {P:f} +/- {e_P:f}",
           "Amplitude:            {amp:f} +/- {e_amp:f}",
           #"Phase (ph):          {ph:f} +/- {e_ph:f}",
           #"Phase (T0):          {T0:f} +/- {e_T0:f}",
           #"Offset:              {offset:f} +/- {e_offset:f}",
           "-----------------------------------")
        for fapi in fap:
            lines += 'p(FAP=%s): %s' % (fapi, self.powerLevel(fapi)),
        text = "\n".join(lines).format(**self.best)
        if stdout:
           print(text)
        else:
           return text

    def plot(self, block=False, period=False, fap=None, gls=True, data=True, residuals=True):
        """
        Create a plot.

        Parameters
        ----------
        period : boolean
            The periodogram is plotted against log(Period).
        fap : float, list
            Plots the FAP levels.
        gls : boolean
            Plots the GLS periodogram.
        data : boolean
            Plots the data.
        residuals : boolean
            Plots the residuals.

        Returns
        -------
        fig : mpl.figure
            A figure which can be modified.
        """
        try:
            import matplotlib
            import matplotlib.pylab as mpl
        except ImportError:
            raise(ImportError("Could not import matplotlib.pylab."))

        fbest, T0 = self.best["f"], self.best["T0"]

        fig = mpl.figure()
        fig.canvas.manager.set_window_title('GLS periodogram')
        fig.subplots_adjust(hspace=0.05, wspace=0.04, right=0.97, bottom=0.09, top=0.84)
        fs = 10   # fontsize

        nrow = gls + data + residuals
        plt, plt1, plt2, plt3, plt4 = [None] * 5

        if gls:
           # Periodogram
           plt = fig.add_subplot(nrow, 1, 1)
           plt.tick_params(direction='in')
           if period:
              plt.set_xscale("log")
              plt.set_xlabel("Period P")
           else:
              plt.set_xlabel("Frequency $f$")

           plt.set_ylabel(self.label["ylabel"])
           plt.plot(1/self.f if period else self.f, self.power, 'b-', linewidth=.5)
           # mark the highest peak
           plt.plot(1/fbest if period else fbest, self.power[self.p.argmax()], 'r.', label="$1/f = %f$" % (1/fbest))

           x2tics = 1 / np.array([0.5, 1, 2, 3, 5, 10, 20., 100])
           mx2tics = 1 / np.array([0.75, 1.5, 2.5, 4, 15, 40, 60., 80, 100])
           def tick_function(X):
              return ["%g" % (1/z) for z in X]

           plt.tick_params(direction='in', which='both', top=True, right=True)
           plt.minorticks_on()
           plt.autoscale(enable=True, axis='x', tight=True)
           if not period:
              ax2 = plt.twiny()
              ax2.tick_params(direction='in', which='both')
              ax2.format_coord = lambda x,y: "x=%g, x2=%g, y=%g"% (x, 1/x, y)
              ax2.set_xticks(x2tics)
              ax2.set_xticks(mx2tics, minor=True)
              ax2.set_xticklabels(tick_function(x2tics))
              ax2.set_xlim(plt.get_xlim())
              ax2.set_xlabel("Period")
              plt.tick_params(top=False)

           if fap is not None:
              if isinstance(fap, float):
                 fap = [fap]
              n = max(1, len(fap)-1)   # number of dash types
              for i,fapi in enumerate(fap):
                 plt.axhline(self.powerLevel(fapi), linewidth=0.5, color='r', dashes=(8+32*(n-i)/n,8+32*i/n), label="FAP = %s%%"%(fapi*100))
           plt.legend(numpoints=1, fontsize=fs, frameon=False)

        # Data and model
        col = mpl.cm.rainbow(mpl.Normalize()(self.t))
        def plot_ecol(plt, x, y):
           # script for scatter plot with errorbars and time color-coded
           datstyle = dict(color=col, marker='.', edgecolor='k', linewidth=0.5, zorder=2)
           if self.e_y is not None:
              errstyle = dict(yerr=self.e_y, marker='', ls='', elinewidth=0.5)
              if matplotlib.__version__ < '2.' :
                 errstyle['capsize'] = 0.
                 datstyle['s'] = 8**2   # requires square size !?
              else:
                 errstyle['ecolor'] = col
              _, _, (c,) = plt.errorbar(x, y, **errstyle)
              if matplotlib.__version__ < '2.':
                 c.set_color(col)
           plt.scatter(x, y, **datstyle)

        def phase(t):
           #return (t-T0)*fbest % 1
           return (t-T0) % (1/fbest)

        if data:
           # Time series
           tt = arange(self.t.min(), self.t.max(), 0.01/fbest)
           ymod = self.sinmod(tt)
           plt1 = fig.add_subplot(nrow, 2, 2*gls+1)
           plt1.set_ylabel("Data")
           if residuals:
              mpl.setp(plt1.get_xticklabels(), visible=False)
           else:
              plt1.set_xlabel("Time")
           plot_ecol(plt1, self.t, self.y)
           plt1.plot(tt, ymod, 'k-', zorder=0)

           # Phase folded data
           tt = arange(T0, T0+1/fbest, 0.01/fbest)
           yy = self.sinmod(tt)
           plt2 = fig.add_subplot(nrow, 2, 2*gls+2, sharey=plt1)
           mpl.setp(plt2.get_yticklabels(), visible=False)
           if residuals:
              mpl.setp(plt2.get_xticklabels(), visible=False)
           else:
              plt2.set_xlabel("Phase")
           plot_ecol(plt2, phase(self.t), self.y)
           xx = phase(tt)
           ii = np.argsort(xx)
           plt2.plot(xx[ii], yy[ii], 'k-')
           plt2.format_coord = lambda x,y: "x=%g, x2=%g, y=%g"% (x, x*fbest, y)

        if residuals:
           # Time serie of residuals
           yfit = self.sinmod()
           yres = self.y - yfit
           plt3 = fig.add_subplot(nrow, 2, 2*(gls+data)+1, sharex=plt1)
           plt3.set_xlabel("Time")
           plt3.set_ylabel("Residuals")
           plot_ecol(plt3, self.t, yres)
           plt3.plot([self.t.min(), self.t.max()], [0,0], 'k-')

           # Phase folded residuals
           plt4 = fig.add_subplot(nrow, 2, 2*(gls+data)+2, sharex=plt2, sharey=plt3)
           plt4.set_xlabel("Phase")
           mpl.setp(plt4.get_yticklabels(), visible=False)
           plot_ecol(plt4, phase(self.t), yres)
           plt4.plot([0,1/fbest], [0,0], 'k-')
           plt4.format_coord = lambda x,y: "x=%g, x2=%g, y=%g"% (x, x*fbest, y)

        for x in fig.get_axes()[2:]:
           x.tick_params(direction='in', which='both', top=True, right=True)
           x.minorticks_on()
           x.autoscale(enable=True, tight=True)

        if hasattr(mpl.get_current_fig_manager(), 'toolbar'):
            # check seems not needed when "TkAgg" is set
            try:
                mpl.get_current_fig_manager().toolbar.pan()
            except:
                pass # e.g. Jupyter
        #t = fig.canvas.toolbar
        #mpl.ToggleTool(mpl.wx_ids['Pan'], False)

        fig.tight_layout()   # to get the left margin
        marleft = fig.subplotpars.left * fig.get_figwidth() * fig.dpi / fs
        def tighter():
           # keep margin tight when resizing
           xdpi = fs / (fig.get_figwidth() * fig.dpi)
           ydpi = fs / (fig.get_figheight() * fig.dpi)
           fig.subplots_adjust(bottom=4.*ydpi, top=1-ydpi-4*gls*ydpi, right=1-1*xdpi, wspace=4*xdpi, hspace=4*ydpi, left=marleft*xdpi)
           if gls and (residuals or data):
              # gls plot needs additional space for x2axis
              fig.subplots_adjust(top=1-8*ydpi)
              if matplotlib.__version__ < '2.':
                 ax2.set_position(plt.get_position().translated(0,4*ydpi))
              plt.set_position(plt.get_position().translated(0,4*ydpi))

        #fig.canvas.mpl_connect("resize_event", lambda _: (fig.tight_layout()))
        fig.canvas.mpl_connect("resize_event", lambda _: (tighter()))
        fig.show()
        if block:
           print("Close the plot to continue.")
           # needed when called from shell
           mpl.show()
        else:
           # avoids blocking when: import test_gls
           mpl.ion()
        # mpl.show(block=block) # unexpected keyword argument 'block' in older matplotlib
        return fig

    def prob(self, Pn):
        """
        Probability of obtaining the given power.

        Calculate the probability to obtain a power higher than
        `Pn` from the noise, which is assumed to be Gaussian.

        .. note:: Normalization
          (see [ZK09]_ for further details).

          - `Scargle`:
          .. math::
            exp(-Pn)

          - `HorneBaliunas`:
          .. math::
            \\left(1 - 2 \\times \\frac{Pn}{N-1} \\right)^{(N-3)/2}

          - `Cumming`:
          .. math::
            \\left(1+2\\times \\frac{Pn}{N-3}\\right)^{-(N-3)/2}

        Parameters
        ----------
        Pn : float
            Power threshold.

        Returns
        -------
        Probability : float
            The probability to obtain a power equal or
            higher than the threshold from the noise.

        """
        self._normcheck(self.norm)
        if self.norm == "Scargle": return exp(-Pn)
        if self.norm == "HorneBaliunas": return (1-2*Pn/(self.N-1)) ** ((self.N-3)/2)
        if self.norm == "Cumming": return (1+2*Pn/(self.N-3)) ** (-(self.N-3)/2)
        if self.norm == "wrms": return (Pn**2/self._YY) ** ((self.N-3)/2)
        if self.norm == "chisq": return (Pn/self._YY/self.wsum) ** ((self.N-3)/2)
        if self.norm == "ZK":
            p = Pn
        if self.norm == "dlnL":
            p = 2 * Pn / self._YY / self.wsum
        if self.norm == "lnL":
            chi2 = -2*Pn - np.sum(np.log(2*np.pi * self.e_y**2))
            p = 1 - chi2/self._YY/self.wsum
        return (1-p) ** ((self.N-3)/2)

    def probInv(self, Prob):
        """
        Calculate minimum power for given probability.

        This function is the inverse of `Prob(Pn)`.
        Returns the minimum power for a given probability threshold `Prob`.

        Parameters
        ----------
        Prob : float
            Probability threshold.

        Returns
        -------
        Power threshold : float
            The minimum power for the given false-alarm probability threshold.

        """
        self._normcheck(self.norm)
        if self.norm == "Scargle": return -log(Prob)
        if self.norm == "HorneBaliunas": return (self.N-1) / 2 * (1-Prob**(2/(self.N-3)))
        if self.norm == "Cumming": return (self.N-3) / 2 * (Prob**(-2./(self.N-3))-1)
        if self.norm == "wrms": return sqrt(self._YY * Prob**(2/(self.N-3)))
        if self.norm == "chisq": return self._YY * self.wsum * Prob**(2/(self.N-3))
        p = 1 - Prob**(2/(self.N-3))
        if self.norm == "ZK": return p
        if self.norm == "lnL": return -0.5*self._YY*self.wsum*(1.-p) - 0.5*np.sum(np.log(2*np.pi * self.e_y**2))
        if self.norm == "dlnL": return 0.5 * self._YY * self.wsum * p

    def FAP(self, Pn=None):
        """
        Obtain the false-alarm probability (FAP).

        The FAP denotes the probability that at least one out of M independent
        power values in a prescribed search band of a power spectrum computed
        from a white-noise time series is as large as or larger than the
        threshold, `Pn`. It is assessed through

        .. math:: FAP(Pn) = 1 - (1-Prob(P>Pn))^M \\; ,

        where "Prob(P>Pn)" depends on the type of periodogram and normalization
        and is calculated by using the *prob* method; *M* is the number of
        independent power values and is computed internally.

        Parameters
        ----------
        Pn : float
            Power threshold. If None, the highest periodogram peak is used.

        Returns
        -------
        FAP : float
            False alarm probability.

        """
        if Pn is None:
           Pn = self.pmax
        prob = self.M * self.prob(Pn)
        if prob > 0.01:
           return 1 - (1-self.prob(Pn))**self.M
        return prob

    def powerLevel(self, FAPlevel):
        """
        Power threshold for FAP level.

        Parameters
        ----------
        FAPlevel : float or array_like
              "False Alarm Probability" threshold

        Returns
        -------
        Threshold : float or array
            The power threshold pertaining to a specified false-alarm
            probability (FAP). Powers exceeding this threshold have FAPs
            smaller than FAPlevel.

        """
        Prob = 1. - (1.-FAPlevel)**(1./self.M)
        return self.probInv(Prob)

    def stats(self, Pn):
        """
        Obtain basic statistics for power threshold.

        Parameters
        ----------
        Pn : float
            Power threshold.

        Returns
        -------
        Statistics : dictionary
            A dictionary containing {'Pn': *Pn*, 'Prob': *Prob(Pn)* ,
            'FAP': *FAP(Pn)*} for the specified power threshold, *Pn*.

        """
        return {'Pn': Pn, 'Prob': self.prob(Pn), 'FAP': self.FAP(Pn)}

    def toFile(self, ofile, header=True):
        """
        Write periodogram to file.


        The output file is a standard text file with two columns,
        viz., frequency and power.

        Parameters
        ----------
        ofile : string
            Name of the output file.

        """
        with open(ofile, 'w') as f:
            if header:
               f.write("# Generalized Lomb-Scargle periodogram\n")
               f.write("# Parameters:\n")
               f.write("#    Data file: %s\n" % self.df)
               f.write("#    ofac     : %s\n" % self.ofac)
               f.write("#    norm     : %s\n" % self.norm)
               f.write("# 1) Frequency, 2) Normalized power\n")
            for line in zip(self.f, self.power):
               f.write("%f  %f\n" % line)

        print("Results have been written to file: ", ofile)



