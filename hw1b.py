
# coding: utf-8

# In[1]:

import os
from os import walk

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy.linalg as linalg

from PIL import Image

import theano
import theano.tensor as T


# In[2]:

def reconstructed_image(D,c,num_coeffs,X_mean,im_num):
    '''
    This function reconstructs an image given the number of
    coefficients for each image specified by num_coeffs
    '''
    
    '''
        Parameters
    ---------------
    c: np.ndarray
        a n x m matrix  representing the coefficients of all the image blocks.
        n represents the maximum dimension of the PCA space.
        m is (number of images x n_blocks**2)

    D: np.ndarray
        an N x n matrix representing the basis vectors of the PCA space
        N is the dimension of the original space (number of pixels in a block)

    im_num: Integer
        index of the image to visualize

    X_mean: np.ndarray
        a matrix representing the mean block.

    num_coeffs: Integer
        an integer that specifies the number of top components to be
        considered while reconstructing
    '''
    
    c_im = c[:num_coeffs,im_num]
    D_im = D[:,:num_coeffs]
    
#     X_recon_img=np.zeros([256,256])
    temp=np.dot(D_im,c_im).T 
    temp=temp+X_mean.flatten() 
    return temp.reshape(256,256)


# In[3]:

def plot_reconstructions(D,c,num_coeff_array,X_mean,im_num):
    '''
    Plots 9 reconstructions of a particular image using D as the basis matrix and coeffiecient
    vectors from c

    Parameters
    ------------------------
        num_coeff_array: Iterable
            an iterable with 9 elements representing the number_of coefficients
            to use for reconstruction for each of the 9 plots
        
        c: np.ndarray
            a l x m matrix  representing the coefficients of all blocks in a particular image
            l represents the dimension of the PCA space used for reconstruction
            m represents the number of blocks in an image

        D: np.ndarray
            an N x l matrix representing l basis vectors of the PCA space
            N is the dimension of the original space (number of pixels in a block)

        X_mean: basis vectors represent the divergence from the mean so this
            matrix should be added to all reconstructed blocks

        im_num: Integer
            index of the image to visualize
    '''
    f, axarr = plt.subplots(3,3)
    for i in range(3):
        for j in range(3):
            plt.axes(axarr[i,j])
            plt.imshow(reconstructed_image(D,c,num_coeff_array[i*3+j],X_mean,im_num))
            
    f.savefig('output/hw1b_{0}.png'.format(im_num))
    plt.close(f)


# In[4]:

def plot_top_16(D, sz, imname):
    '''
    Plots the top 16 components from the basis matrix D.
    Each basis vector represents an image block of shape (sz, sz)

    Parameters
    -------------
    D: np.ndarray
        N x n matrix representing the basis vectors of the PCA space
        N is the dimension of the original space (number of pixels in a block)
        n represents the maximum dimension of the PCA space (assumed to be atleast 16)

    sz: Integer
        The height and width of each block

    imname: string
        name of file where image will be saved.
    '''
    #TODO: Obtain top 16 components of D and plot them
    D_16=D[:,:16]
    f, axarr = plt.subplots(4,4)
    k=0
    for i in range(4):
        for j in range(4):
            plt.axes(axarr[i,j])
            plt.imshow(D_16[:,k].reshape((sz,sz)))
            k=k+1
            
    f.savefig(imname)
#     print 'output/hw1a_{0}_im{1}.png'.format(n_blocks, im_num)
    plt.close(f)
    return
    raise NotImplementedError


# In[5]:

def main():
    '''
    Read here all images(grayscale) from Fei_256 folder and collapse 
    each image to get an numpy array Ims with size (no_images, height*width).
    Make sure the images are read after sorting the filenames
    '''
    #TODO: Write a code snippet that performs as indicated in the above comment
    for root, dirs, images in os.walk("Fei_256/", topdown=False):
        pass
    imageslist=[]
    imageslist = map(lambda x: os.path.join("Fei_256", x), images)
    imageslist.sort()
    im=Image.open(imageslist[0])
    (width, height) = im.size
    Ims = np.zeros([len(imageslist), height*width])
    for i in  range(0, len(imageslist)):
        im=Image.open(imageslist[i])
        Ims[i,:]= np.array(im).flatten()
    
    Ims = Ims.astype(np.float32)
    X_mn = np.mean(Ims, 0)
    Images = Ims - np.repeat(X_mn.reshape(1, -1), Ims.shape[0], 0)
    D=np.random.rand(16,65536)
    eigen_val=np.zeros(16)
    for i in range(0,16):
        learning_para=0.5
        v=theano.shared(np.random.randn(65536))
        X=T.matrix()
#         X_tensor=X
        Xv=T.dot(X,v)
#         print Xv.get_value()
        
        if i==0:
            cost=T.dot(Xv.T,Xv)
            
        else:
            
            cost=T.dot(Xv.T,Xv)-np.sum(eigen_val[j]*T.dot(D[j],v)*T.dot(D[j],v) for j in xrange(i))
#             cost = T.dot(Xv.T, Xv) - np.sum(evals[j]*T.dot(evecs[j], v)*T.dot(evecs[j], v) for j in xrange(i))
        
        gradient=T.grad(cost,v)
        y=v+learning_para*gradient
        updated_v_after_descent=y/y.norm(2)
        final_updated_v=theano.function([X],updates=[(v,updated_v_after_descent)])
        t=1
        change =1
        print "iteration"+str(i)
        while t<50 and change>0.005:
            print "T \t"+str(t)
            print "change \t"+str(change)
#             set_random(x)
            final_updated_v(Images)
            new_di = v.get_value()

            change = np.linalg.norm(D[i]-new_di)
            D[i]=new_di


#             print "Change \t"+str(change)
            t=t+1
        tempdot = np.dot(Images, D[i])
        eigen_val[i] = np.dot(tempdot.T,tempdot)
        

            
            
        
    D = D.T
    c = np.dot(D.T, Images.T)

    '''
    Use theano to perform gradient descent to get top 16 PCA components of X
    Put them into a matrix D with decreasing order of eigenvalues

    If you are not using the provided AMI and get an error "Cannot construct a ufunc with more than 32 operands" :
    You need to perform a patch to theano from this pull(https://github.com/Theano/Theano/pull/3532)
    Alternatively you can downgrade numpy to 1.9.3, scipy to 0.15.1, matplotlib to 1.4.2
    '''
    
    #TODO: Write a code snippet that performs as indicated in the above comment
        
    for i in range(0, 200, 10):
        plot_reconstructions(D=D, c=c, num_coeff_array=[1, 2, 4, 6, 8, 10, 12, 14, 16], X_mean=X_mn.reshape((256, 256)), im_num=i)

    plot_top_16(D, 256, 'output/hw1b_top16_256.png')


if __name__ == '__main__':
    main()

