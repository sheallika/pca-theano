
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
from numpy import linalg as LA

import theano
import theano.tensor as T


# In[2]:


'''
Implement the functions that were not implemented and complete the
parts of main according to the instructions in comments.
'''


# In[3]:


def reconstructed_image(D,c,num_coeffs,X_mean,n_blocks,im_num,sz):
    '''
    This function reconstructs an image X_recon_img given the number of
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
        

    n_blocks: Integer
        number of blocks comprising the image in each direction.
        For example, for a 256x256 image divided into 64x64 blocks, n_blocks will be 4
    '''
    
    c_im = c[:num_coeffs,n_blocks*n_blocks*im_num:n_blocks*n_blocks*(im_num+1)]
    D_im = D[:,:num_coeffs]   
    
#     X_recon_img=np.zeros([256,256])
    temp=np.dot(D_im,c_im).T 
    temp=temp+X_mean.flatten() 
    temp=np.array(temp,dtype=np.float32)
#     temp=T.tensor2(temp)
    images = T.tensor4('images')
    neibs = T.nnet.neighbours.images2neibs(images, neib_shape=(sz, sz))
    im_new = T.nnet.neighbours.neibs2images(neibs, (sz, sz), (1,1,256,256))
    inv_window = theano.function([neibs], im_new)
# Function application
#     im_new_val = inv_window(neibs_val)

    X_recon_img=inv_window(temp)
#     temp=temp+X_mean.flatten() 
#     for k in range(0,len(temp)):
#         for i in range(0,256,sz):
#                 for j in range(0,256,sz):
#                     box=(i,j,i+sz,j+sz)
#                     X_recon_img[i:i+sz,j:j+sz]=temp[k,:].reshape((sz,sz))

#     X_recon_img=temp.reshape((256,256))
    
    
    #TODO: Enter code below for reconstructing the image X_recon_img
    #......................
    #......................
    #X_recon_img = ........
#     print X_recon_img[0][0].eval()
    return X_recon_img.reshape(256,256)


# In[4]:

def plot_reconstructions(D,c,num_coeff_array,X_mean,n_blocks,im_num,sz):
    '''
    Plots 9 reconstructions of a particular image using D as the basis matrix and coeffiecient
    vectors from c

    Parameters
    ------------------------
        num_coeff_array: Iterable
            an iterable with 9 elements representing the number of coefficients
            to use for reconstruction for each of the 9 plots
        
        c: np.ndarray
            a l x m matrix  representing the coefficients of all blocks in a particular image
            l represents the dimension of the PCA space used for reconstruction
            m represents the number of blocks in an image

        D: np.ndarray
            an N x l matrix representing l basis vectors of the PCA space
            N is the dimension of the original space (number of pixels in a block)

        n_blocks: Integer
            number of blocks comprising the image in each direction.
            For example, for a 256x256 image divided into 64x64 blocks, n_blocks will be 4

        X_mean: basis vectors represent the divergence from the mean so this
            matrix should be added to all reconstructed blocks

        im_num: Integer
            index of the image to visualize
    '''
    f, axarr = plt.subplots(3,3)
    for i in range(3):
        for j in range(3):
            plt.axes(axarr[i,j])
            plt.imshow(reconstructed_image(D,c,num_coeff_array[i*3+j],X_mean,n_blocks,im_num,sz), interpolation=None)
            
    f.savefig('output/hw1a_{0}_im{1}.png'.format(n_blocks, im_num))
    print 'output/hw1a_{0}_im{1}.png'.format(n_blocks, im_num)
    plt.close(f)


# In[5]:

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

    
    


# In[6]:

def main():
    '''
    Read here all images(grayscale) from Fei_256 folder
    into an numpy array Ims with size (no_images, height, width).
    Make sure the images are read after sorting the filenames
    '''
    #TODO: Read all images into a numpy array of size (no_images, height, width)
    for root, dirs, images in os.walk("Fei_256/", topdown=False):
        pass
    imageslist=[]
    imageslist = map(lambda x: os.path.join("Fei_256", x), images)
    imageslist.sort()
    im=Image.open(imageslist[0])
    (width, height) = im.size
    allimages = np.zeros([len(imageslist), height, width])
    for i in  range(0, len(imageslist)):
        im=Image.open(imageslist[i])
        allimages[i,:,:]= np.array(im,dtype=np.float32)
#     allimages=T.tensor4(allimages)
        
    szs = [8,32,64]
    num_coeffs = [range(1, 10, 1), range(3, 30, 3), range(5, 50, 5)]


    for sz, nc in zip(szs, num_coeffs):
        print sz
#         block_images=[]
        block_images=np.ndarray([len(imageslist)*(256*256)/(sz*sz),sz*sz])
        l=0
#         temp= np.zeros([sz,sz])
        im_val = np.arange(256*256).reshape((1,1,256, 256))
    
        images = T.tensor4('images')
        neibs = T.nnet.neighbours.images2neibs(images, neib_shape=(sz,sz))
        window_function = theano.function([images], neibs)
        for k in range(0,len(imageslist)):
#             temp=T.tensor4('allimages[k,:,:]')
            im_val[0][0]=allimages[k,:,:]
            neibs_val=window_function(np.float32(im_val))
#             print neibs_val.shape
            
            block_images[(k)*(256*256)/(sz*sz):(k+1)*(256*256)/(sz*sz),:]=neibs_val 
            
            
#             for i in range(0,width,sz):
#                 for j in range(0,height,sz):
#                     box=(i,j,i+sz,j+sz)
#                     temp=allimages[k,j:j+sz,i:i+sz]
#                     temp=temp.flatten()
#                     block_images[l,:]=temp
#                     l=l+1


 
    
      #TODO: Write a code snippet that performs as indicated in the above comment
#         print block_images.shape
        X=np.array(block_images)    
        X_mean= np.mean(X, 0)
#         print X.shape
        X = X - np.repeat(X_mean.reshape(1, -1), X.shape[0], 0)
        cov_matrix=np.dot(X.T,X)
        eigen_val, eigen_vector= LA.eigh(cov_matrix)
#         print eigen_val
#         D=eigen_vector[::-1]
        idx=np.argsort(-eigen_val)
        D=eigen_vector[:,idx]
#         print D

        '''
        Perform eigendecomposition on X^T X and arrange the eigenvectors
        in decreasing order of eigenvalues into a matrix D
        '''

        #TODO: Write a code snippet that performs as indicated in the above comment

        c = np.dot(D.T, X.T)

        for i in range(0, 200, 10):
            
            plot_reconstructions(D=D, c=c, num_coeff_array=nc, X_mean=X_mean.reshape((sz, sz)), n_blocks=int(256/sz), im_num=i,sz=sz)
            
        plot_top_16(D, sz, imname='output/hw1a_top16_{0}.png'.format(sz))




        
      


# In[7]:

if __name__ == '__main__':
    main()

