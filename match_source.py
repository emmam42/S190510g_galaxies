#Emma Mirizio: Matches radio sources from source finder output to NED optical catalogue sources
#June 4th 2019

#imports
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import Table
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
from astropy import wcs
import cv2
from mpl_toolkits.axes_grid1 import make_axes_locatable

#inputs
error = 5.0 #uncertainty in image in arcseconds: input

max_ra = 93.4917 #maximum RA within image limits
min_ra = 84.6250 #minimum RA within image limits
max_dec = -29.8583 #maximum declination within image limits
min_dec = -36.3556 #minimum declination within image limits

radius = 15 #integer value of amount to extend image in ech direction for radio images

#contour levels
c1 = .9
c2 = .7
c3 = .5
cc = ('deepskyblue','aquamarine','lawngreen','gold') #contour colors


########## Reading in data and taking only dat in range of the LIGO fov image ##########

#Read in data from NED
data_in = pd.read_csv("S190510g_Update_galaxies.csv", header=None, delimiter=",")

#extract ra dec and name, compress to only ones in range

gal_name = data_in.iloc[1:,0]
ra_deg = np.asarray(data_in.iloc[1:,1],dtype=np.float32)
dec_deg = np.asarray(data_in.iloc[1:,2],dtype=np.float32)
ra_range = [i if i < max_ra else float(-999) for i in ra_deg]
ra_range1 = ([i if i > min_ra else float(-999) for i in ra_range])
dec_range = [i if i < max_dec else float(-999) for i in dec_deg]
dec_range1 = ([i if i > min_dec else float(-999) for i in dec_range])
coords = np.asarray(list(zip(gal_name,ra_range1,dec_range1)))
coords_name = list(zip(gal_name,ra_range1,dec_range1))
#All optical sources within range, RA, Dec
coords1 = np.ma.compress_rows(np.ma.masked_where(coords == '-999.0',coords))
gal_name1 = coords1[:,0]
ra_cat = coords1[:,1]
ra_cat1 = [float(i) for i in ra_cat]
dec_cat = coords1[:,2]
dec_cat1 = [float(i) for i in dec_cat]


#Read in data from source finder
data_in1 = pd.DataFrame(pd.read_csv("RACS_test4_1.05_0537-37A_annotate.ann", header=None, delimiter=" "))
data_in2 = pd.DataFrame(pd.read_csv("RACS_test4_1.05_0538-31A_annotate.ann", header=None, delimiter=" "))
data_in3 = pd.DataFrame(pd.read_csv("RACS_test4_1.05_0607-31A_annotate.ann", header=None, delimiter=" "))
data_in4 = pd.DataFrame(pd.read_csv("RACS_test4_1.05_0607-37A_annotate.ann", header=None, delimiter=" "))
coords2 = pd.concat([data_in1,data_in2,data_in3,data_in4])
ra_source = np.asarray(coords2[2])
dec_source = np.asarray(coords2[3])

########## cross match radio galaxy to optical galaxy ##########

#cross matching 
radio = SkyCoord(ra=ra_source*u.degree, dec=dec_source*u.degree)  
optical = SkyCoord(ra=ra_cat1*u.degree, dec=dec_cat1*u.degree)  
idx, d2d, d3d = optical.match_to_catalog_sky(radio) 

#using separation constraint defined at top
max_sep = error * u.arcsec 
sep_constraint = d2d < max_sep 
optical_matches = optical[sep_constraint]
index_matches = [i for i, x in enumerate(sep_constraint) if x]


########## writing out data in convenient ways (annotation file, astropy table, numpy coord array) ##########

#creating lists/floats of matched galaxy names and RA Dec
galaxy_name_match = gal_name1[index_matches]

print(galaxy_name_match[0])
ra_match = ra_cat[index_matches]
ra_match1 = [float(i) for i in ra_match]
dec_match = dec_cat[index_matches]
dec_match1 = [float(i) for i in dec_match]

galnum = len(ra_match)

matched_galaxies = list(zip(galaxy_name_match,ra_match,dec_match))
d2dind = d2d[index_matches]
print(galaxy_name_match)
#create astropy table of matches
match_table = Table([galaxy_name_match, ra_match1, dec_match1], names=('full_name', 'ra', 'dec'), meta={'name': 'S190510g_matches'})
#print(match_table)
#create numpy array of coordinates to find on image 
zerocol = [0.0]*galnum
matched_gal_coords = np.column_stack((ra_match1,dec_match1))
#print(matched_gal_coords)

#Create .ann file to place matched galaxies on image
#define collumns as necessary data, ignore header rows, turn into lists

#Create collumn of shape and "W" for annotation file
ellipse_col = ['CIRCLE']*galnum
W_col = ['W']*galnum
width = [.01]*galnum
length = [.01]*galnum

##create list of all information
annotate_list = list(zip((ellipse_col),(W_col),(ra_match),(dec_match),width,length))
#annotate_list = list(zip((ellipse_col),(W_col),(ra_deg),(dec_deg),(maj_axis),(min_axis),(angle)))

#Turn each row into a string
for i in range(galnum):
	annotate_list[i] = str(annotate_list[i])

#Save list to a .ann file
f = open('matches_NEDOpticalGalaxiesS190510g.ann','w')
for s in annotate_list:
    f.write(s+'\n')
f.close()

# Read in the file in order to delete unwanted characters
with open('matches_NEDOpticalGalaxiesS190510g.ann', 'r') as file :
  filedata = file.read()

# Replace (delete) commas, parentheses and apostrophes
filedata = filedata.replace("'", "")
filedata = filedata.replace("(", "")
filedata = filedata.replace(")", "")
filedata = filedata.replace(",", "")

# Write the file out again
with open('matches_NEDOpticalGalaxiesS190510g.ann', 'w') as file:
  file.write(filedata)

################################################################################
########## make images of specific radio galaxies to extract contours ##########

print(matched_galaxies[0])

#four RACS images
filename1 = 'RACS_test4_1.05_0537-37A.I.fits' 
filename2 = 'RACS_test4_1.05_0538-31A.I.fits'
filename3 = 'RACS_test4_1.05_0607-31A.I.fits'
filename4 = 'RACS_test4_1.05_0607-37A.I.fits'

def get_data(filename):
	#import RACS image
	image_file = fits.open('{}'.format(filename))
	image_file.info()
	image_data = image_file[0].data[0,0,:,:]
	header = image_file['PRIMARY'].header
	wcs1 = wcs.WCS(header,naxis =2)

	# Convert the world coordinates of galaxies to pixel coordinates
	pixcrd2 = wcs1.wcs_world2pix(matched_gal_coords, 1)
	pixcrd2 = np.column_stack((pixcrd2[:,0],pixcrd2[:,1])) #list of pixel coords of galaxies

	#galaxy center point
	center_ra = int(pixcrd2[i,0])
	center_dec = int(pixcrd2[i,1])

	#take only center of image to create contours
	data_inrange = image_data[center_dec-radius:center_dec+radius,center_ra-radius:center_ra+radius]

	#get optical file by name: DSS now, could do DSS2...
	optfilename = '/import/ada2/emir1984/S190510g_ASKAPfield/gal_images/{}_DSS.fits'.format(galaxy_name_match[i])
	optfilename = optfilename.replace(' ','_')
	opt_file = get_pkg_data_filename(optfilename) 
	image_data2 = fits.getdata(opt_file)
	center_pix = int((len(image_data2))/2)
	image_data2 = (image_data2[center_pix-radius:center_pix+radius,center_pix-radius:center_pix+radius])
	
	#checks which RACS image the source is actually in, if its on the edge of one before it gets to where its in a better spot of the other image you'll get bad data?
	notin_image = np.isnan(image_data[center_ra,center_dec]) #true if data is outside of image
	if notin_image == False:
		ax = plt.subplot(projection=wcs1)
		brightest =np.amax(image_data[center_dec-radius:center_dec+radius,center_ra-radius:center_ra+radius])
		CS =plt.contour(data_inrange,levels = [c3*np.amax(data_inrange),c2*np.amax(data_inrange),c1*np.amax(data_inrange)], colors=cc)
		plt.clabel(CS, fontsize=9, inline=1)
		
		cp1 = int(c1*100)
		cp2 = int(c2*100)
		cp3 = int(c3*100)
		labels = ['{}% of maximum'.format(cp3), '{}% of maximum'.format(cp2),'{}% of maximum'.format(cp1)]
		for j in range(len(labels)):
    			CS.collections[j].set_label(labels[j])

		plt.legend(loc='upper left')
		fig = ax.imshow(image_data2, cmap='gray') #shows DSS image
		#fig = ax.imshow(data_inrange, cmap='gray',vmax=brightest) #shows radio image
		#colorbar settings
		divider = make_axes_locatable(ax)
		cax = divider.append_axes("right", size="10%", pad=0.9)
		c = plt.colorbar(fig, cax=cax)
		c.set_label('', rotation=180,labelpad=-8) 
		cax.yaxis.set_ticks_position('right') #NOT WORKING
		cax.yaxis.set_label_position('right') #NOT WORKING
		plt.grid(b=None)
		ax.set_xlabel('Right Ascension (Degrees)')
		ax.set_ylabel('Declination (Degrees)',labelpad=-1)
		ax.set_title('Galaxy {}: DSS image with Radio Contours'.format(galaxy_name_match[i]),fontsize = 12)
		#plt.subplot(projection=w)
		#plt.show()
		plt.savefig('contour_figs/galfig{:03d}.png'.format(i))
	else:
		oops = 0/0 #throws an error so that it'll move onto the except


for i in range(58,77):
	try:
		get_data(filename1)
		
	except:
		try:
			get_data(filename2)
			
		except:
			try:
				get_data(filename3)
				
			except:
				get_data(filename4)
				



#####################################################################
#     download optical images and save them to gal_images folder    #

"""

from astroquery.skyview import SkyView
import urllib.request

#target=match_table[0]
#survey_list = ['DSS', 'DSS1 Blue', 'DSS1 Red', 'DSS2 Red', 'DSS2 Blue', 'DSS2 IR']
#survey_list = ['DSS']


def get_source_images(target, survey_list, pixels=512, folder='/import/ada2/emir1984/S190510g_ASKAPfield/gal_images/', verbose=True):

  '''
  Download postagestamp images of targets from SkyView
  
  :param target: An astropy table row containing the name and coordinates of the target
  :param survey_list: A list of strings containing survey names
  :param pixels: An integer, the dimensions of the requested (square) image
  :param folder: A string, the location where downloaded images should be saved
  :param verbose: A boolean, determine whether or not to print errors
  
  '''


  paths = SkyView.get_image_list(position='%s, %s'%(target['ra'], target['dec']), survey=survey_list) # can't use get_image() because it throws errors when files don't exist
  print(paths)

  for survey, path in zip(survey_list, paths):
    filepath = "%s%s_%s.fits"%(folder,target['full_name'], survey)
    filepath = filepath.replace(' ','_')
    
    try:
      urllib.request.urlretrieve(path,filepath)
    except urllib.error.HTTPError:
      if verbose:
        print("%s not in %s"%(target['full_name'], survey))
      pass

for i in range(70,75):
    get_source_images(target=match_table[i], survey_list = ['DSS', 'DSS1 Blue', 'DSS1 Red', 'DSS2 Red', 'DSS2 Blue', 'DSS2 IR'])

"""
