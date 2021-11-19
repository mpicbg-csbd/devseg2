import numpy as np
import ipdb

def place_label_spheres(pts,labelvals,sh,scale,radius=7):
  s  = np.array(scale)
  ndim = len(sh)
  # ks = np.ceil(s*7).astype(np.int) #*6 + 1 ## gaurantees odd size and thus unique, brightest center pixel
  ks = np.ceil(radius*np.array([2]*ndim)).astype(np.uint32)
  ks = ks - ks%2 + 1## enfore ODD shape so kernel is centered! (grow even dims by 1 pix)

  def f(x):
    x = x - (ks-1)/2
    # return np.exp(-(x*x/s/s).sum()/2)
    return max(radius - np.sqrt((x*x*s*s).sum()) , 0) ## takes value `radius` at r=0 i.e. distance to boundary
  kern = np.array([f(x) for x in np.indices(ks).reshape((ndim,-1)).T]).reshape(ks)
  # kern = kern / kern.max()
  # target = conv_at_pts4(pts,kern,sh,lambda a,b:np.maximum(a,b))

  dist = np.zeros(sh + ks) ## add kernel shape as padding
  labels = np.zeros(sh + ks, dtype=np.uint16) ## add kernel shape as padding
  pts = pts + ks//2  ## translate points into padded coord system
  labkern = (kern>0).astype(np.uint16)

  for i,p in enumerate(pts):
    ss =       (slice(p[0]-ks[0]//2 , p[0]+ks[0]//2 + 1),
                slice(p[1]-ks[1]//2 , p[1]+ks[1]//2 + 1))
    if ndim==3: 
    	ss = ss + (slice(p[2]-ks[2]//2 , p[2]+ks[2]//2 + 1),)
                
    lval = labelvals[i]
    mask = dist[ss] > kern
    labels[ss] = np.where(mask, labels[ss] , labkern*lval)
    dist[ss]   = np.where(mask, dist[ss] , kern)

  ss =       (slice(ks[0]//2 , - (ks[0]//2 + 1)),
              slice(ks[1]//2 , - (ks[1]//2 + 1)))
  if ndim==3: 
  	ss = ss + (slice(ks[2]//2 , - (ks[2]//2 + 1)),)

  labels = labels[ss]
  return labels
