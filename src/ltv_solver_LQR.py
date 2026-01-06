# Optimal Control and Reinforcement Learning Project 2025-2026
# Group 23: Alessandro Binci, Alessandro Tampieri, Lorenzo Tucci
# Problem 2 with set of parameters 1

import numpy as np
import cvxpy as cp

def ltv_LQR(AAin, BBin, QQin, RRin, SSin, QQfin, TT, x0, qqin = None, rrin = None, qqfin = None):

  """
	LQR for LTV system with (time-varying) affine cost
	
  Args
    - AAin (nn x nn (x TT)) matrix
    - BBin (nn x mm (x TT)) matrix
    - QQin (nn x nn (x TT)), RR (mm x mm (x TT)), SS (mm x nn (x TT)) stage cost
    - QQfin (nn x nn) terminal cost
    - qq (nn x (x TT)) affine terms
    - rr (mm x (x TT)) affine terms
    - qqf (nn x (x TT)) affine terms - final cost
    - TT time horizon
  Return
    - KK (mm x nn x TT) optimal gain sequence
    - PP (nn x nn x TT) riccati matrix
"""
  try:
    # check if matrix is (.. x .. x TT) - 3 dimensional array 
    ns, lA = AAin.shape[1:]
  except:
    # if not 3 dimensional array, make it (.. x .. x 1)
    AAin = AAin[:,:,None]
    ns, lA = AAin.shape[1:]

  try:  
    ni, lB = BBin.shape[1:]
  except:
    BBin = BBin[:,:,None]
    ni, lB = BBin.shape[1:]

  try:
      nQ, lQ = QQin.shape[1:]
  except:
      QQin = QQin[:,:,None]
      nQ, lQ = QQin.shape[1:]

  try:
      nR, lR = RRin.shape[1:]
  except:
      RRin = RRin[:,:,None]
      nR, lR = RRin.shape[1:]

  try:
      nSi, nSs, lS = SSin.shape
  except:
      SSin = SSin[:,:,None]
      nSi, nSs, lS = SSin.shape

  # Check dimensions consistency -- safety
  if nQ != ns:
    print("Matrix Q does not match number of states")
    exit()
  if nR != ni:
    print("Matrix R does not match number of inputs")
    exit()
  if nSs != ns:
    print("Matrix S does not match number of states")
    exit()
  if nSi != ni:
    print("Matrix S does not match number of inputs")
    exit()


  if lA < TT:
    AAin = AAin.repeat(TT, axis=2) 
  if lB < TT:                       
    BBin = BBin.repeat(TT, axis=2) 
  if lQ < TT:
    QQin = QQin.repeat(TT, axis=2)  #ripete TT volte la matrice in modo da renderla TT dimensionale nella terza dimensione
  if lR < TT:
    RRin = RRin.repeat(TT, axis=2)  #ripete TT volte la matrice in modo da renderla TT dimensionale nella terza dimensione
  if lS < TT:
    SSin = SSin.repeat(TT, axis=2)  #ripete TT volte la matrice in modo da renderla TT dimensionale nella terza dimensione

  # Inizialization of the matrices
  KK = np.zeros((ni, ns, TT))
  sigma = np.zeros((ni, TT))
  PP = np.zeros((ns, ns, TT))
  pp = np.zeros((ns, TT))

  # Renaming the input matrices
  QQ = QQin
  RR = RRin
  SS = SSin
  QQT = QQfin
    
  qq = qqin
  rr = rrin

  qqT = qqfin

  AA = AAin
  BB = BBin

  xx = np.zeros((ns, TT))
  uu = np.zeros((ni, TT))

  xx[:,0] = x0    # the first variation delta_x0 = 0

  #Set the final conditions

  PP[:,:,-1] = QQT
  pp[:,-1] = qqT

# Solve Riccati equation
  for tt in reversed(range(TT-1)):
    QQt = QQ[:,:,tt]
    qqt = qq[:,tt][:,None]
    RRt = RR[:,:,tt]
    rrt = rr[:,tt][:,None]
    AAt = AA[:,:,tt]
    BBt = BB[:,:,tt]
    SSt = SS[:,:,tt]
    PPtp = PP[:,:,tt+1]
    pptp = pp[:, tt+1][:,None]

    MMt_inv = np.linalg.inv(RRt + BBt.T @ PPtp @ BBt)
    mmt = rrt + BBt.T @ pptp
    
    PPt = AAt.T @ PPtp @ AAt - (BBt.T@PPtp@AAt + SSt).T @ MMt_inv @ (BBt.T@PPtp@AAt + SSt) + QQt
    ppt = AAt.T @ pptp - (BBt.T@PPtp@AAt + SSt).T @ MMt_inv @ mmt + qqt

    PP[:,:,tt] = PPt
    pp[:,tt] = ppt.squeeze() #CHIARIMENTI SU SQUEEZE AL PROF

# Evaluate KK
  
  for tt in range(TT-1):
    QQt = QQ[:,:,tt]
    qqt = qq[:,tt][:,None]
    RRt = RR[:,:,tt]
    rrt = rr[:,tt][:,None]
    AAt = AA[:,:,tt]
    BBt = BB[:,:,tt]
    SSt = SS[:,:,tt]

    PPtp = PP[:,:,tt+1]
    pptp = pp[:,tt+1][:,None]

    # Check positive definiteness  CHIARIMENTI

    MMt_inv = np.linalg.inv(RRt + BBt.T @ PPtp @ BBt)
    mmt = rrt + BBt.T @ pptp

    

    KK[:,:,tt] = -MMt_inv@(BBt.T@PPtp@AAt + SSt)
    sigma_t = -MMt_inv@mmt

    sigma[:,tt] = sigma_t.squeeze()

  for tt in range(TT - 1):
      
      # Trajectory

      uu[:, tt] = KK[:,:,tt]@xx[:, tt] + sigma[:,tt]    # updating variation deltau of the LQR problem
      xx_p = AA[:,:,tt]@xx[:,tt] + BB[:,:,tt]@uu[:, tt] # updating variation deltax of the LQR problem

      xx[:,tt+1] = xx_p

      xxout = xx
      uuout = uu

  return KK, sigma, PP, xxout, uuout