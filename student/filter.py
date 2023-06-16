# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Kalman filter class
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params 

class Filter:
    '''Kalman filter class'''
    def __init__(self):
        pass

    def F(self):
        ############
        # TODO Step 1: implement and return system matrix F
        ############
        dt = params.dt
        F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        return F
        
        ############
        # END student code
        ############ 

    def Q(self):
        ############
        # TODO Step 1: implement and return process noise covariance Q
        ############
        q = params.q
        dt = params.dt
        Q = np.zeros((6,6))
        Q[0][0] = q*pow(dt,3)/3
        Q[0][2] = q*pow(dt,2)/2
        Q[1][1] = q*pow(dt,3)/3
        Q[1][3] = q*pow(dt,2)/2
        Q[2][0] = q*pow(dt,2)/2
        Q[2][2] = q*dt
        Q[3][1] = q*pow(dt,2)/2
        Q[3][3] = q*dt
        Q[4][4] = q*pow(dt,3)/3
        Q[5][5] = q*pow(dt,3)/3
        return Q
        
        ############
        # END student code
        ############ 

    def predict(self, track):
        ############
        # TODO Step 1: predict state x and estimation error covariance P to next timestep, save x and P in track
        ############
        F = self.F()
        Q = self.Q()
        new_x = F * track.x
        new_P = F * track.P * np.transpose(F) + Q
        track.set_x(new_x)
        track.set_P(new_P)
        
        ############
        # END student code
        ############ 

    def update(self, track, meas):
        ############
        # TODO Step 1: update state x and covariance P with associated measurement, save x and P in track
        ############
        H = meas.sensor.get_H(track.x)
        S = self.S(track, meas, H)
        gamma = self.gamma(track, meas)
        I = np.identity(params.dim_state)
        K = track.P * np.transpose(H) * np.linalg.inv(S)
        new_x = track.x + K * gamma
        new_P = (I - K*H)*track.P
        track.set_x(new_x)
        track.set_P(new_P)
        ############
        # END student code
        ############ 
        track.update_attributes(meas)
    
    def gamma(self, track, meas):
        ############
        # TODO Step 1: calculate and return residual gamma
        ############
        gamma_ = meas.z - meas.sensor.get_hx(track.x)
        return gamma_
        
        ############
        # END student code
        ############ 

    def S(self, track, meas, H):
        ############
        # TODO Step 1: calculate and return covariance of residual S
        ############
        S_ = H * track.P * np.transpose(H) + meas.R
        return S_
        
        ############
        # END student code
        ############ 