#!/usr/bin/env python
from time import sleep
import numpy as np
import serial
import subprocess
import argparse
import sys
import os
import struct
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from scipy.signal import argrelmax
import matplotlib
# matplotlib.use('TKAgg')
import matplotlib.pyplot as plt


#Record the argv in .out file, and create the Arduinio sketch
def initialize():
    file=open((data+"/"+filebase+"/"+filebase+".out"),"w")
    print(*sys.argv, file=file)
    file.close()
    #Add a
    program="void setup(){Serial.begin(115200);Serial.setTimeout(1);}\nvoid loop(){"+\
    "if(!Serial.available()){"+\
    "Serial.readString(); unsigned long t_last; int count = 0;"+\
    (("unsigned short Xvals[%i]; unsigned short Yvals[%i];"%(Nt,Nt)) if xy==1 else "")+\
    ("unsigned short Zvals[%i];\n"%(Nt))+\
    ("unsigned long tvals[%i];\n"%(Nt))+\
    "{ count=0; Zvals[count]=analogRead(A0); t_last=micros(); while(count<%i) {"%(Nt)+\
    "if(micros()-t_last > 1000*%f) {" %(delay)+\
      ("Xvals[count]=analogRead(A2); Yvals[count]=analogRead(A1);" if xy==1 else "")+\
      "Zvals[count]=analogRead(A0); tvals[count]=micros(); t_last=tvals[count]; count = count+1; }}"+\
      "for (int i=0; i<%i; i++){"%(Nt)+\
            ("Serial.println(Xvals[i]); Serial.println(Yvals[i]);" if xy==1 else "")+\
            "Serial.println(Zvals[i]); Serial.println((tvals[i]-tvals[0])); } }}}"
    file=open(data+"/"+filebase+"/"+filebase+".ino","w")
    file.write(program)
    file.close()

    print(sys.platform)
    #Use the newer arduino-cli, and check if the -p port is necessary
    subprocess.run(["./arduino-cli-mac", "compile", "--fqbn", "arduino:avr:uno", data+"/"+filebase])
    if(run==1):
        subprocess.run(["./arduino-cli-mac", "upload", "-p", "/dev/cu.usbmodem14101", "--fqbn", "arduino:avr:uno", data+"/"+filebase])
        print(*["./arduino-cli-mac", "upload", "-p", "/dev/cu.usbmodem14101", "--fqbn", "arduino:avr:uno", data+"/"+filebase])


#Fitting sinusoidal function
def sinfunc(t, A, f, p, c):  return A * np.sin(2*np.pi*f*t + p) + c

#Send the sketch to the Arduino and read the serial port to retrieve Nt data points
def get_sample():
    xlst=[]
    ylst=[]
    zlst=[]
    tlst=[]
    if run==1:
        print('Reading serial port')
        ser = serial.Serial(port='/dev/cu.usbmodem14101', baudrate=115200)
        ser.flushInput()
        ser.flushOutput()

        ser.write(bytes("1\n", 'utf-8')) #I can send Nt and delay here.
        ser.write(bytes("1\n", 'utf-8')) #I can send Nt and delay here.
        # sleep(0.1)
        t=0

        for i in range(Nt):
            print(i,end='   \r')
            if(xy==1):
                xlst.append(int(ser.readline().decode("utf-8")))
                ylst.append(int(ser.readline().decode("utf-8")))
            zlst.append(int(ser.readline().decode("utf-8")))
            tlst.append(0.001*int(ser.readline().decode("utf-8")))

        #This will depend on the accelerometer
        amax=5
        amin=-5
        vscale=5


        zlst=amin+(amax-amin)/(3.3/vscale*1024)*np.array(zlst)
        if(xy==1):
            xlst=amin+(amax-amin)/(3.3/vscale*1024)*np.array(xlst)
            ylst=amin+(amax-amin)/(3.3/vscale*1024)*np.array(ylst)



    else:
        if(xy==1):
            xlst=np.load(data+"/"+filebase+"/"+filebase+"x%i"%(count)+".npy")
            ylst=np.load(data+"/"+filebase+"/"+filebase+"y%i"%(count)+".npy")
        tlst=np.load(data+"/"+filebase+"/"+filebase+"t%i"%(count)+".npy")
        zlst=np.load(data+"/"+filebase+"/"+filebase+"z%i"%(count)+".npy")

    if(xy==1):
        xlst=np.array(xlst)
        ylst=np.array(ylst)
    tlst=np.array(tlst)/1000
    zlst=np.array(zlst)
    #
    # if args.freq >0:
    #     freq0=args.freq
    # else:
    freq0=(1+np.argmax(np.abs(np.fft.rfft(zlst))[1:]))/tlst[-1]

    acc0=0.5*(np.max(zlst)-np.min(zlst))
    cx=np.mean(xlst)
    cy=np.mean(ylst)
    cz=np.mean(zlst)
    if (zlst[0]-cz)/acc0>1:
        phi0=np.pi/2
    elif (zlst[0]-cz)/acc0<-1:
        phi0=-np.pi/2
    elif (zlst[1]>zlst[0]):
        phi0=np.arcsin((zlst[0]-cz)/acc0)
    else:
        phi0=np.pi-np.arcsin((zlst[0]-cz)/acc0)
    popt, pcov = curve_fit(sinfunc, tlst, zlst, p0=[acc0,freq0,phi0,cz])
    acc, freq, phi, c = popt
    dacc, dfreq, dphi, dc = np.sqrt(np.diag(pcov))
    # print("[acc0,freq0,phi0,cx,cy,cz]=",[acc0,freq0,phi0,cx,cy,cz])
    print("[acc,freq,phi,c]=",popt)
    print()
    return xlst,ylst,zlst,tlst,acc,freq,phi,c,cx,cy

#Append the fit parameters to .txt file, and save the time series
def save_data(count,xlst,ylst,zlst,tlst,acc,freq,phi,cz,cx,cy):
    file=open((data+"/"+filebase+"/"+filebase+".txt"),"a+")
    print("%i %f %f %f %f %f %f"%(count, acc, freq, phi, cz, cx, cy),file=file)
    file.close()
    if xy:
        np.save(data+"/"+filebase+"/"+filebase+"x"+"%i"%(count),xlst)
        np.save(data+"/"+filebase+"/"+filebase+"y"+"%i"%(count),ylst)
        xlst=np.array(xlst)
        ylst=np.array(ylst)
    np.save(data+"/"+filebase+"/"+filebase+"t"+"%i"%(count),tlst)
    np.save(data+"/"+filebase+"/"+filebase+"z"+"%i"%(count),zlst)

#Plot the time series and save a pdf
def plot_data(count,xlst,ylst,zlst,tlst,acc,freq,phi,c):
    plt.clf()
    if xy:
        plt.subplot(233,xlabel="Time (s)",ylabel="Acceleration (g)",title="X acceleration",ylim=(-0.5,0.5))
        plt.plot(tlst,xlst,'bx',markersize=2.0)
        plt.subplot(234,xlabel="Time (s)",ylabel="Acceleration (g)",title="Y acceleration",ylim=(-0.5,0.5))
        plt.plot(tlst,ylst,'bx',markersize=2.0)
    plt.subplot(231,xlabel="Time (s)",ylabel="Acceleration (g)",title="Z acceleration",ylim=(-1,3))
    plt.plot(tlst,np.array([sinfunc(t,acc,freq,phi,c) for t in tlst]),'r')
    plt.plot(tlst,zlst,'bx',markersize=2.0)
    plt.subplot(232,xlabel="Time (s)",ylabel="Acceleration (g)",title="Z residual")
    plt.plot(tlst,zlst-np.array([sinfunc(t,acc,freq,phi,c) for t in tlst]),'bx',markersize=2.0)
    plt.subplot(235,xlabel="Index",ylabel="Time (s)",title="Measurement times")
    plt.plot(np.arange(len(tlst)),tlst,'bx',markersize=2.0)

    plt.tight_layout()
    plt.savefig(data+"/"+filebase+"/"+filebase+"%i"%(count)+".pdf")
    plt.show(block=False)
    plt.pause(0.1)

#Plot the acceleration vs. frequency for saved data
def plot_sweep():
    try:
        dat=np.loadtxt(data+"/"+filebase+"/"+filebase+".txt")
        plt.subplot(111,xlabel="Frequency (Hz)",ylabel="Acceleration (g)")
        plt.plot(dat[:,2],dat[:,1],'bx',markersize=2.0)
        plt.savefig(data+"/"+filebase+"/"+filebase+".pdf")
        plt.show(block=False)
        plt.pause(0.1)
    except Exception  as error:
        print("Could not plot - did you save any data?")
        print(error)

#Command-line arguments
parser = argparse.ArgumentParser(description='Upload an Arduino sketch and read output from the accelerometer.')
parser.add_argument("--filebase", type=str, required=True, dest='filebase', help='Base string for file output.')
parser.add_argument("--directory", type=str, required=False, default='data', dest='data', help='Directory to save files. Default "data".')
parser.add_argument("--Nt", type=int, required=False, dest='Nt', default=150, help='Number of buffer ints. Default 150.')
parser.add_argument("--delay", type=float, required=False, dest='delay', default=2.0, help='Delay between samples. Default 2.0.')
parser.add_argument("--xy", type=int, choices=[0,1], required=False, default=1, dest='xy', help='Flag for x and y output. Default 1.')
# parser.add_argument("--freq", type=float, required=False, default=0, dest='freq', help='Frequency for fitting')
parser.add_argument("--run", type=int, choices=[0,1], required=False, default=1, dest='run', help='Flag for running arduino and reading output; if 0, data is read from previous runs if files exist. Default 1.')
parser.add_argument("--count", type=int, required=False, default=0, dest='count', help='Initial count. Default 0.')

args = parser.parse_args()
filebase=args.filebase
data=args.data
Nt=args.Nt
delay=args.delay
xy=args.xy
run=args.run
count=args.count

if not os.path.isdir(data+"/"+filebase):
    if not os.path.isdir(data):
        os.mkdir(data)
    os.mkdir(data+"/"+filebase)

#Main loop
initialize()

try:
    xlst,ylst,zlst,tlst,acc,freq,phi,cz,cx,cy=get_sample()
except:
    print("Failed to get sample or fit data! Is Arduino busy?")
while True:
    print("Enter 'p' to plot current data, 'P' to plot sweep of saved data, 's' to save data, 'r' to resample, 'd' to specify delay, or 'q' to quit")
    input=sys.stdin.readline()
    if input == 's\n':
        save_data(count,xlst,ylst,zlst,tlst,acc,freq,phi,cz,cx,cy)
        count=count+1
    elif input == 'p\n':
        plt.close()
        plt.subplots(2,3,figsize=(10,5))
        plot_data(count,xlst,ylst,zlst,tlst,acc,freq,phi,cz)
    elif input == 'P\n':
        plot_sweep()
    elif input == 'r\n':
        try:
            xlst,ylst,zlst,tlst,acc,freq,phi,cz,cx,cy=get_sample()
        except:
            print("Fit failed!")
    elif input=='d\n':
        print("Enter delay in ms (greater than 2.0), or 'a' for automatic based on last frequency:")
        line=sys.stdin.readline()
        try:
            if line == 'a\n':
                delay = 10*1e3/freq/Nt #10 cycles in sample
                print(delay)
            else:
                delay=float(line)
        except:
            print("Bad input %s"%(line))
        # if delay < 2.0:
        #     delay = 2.0
        print("Delay set to %f"%(delay))
        initialize()
    elif input == 'q\n':
        break
    elif input == '\n':
        continue
    else:
        print("Unrecognized input '%s' Try again"%(input))
