
# Libs
import sounddevice as sd
import numpy as np
from scipy.fftpack import fft
import os
import copy
import math
import time

# Local imports
from constants import *



def getAvgedKernels(inputArray, numKernels):

    # Decide on the step-size needed
    stepSize = math.floor(len(inputArray) / numKernels)

    return np.mean(inputArray.reshape(-1, stepSize), 1)



class StringStreamer:
    '''
        The StringStreamer class provides functionality for streaming and detecting notes played on a single string.

        windowBuffer : A buffer filled with the current window buffer.

    '''

    # Constructor
    def __init__(self, name, valueCallback):

        # Init params
        self.name = name
        self.streamSampleRate = SAMPLE_RATE
        self.blocksize = AUDIO_BUFFER_SAMPLES_PER_CALLBACK
        self.windowSizeSamps = FFT_WINDOW_SIZE_SAMPS
        self.windowBuffer = np.asarray( [0 for _ in range(self.windowSizeSamps)] )
        self.valueCallback = valueCallback

        # Hamming Window 
        self.hammingWindow = np.hamming(self.windowSizeSamps)

        # Harmonic Product Spectrums
        self.harmonicProdSpectrums = 8

        # TODO @Marcel: Make robust for other devices with more channels! (Attach each to a string, etc.)
        # Init stream
        self.stream = sd.InputStream(callback = self.audioBufferCallback,
                                    blocksize = self.blocksize,
                                    channels = 1,
                                    samplerate = self.streamSampleRate
            )

        self.stream.start()

        self.currentTime = 0

        # A flag used so we don't count impulses twice
        self.impulseSeenLastWindow = False

        return


    def tFromCurWindowIdx(self, windowIdx):

        return (self.currentTime + windowIdx / self.streamSampleRate)


    # Audio Buffer Callback
    def audioBufferCallback(self, indata, frame_count, time_info, status):

        # TODO @Marcel: Move constants over to members, as makes sense.

        print("\n\n" , self.name + " :Buffer Hit...")
        # print("CurTime: ", self.currentTime)

        # Progress window buffer
        self.updateWindowWithBuffer(indata, frame_count)

        # Increment time 
        self.currentTime += len(indata) / self.streamSampleRate


        # Detect any buffer impulses!
        bufferImpulses = self.detectImpulses()

        # Discard any redundant impulses
        newBufferImpulses = [ ]
        for impulseIdx in bufferImpulses:
            
            # Make sure the idx is in the range of what was taken in.
            if not self.impulseSeenLastWindow or impulseIdx >= (len(self.windowBuffer) - len(indata)) :
                newBufferImpulses.append(impulseIdx)

        # TODO @Marcel: This is discarding good indices!!! Needs a better "lastDetected" check...
        
        # Update flag for impulse taken in
        if len(newBufferImpulses) > 0:
            self.impulseSeenLastWindow = True
        else:
            self.impulseSeenLastWindow = False

            # NOTE: We'll likely need to note these, then get the best reading on them as they go. 


        # When performing FFT, the WINDOW actually introduces some bias. 
        # Imagine Sinusoidal frequencies whose true FFT is zero at all but  one freq...
        # ... but becuase we're windowing, the FFT picks up some other freuqencies. 

        # Low sampling also causes this! Not enough granularity to define make non-important spectral peaks converge to 0.

        # Generally speaking, this is all known as Spectral Leakage.

        '''
        N (samples per window) / fs is the # seconds per window. 

        fs / N (samples per window) is the corresponding frequecy with a period of one window. 

        Frequencies that are integer multiples of fs / N will go through a full cycle in the course of the window AND will always be starting and ending cycles at the start and end of windows. 

        * HOWEVER, any other frequencies will NOT go through clean cycles within the window range. 

        So if the signal is only comprised of those... great. All other frequencies will be zero. An exact match found.
        If not... some frequencies NOT in the spectral sample will be present in the analysis. 

        Because there are sharp jumps into and out of the frequencies.

        A popular solution is a Hamming Window.
        This is a window that approachs 0 at n=0 and n=N-1. And reaches a peak of 1 @ n = N/2
        https://mil.ufl.edu/nechyba/www/__eel3135.s2003/lectures/lecture19/spectral_leakage.pdf

        This lowers the significance of those discontinuous regions on the ends. 

        '''

        # Multiply input by hamming window (for help preventing Spectral Leaking)
        filteredBuffer = self.windowBuffer * self.hammingWindow

        # Get raw FFT data
        fftValues = abs ( fft(filteredBuffer)[ : FFT_MAX_FREQUENCY] )


        # Supress main frequency hum
        for i in range( int(62/(self.streamSampleRate/self.windowSizeSamps)) ):
            fftValues[i] = 0 

        '''
    
        Now we need to take care of the harmonic analysis... to ensure no harmonics take precedence over the fundamentals.

        To do this, we'll 

            - Kick low energy frequencies out of the octave bands. 
            
            - Upsample the data

        '''

        # TODO @Marcel: Kick out the low envegy frequencies, to provide better HPS 


        # HPS
        harmonicProdSpectrum = self.getHarmonicProdSpectrum(fftValues)
        

        # TODO @Marcel: Better methodology for scaling these intuitively using scipy? After stabalized...

        # NOTE: Divisision by # HPS because 

        # Divide by the number of harmonic product spectrums performed!
        # ... due to the fact that by the end, we've shifted the pitches meaning by that much. 

        highestFFT_idx = np.argmax( harmonicProdSpectrum )
        highestFFT_freq = highestFFT_idx * (self.streamSampleRate / self.windowSizeSamps) / self.harmonicProdSpectrums       

        print("Highest FFT Frequency: ", highestFFT_freq)

        # TODO @Marcel: Do this more smartly!!! Temporary hack!
            # We need to detect the frequency profiles on the separate impulses!

        # TODO @Marcel: Time should be based on the curTime + timeOfImpulse!

        if (len(newBufferImpulses) > 0):
            self.valueCallback(highestFFT_freq, 0, self.currentTime)


        return


    def getHarmonicProdSpectrum(self, fftValues):

        # Upsample the data
        # Since HPS is going to use harmonics... 
        # Each time we make a jump in HPS we're effectively stretching the spectrum by the multiplier... 

        # So to accieve the granularity we'd like at the final step of HPS... we'll upsample by # product runs.
            # Meaning, we'll need to interpolate, stepping by (1 / # HPS Runs)

        upsampleInterpValues = np.arange(0, len(fftValues), 1 / self.harmonicProdSpectrums)
        interpIdxRange = np.arange(0, len(fftValues))
        upsampledFftValues = np.interp(upsampleInterpValues, interpIdxRange, fftValues)

        # Re Normalize Spectrum
        upsampledFftValues = upsampledFftValues / np.linalg.norm(upsampledFftValues, ord=2)


        # Now that we've upsampled... time to perform Harmonic Product Analysis!
        harmonicProdSpectrum = copy.deepcopy(upsampledFftValues)

        for i in range(self.harmonicProdSpectrums):

            # NOTE: This loop downSamples, until it reaches the original size
            # The initial upsampling was to allow for the first, hyper-stretched element wise mult.

            # Get the first  1 / (i+1) th of the spectrum
            subSpectrum = harmonicProdSpectrum [ : int(np.ceil( len(upsampledFftValues) / (i+1) ) )]

            # Get the entire ORIGINAL spectrum, interpolated to the same size
            fullSpectrum = upsampledFftValues[::(i+1)]

            # Multiply the full spectrum by the 
            updatedSpec = np.multiply(subSpectrum, fullSpectrum)

            if not any(updatedSpec):
                break

            harmonicProdSpectrum = updatedSpec

        # TODO @Marcel: There's an off case if we break out of the above loop. The indexing thing below wont work!


        return harmonicProdSpectrum



    def detectImpulses(self):
        '''
            Run analysis on the current windowBuffer, and return the idx of any detected impulses

        '''

        # Absolute value
        absWindowBuffer = abs(self.windowBuffer)

        # Convert SubKernel W / StepSize to Samples
        kernelWidthSamps = IMPULSE_KERNEL_W_T * self.streamSampleRate
        kernelStepSamps = IMPULSE_KERNEL_STEP_SIZE * self.streamSampleRate

        # Get number of kernel steps : what kernel is the last one before the end overflows? 
        kernelsNeeded = math.floor( (len(absWindowBuffer) - kernelWidthSamps) / kernelStepSamps )

        # print("Kernel Width: ", kernelWidthSamps)
        # print("Kernel Step: ", kernelStepSamps)
        # print("Window Width: ", len(absWindowBuffer))
        # print("Kernels Needed: ", kernelsNeeded)

        kernelImpulseIdxs = [ ]

        # Move through window buffer...
        for i in range(kernelsNeeded-1):

            # Kernels
            kernel1 = self.windowBuffer[ int(i*kernelStepSamps) : int(i*kernelStepSamps + kernelWidthSamps) ]
            kernel2 = self.windowBuffer[ int((i+1)*kernelStepSamps) : int((i+1)*kernelStepSamps + kernelWidthSamps) ]

            # Get current kernel power
            kernel1Power = (np.linalg.norm(kernel1, ord=2)**2) / len(kernel1)

            # Get next kernel power
            kernel2Power = (np.linalg.norm(kernel2, ord=2)**2) / len(kernel2)

            print("\nPower: ", kernel1Power, " => ", kernel2Power, '\n')

            # Power Jump? 
            if ( (kernel2Power / kernel1Power) > IMPULSE_POWER_MULT_THRESHOLD ):
                print("\n------ IMPULSE! ------\n: ", i+1, " / ", kernelsNeeded)
                kernelImpulseIdxs.append(i+1)

            elif ( (kernel2Power - kernel1Power) > IMPULSE_POWER_DELTA_THRESHOLD ): 
                print('\n ------ STEP IMPULSE! ------ \n : ', i+1, " / ", kernelsNeeded)
                kernelImpulseIdxs.append(i+1)
        

        # Map to buffer idxs... and remove any kernel impulses that are back to back! This means contiual growth of power.
        bufferImpulseIdxs = [ ]
        for i in range(len(kernelImpulseIdxs)):

            kernelIdx = kernelImpulseIdxs[i]

            # Sequential Power Check
            if ( i < len(kernelImpulseIdxs) - 1):
                if ( kernelIdx == kernelImpulseIdxs[i+1] - 1):
                    continue

            bufferImpulseIdxs.append( (kernelIdx * kernelStepSamps ))

        # print("Buffer Impulse Idxs: ", bufferImpulseIdxs)

        return bufferImpulseIdxs
    

    def updateWindowWithBuffer(self, indata, frame_count):

        # Move the buffer along! 

        # Unpack buffer data
        bufferData = np.asarray ( indata[:, 0] )

        # Add new samples to the end of the window
        self.windowBuffer = np.concatenate((self.windowBuffer, bufferData))

        # Trim any tailing samples, to maintain buffer len
        self.windowBuffer = self.windowBuffer[ len(bufferData) : ]


        return



