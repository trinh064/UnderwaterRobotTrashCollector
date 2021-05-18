import RPi.GPIO as GPIO
import time
# Set GPIO numbering mode
GPIO.cleanup()
GPIO.setmode(GPIO.BOARD)

# Set pin 11 as an output, and set servo1 as pin 11 as PWM
GPIO.setup(11,GPIO.OUT)
GPIO.setup(12,GPIO.OUT)
pin=0
def up():
        GPIO.output(11,1)
        GPIO.output(12,0)
        time.sleep(0.5)
        stay()

def down():
        GPIO.output(11,0)
        GPIO.output(12,1)
        time.sleep(0.5)
        stay()


def stay():
        GPIO.output(11,0)
        GPIO.output(12,0)

'''
while(1):
	#servo1.ChangeDutyCycle(duty)
	#duty=(duty+10)%100
        down()
        time.sleep(2)
        up()
        time.sleep(2)
	#GPIO.output(31,1)
	#time.sleep(0.5)
        print('loop')
'''
