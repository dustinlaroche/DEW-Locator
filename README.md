# Directed Energy Weapon(DEW) Locator
A project to find the simplest and cheapest way to detect the signals of Directed Energy Weapons so they can be tracked and signals demodulated.

Beginning hardware platform consist of an Adafruit ESP32-C3 that is running ESP-IDF code (<a href="https://github.com/dustinlaroche/ESP32C3_Dual_Wifi">Repository</a>). In addition, an LM324-N buffer opamp configured with one opamp as inverting, and another as non-inverting. The inputs to the LM324-N are then connected to two different SMA snub antennas. The hardware works great at sampling RF signals from -1MHz to 1MHz...  :grimacing:

The code configures the two ADC input pins to utilize the FIR filter with 64 coeffecients and a continous sample rate of 1MHz. I have the MCU output the ADC values to the USB serial as two comma seperated values. These get saved onto a computer where Python is then used to process them through their own FFT functions that are then plotted to individual plots.

<h1>Hardware</h1>

<img src="https://github.com/dustinlaroche/DEW-Locator/blob/main/Hardware/20240601_132526.jpg" />

<img src="https://github.com/dustinlaroche/DEW-Locator/blob/main/Plots/Figure_1.png" />

<h1>Upsampled to 1GHz</h1>

<img src="https://github.com/dustinlaroche/DEW-Locator/blob/main/Plots/Python_1GHz_Plot.png" />
