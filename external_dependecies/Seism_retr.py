from obspy import UTCDateTime, read
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from scipy.signal import spectrogram

dt = UTCDateTime("2014-05-25T06:10:00")
dt_end = UTCDateTime("2014-06-04T00:00:00")
#client = Client()
#st = client.timeseries("IU", "ANMO", "00", "BHZ", dt, dt_end)
#st.taper(0.05)
st = read("Datasets/quake.mseed")

st.plot(color='purple', tick_format='%I:%M %p', starttime = st[0].stats.starttime, endtime = st[0].stats.endtime, outfile='quak.png')
#st[0].spectrogram(log=True, outfile='spectrogram.png')

tr = st[0]
data = tr.data
fs = 1 / tr.stats.delta 

# Compute the spectrogram
frequencies, times, Sxx = spectrogram(data, fs, nperseg=8, noverlap=4)
n_bands = 32
min_freq = 1  # Minimum frequency of interest
max_freq = 20  # Maximum frequency of interest

# Create 32 frequency bands between min_freq and max_freq
band_edges = np.linspace(min_freq, max_freq, n_bands + 1)
print(data.shape)

# Initialize an array to store energy in each frequency band
band_energies = np.zeros((n_bands, len(times)))

for i in range(n_bands):
    band_mask = (frequencies >= band_edges[i]) & (frequencies < band_edges[i + 1])
    band_energies[i, :] = np.mean(Sxx[band_mask, :], axis=0)

for i in range(n_bands):
    plt.plot(times, band_energies[i, :], label=f'Band {i+1} ({band_edges[i]:.2f}-{band_edges[i+1]:.2f} Hz)')

parquet = pd.DataFrame(band_energies).T
print(parquet.shape)
parquet.to_parquet("Datasets/quake.parquet")

plt.legend(ncol=2, fontsize='small')
plt.xlabel('Time (s)')
plt.ylabel('Energy')
plt.title('Energy in 32 Frequency Bands Over Time')
plt.tight_layout()
plt.savefig("Datasets/energy.svg", format='svg')

#df = pd.DataFrame(band_energies)
#df.to_csv("Datasets/earthquake.csv", header=False, index=False)

