from obspy.clients.iris import Client
from obspy import UTCDateTime
import matplotlib.pyplot as plt

dt = UTCDateTime("2005-01-01T00:00:00")
client = Client()
st = client.timeseries("IU", "ANMO", "00", "BHZ", dt, dt+10000)

#st.plot(color='purple', tick_format='%I:%M %p', starttime = st[0].stats.starttime, endtime = st[0].stats.endtime)
st[0].spectrogram(log=True)