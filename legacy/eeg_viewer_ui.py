import streamlit as st
import mne
import matplotlib.pyplot as plt
from bids_eeg_viewer import BIDS_EEG_Viewer

st.set_page_config(layout="wide")
st.title("🧠 BIDS EEG Visualization App")

# Initialize viewer and cache it
@st.cache_resource

def init_viewer():
    return BIDS_EEG_Viewer(data_dir="/mount/sub/cmi_bids_R1/eeg")

viewer = init_viewer()

# Sidebar - Subject and Task Selection
st.sidebar.header("Subject and Task Selection")
subject = st.sidebar.selectbox("Select Subject", viewer.subjects)
viewer.set_subject(subject)

tasks = viewer.get_available_tasks()
task = st.sidebar.selectbox("Select Task", tasks)
run_options = [info['run'] for info in viewer.task_info if info['task'] == task]
run = st.sidebar.selectbox("Select Run (optional)", [None] + run_options)

# Controls
st.sidebar.header("Visualization Options")
show_metadata = st.sidebar.checkbox("Show Metadata", value=True)
show_events = st.sidebar.checkbox("Show Events", value=True)
show_channels = st.sidebar.checkbox("Show Channels", value=False)
show_electrodes = st.sidebar.checkbox("Show Electrodes", value=False)

st.sidebar.header("EEG Plotting")
plot_time = st.sidebar.checkbox("Time Domain Plot", value=True)
plot_psd = st.sidebar.checkbox("Frequency Domain (PSD)", value=True)

if st.sidebar.button("🔍 Load and Plot"):
    raw = viewer.get_raw(task, run)

    # Display metadata and tables
    if show_metadata:
        st.subheader("Metadata")
        st.json(viewer.get_metadata(task, run))
    if show_events:
        st.subheader("Event Table")
        st.dataframe(viewer.get_event_data(task, run))
    if show_channels:
        st.subheader("Channel Info")
        st.dataframe(viewer.get_channel_data(task, run))
    if show_electrodes:
        st.subheader("Electrode Coordinates")
        st.dataframe(viewer.get_electrode_data(task, run))

    # Time domain plot
    if plot_time:
        st.subheader("Time Domain")
        fig_time = raw.plot(n_channels=10, show=False, scalings='auto', title=f"{subject} - {task}")
        st.pyplot(fig_time)

    # Frequency domain (PSD)
    if plot_psd:
        st.subheader("Power Spectral Density")
        psd = raw.compute_psd(fmax=60)
        psd_clean = psd.copy().pick([ch for ch, d in zip(psd.info['ch_names'], psd.get_data()) if d.any() and all(np.isfinite(d))])
        fig_psd = psd_clean.plot(average=True, dB=True, spatial_colors=False, show=False)
        st.pyplot(fig_psd)
