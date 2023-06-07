import streamlit as st
from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


from matplotlib import colors as mcolors


def get_darker_shades(base_color, n):
    base_color_rgb = mcolors.hex2color(base_color)
    base_color_hsv = mcolors.rgb_to_hsv(base_color_rgb)
    shades = [mcolors.hsv_to_rgb([base_color_hsv[0], base_color_hsv[1], max(0, base_color_hsv[2] - i * 0.05)]) for i in range(n)]
    return [mcolors.rgb2hex(shade) for shade in shades]

data = genfromtxt('/content/AllBacteria.csv', delimiter=',',dtype='str')
data = np.char.lower(data) #convert to lowercase

new_species_indices = [i for i, item in enumerate(data[0,:]) if item.startswith('name')]

num_samples = 0

dataset = {}
wavelengths = data[1:,2].astype(float)

for idx,new_start in enumerate(new_species_indices):
  #extract end point of the data block
  if(idx == len(new_species_indices) - 1):
    end = data.shape[1] #last column
  else:
    end = new_species_indices[idx+1]

  #extract concentration
  if(data[0,new_start+1] == 'concentration' or 'conc.'):
    concentration = data[1,new_start+1]
  else:
    raise ValueError(f"Expected 'concentration' at position {new_start+1}, but found {data[0,new_start+1]} instead.")

  bacteria_name = data[1,new_start]
  data_block = data[:,new_start:end]

  print(bacteria_name, end = " {")
  print(concentration, end = "}\n")

  #add new_entry for the bacteria if not in dict:
  if(bacteria_name not in dataset.keys()):
    dataset[bacteria_name] = {}
  
  #add new array within concentration level is not added previously
  if(concentration not in dataset[bacteria_name].keys()):
    dataset[bacteria_name][concentration] = []

  measurement_idxs = [i for i, item in enumerate(data[0,new_start:end]) if 'meas' in item]

  # print(data_block)
  for m_idx in measurement_idxs:
    measurement = data_block[1:, m_idx]

    dataset[bacteria_name][concentration].append(measurement)
    num_samples += 1

data_dict = dataset

data_dict = dataset

bacteria_options = list(data_dict.keys())
selected_bacteria = st.selectbox('Select a bacteria', bacteria_options)

concentration_options = list(data_dict[selected_bacteria].keys())

# Base colors for the concentrations (you might want to add more colors here if there are more concentrations)
base_colors = ['#0000ff', '#ff0000', '#008000', '#800080', '#ffa500']

# Create a plotly graph
fig = go.Figure()

# For each concentration, add a trace with a color that is a shade of the base color
for i, concentration in enumerate(concentration_options):
    # Access the waveform data
    waveforms = data_dict[selected_bacteria][concentration]
    color_shades = get_darker_shades(base_colors[i % len(base_colors)], len(waveforms))
    
    for j, waveform in enumerate(waveforms):
        color = color_shades[j]
        fig.add_trace(go.Scatter(x=wavelengths, y=waveform.astype(float), mode='lines', line=dict(color=color), name=f'{concentration} Measurement {j+1}'))

fig.update_layout(
    title=f'{selected_bacteria} Concentrations',
    xaxis_title='Wavelength',
    yaxis_title='Value',
    autosize=False,
    width=800,
    height=500,
)

st.plotly_chart(fig, use_container_width=True)

# concentration_options = list(data_dict[selected_bacteria].keys())

# for concentration in concentration_options:

#   with st.expander(f"Concentration {concentration}"):

#     # Access the waveform data
#     waveforms = data_dict[selected_bacteria][concentration]

#     st.write(len(waveforms))

#     # Create a plotly graph
#     fig = go.Figure()

#     # For each waveform, add a trace
#     for i, waveform in enumerate(waveforms):
#         fig.add_trace(go.Scatter(x=wavelengths, y=waveform.astype(float), mode='lines', name=f'Measurement {i+1}'))

#     fig.update_layout(
#         title=f'{selected_bacteria} at {concentration}',
#         xaxis_title='Wavelength',
#         yaxis_title='Value',
#         autosize=False,
#         width=800,
#         height=500,
#     )

#     st.plotly_chart(fig, use_container_width=True)
