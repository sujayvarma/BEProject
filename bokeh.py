# -*- coding: utf-8 -*-
"""bokeh.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1OAxYRgeE3atzrkVl-RJX7vnb7CrOcyH3
"""

'''!pip install PyDrive

import os
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)'''

import pandas as pd
from bokeh.plotting import figure
from bokeh.io import output_file, show


'''File = drive.CreateFile({'id':'1h4vcJLWWcGxlcpvLXCLIXooRPtdNG9Sy'})
File.GetContentFile('All_Labelled_Reviews.csv')

df = pd.read_csv('All_Labelled_Reviews.csv')
'''

from bokeh.models import ColumnDataSource

reviews = pd.read_csv("/home/dipen/Downloads/All_Labelled_Reviews.csv")
reviews['category_id'] = reviews['Category'].factorize()[0]
reviews = reviews[reviews.category_id != 3]
reviews.reset_index(drop = True, inplace = True)
category_id_df = reviews[['Category', 'category_id']].drop_duplicates().sort_values('category_id')
id_to_category = dict(category_id_df[['category_id', 'Category']].values)
category_to_id = dict(category_id_df.values)
reviews.tail()

def get_counts(drug_name):
  grouped = reviews.groupby('Drug')
  grouped_drug = grouped.get_group(drug_name)
  grouped_drug_adverse = grouped_drug.groupby('Category').get_group('Adverse').category_id.count()
  grouped_drug_effective = grouped_drug.groupby('Category').get_group('Effective').category_id.count()
  grouped_drug_ineffective = grouped_drug.groupby('Category').get_group('Ineffective').category_id.count()
  counts_tuple = {'Drug' : drug_name, 'Adverse' : grouped_drug_adverse, 'Effective' :grouped_drug_effective, 'Ineffective': grouped_drug_ineffective}
  counts_df = pd.DataFrame(counts_tuple, index = [0])
  return counts_df

etho_df = get_counts('Ethosuximide')
pheno_df = get_counts('Phenytoin')
diva_df = get_counts('Divalproex')
valp_df = get_counts('Valporic Acid')
carb_df = get_counts('Carbamazepine')
aceta_df = get_counts('Acetazolamide')
ph_df = get_counts('Phenobarbital')
counts_df = pd.concat([etho_df, pheno_df, diva_df, valp_df, carb_df, aceta_df, ph_df], ignore_index = True)
counts_df = counts_df.reindex(['Drug', 'Adverse', 'Effective', 'Ineffective'], axis = 1)
counts_df

bars = ['Adverse', 'Effective', 'Ineffective']

data = {'Drug' : counts_df.Drug,
        'Adverse'   : counts_df.Adverse,
        'Effective'   : counts_df.Effective,
        'Ineffective'   : counts_df.Ineffective}


x = [ (drug, bar) for drug in counts_df.Drug for bar in bars ]
counts = sum(zip(data['Adverse'], data['Effective'], data['Ineffective']), ()) # like an hstack

# Import the ColumnDataSource class
from bokeh.models import FactorRange
# Convert dataframe to column data source
src = ColumnDataSource(data=dict(x=x, counts = counts))
src.data.keys()

from bokeh.transform import factor_cmap
from bokeh.palettes import Spectral6

p = figure(plot_height = 600, plot_width = 800, 
           title = 'Drug Classification',
          x_axis_label = 'Drug Name', 
           y_axis_label = 'Number of Reviews',
          x_range=FactorRange(factors=list(x)),toolbar_location=None, tools="hover"
)

p.vbar(source = src, bottom=0, top='counts', x='x', width=0.9, line_color="white",
       # use the palette to colormap based on the the x[1:2] values
       fill_color=factor_cmap('x', palette=Spectral6, factors=bars, start=1, end=2), fill_alpha = 0.75,
       hover_fill_alpha = 1.0, hover_fill_color = 'navy')


p.y_range.start = 0
p.x_range.range_padding = 0.1
p.xaxis.major_label_orientation = 1
p.xgrid.grid_line_color = None
output_file('new.html')

show(p)

from bokeh.layouts import widgetbox
from bokeh.models.widgets import CheckboxGroup
liste = ['Ethosuximide','Phenytoin','Divalproex','Valporic Acid','Carbamazepine','Acetazolamide','Phenobarbital']
drug_selection = CheckboxGroup(labels=liste, active = [0, 1])

def make_dataset(drug_list,bin_width = 5):

    by_drug = pd.DataFrame(columns=['Adverse', 'Effective', 'Ineffective','drug_name'])
    
    # Iterate through all the drugs
    for i, drug_name in enumerate(drug_list):
        m = counts_df.loc[counts_df['Drug'] == drug_name]
        arr_df = pd.DataFrame(columns=['Adverse', 'Effective', 'Ineffective','drug_name'])
        arr_df['Adverse'] = m['Adverse']
        arr_df['Effective'] = m['Effective']
        arr_df['Ineffective'] = m['Ineffective']
        arr_df['name'] = drug_name
        by_drug = by_drug.append(arr_df)

    # Overall dataframe
    bars = ['Adverse', 'Effective', 'Ineffective']
    x = [ (drug, bar) for drug in by_drug.name for bar in bars ].sort()
    by_drug = by_drug.sort_values(['name'])
    counts = sum(zip(by_drug['Adverse'], by_drug['Effective'], by_drug['Ineffective']), ()) # like an hstack

    # Convert dataframe to column data source
    return ColumnDataSource(data = dict(x = x, counts = counts))

def make_plot(src):
        # Blank plot with correct labels
        p = figure(plot_width = 700, plot_height = 700, 
                  title = 'Int Drug',
                  x_axis_label = 'Drug', y_axis_label = 'Reviews',x_range=FactorRange(factors=list(x)))

        p.vbar(source = src, bottom=0, top='counts', x='x', width=0.9,
               color =factor_cmap('x', palette=Spectral6, factors=bars, start=1, end=2), fill_alpha = 0.75, line_color = 'black', legend = 'x')
        p.y_range.start = 0
        p.x_range.range_padding = 0.1
        p.xaxis.major_label_orientation = 1
        p.xgrid.grid_line_color = None        
        return p
      
# Update function takes three default parameters
def update(attr, old, new):
    # Get the list of drugs for the graph
    drugs_to_plot = [drug_selection.labels[i] for i in drug_selection.active]
    # Make a new dataset based on the selected drugs and the 
    # make_dataset function defined earlier
    new_src = make_dataset(drugs_to_plot,bin_width = 5)
    
    
    # Update the source used in the quad glpyhs
    src.data.update(new_src.data)
    
drug_selection.on_change('active', update)
show(widgetbox(drug_selection))
output_file('new.html')

init_drugs = [drug_selection.labels[i] for i in drug_selection.active]
src = make_dataset(init_drugs,bin_width= 5)
p = make_plot(src)
output_file('new.html')
show(p)

from bokeh.layouts import column, row, WidgetBox
from bokeh.models import Panel
from bokeh.io import show, curdoc
from bokeh.models.widgets import Tabs
# Put controls in a single element
controls = WidgetBox(drug_selection)
    
# Create a row layout
layout = row(controls, p)
    
# Make a tab with the layout 
tab = Panel(child=layout, title = 'Drug')
tabs = Tabs(tabs=[tab])
curdoc().add_root(tabs)